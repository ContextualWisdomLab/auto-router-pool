"""Validity, scoring and role selection for the auto-router pool."""
from __future__ import annotations

import hashlib
import json

from .assign import cost

ROUTING_POLICY_VERSION = 2


def fingerprint(rows: list[dict], names: list[str]) -> str:
    """Watch identity for the catalogue, blocked flags and routing policy."""
    blocked = sorted(
        (row["model_name"], bool((row.get("model_info") or {}).get("blocked")))
        for row in rows
    )
    payload = json.dumps([ROUTING_POLICY_VERSION, sorted(names), blocked]).encode()
    return hashlib.sha256(payload).hexdigest()


def score_hashes(cache: dict) -> dict[str, str]:
    """Hash only the numbers, not the fetch timestamp, so a re-read that returns
    identical values reports no change."""
    return {
        slug: hashlib.sha256(
            json.dumps(entry["variants"], sort_keys=True).encode()
        ).hexdigest()
        for slug, entry in cache.items()
    }


def changed_scores(hashes: dict[str, str], previous: dict[str, str]) -> list[str]:
    return sorted(
        slug for slug, digest in hashes.items() if previous.get(slug) != digest
    )


# Operators only route long-context traffic, so a group below this window is not
# a candidate: it would either be filtered out per request or truncate silently.
MIN_INPUT_TOKENS = 272_000


def input_window(rows: list[dict], name: str) -> int:
    """Largest declared input window in a group; 0 when nothing is declared."""
    values = [
        (row.get("model_info") or {}).get("max_input_tokens")
        for row in rows
        if row["model_name"] == name
    ]
    known = [value for value in values if type(value) is int and value > 0]
    return max(known) if known else 0


def group_state(rows: list[dict], name: str, minimum: int = MIN_INPUT_TOKENS):
    """``(usable, reason)`` based on registration, blocked flags and context."""
    group = [row for row in rows if row["model_name"] == name]
    if not group:
        return False, "unregistered"
    if all((row.get("model_info") or {}).get("blocked") for row in group):
        return False, "all_blocked"
    if input_window(rows, name) < minimum:
        return False, "context_below_floor"
    return True, "registered"


def flaky_deployments(samples: dict[str, list[str]]) -> list[str]:
    """Deployments that both answered and rate-limited in the sample window.

    A consistently failing deployment is not flaky: blocking it would remove a
    deployment that is merely down, and it may recover without intervention.
    """
    return sorted(
        model_id
        for model_id, states in samples.items()
        if "ok" in states and "rate_limited" in states
    )


def stable_deployments(samples: dict[str, list[str]]) -> set[str]:
    """Deployments that answered every sample; only these can absorb traffic."""
    return {
        model_id
        for model_id, states in samples.items()
        if states and all(state == "ok" for state in states)
    }


def score(variants: dict, effort: str):
    """``(index, index per blended dollar, entry)`` for one effort."""
    entry = variants.get(effort) or variants.get("max")
    if not entry or not entry.get("index") or not entry.get("in") or not entry.get("out"):
        return None
    blended = (entry["in"] + entry["out"]) / 2
    return entry["index"], entry["index"] / blended, entry


def declares_free(row: dict) -> bool:
    """The deployment says it is free, in words rather than in numbers.

    OpenRouter free tiers carry a ``:free`` upstream suffix; an operator can also
    tag a deployment ``is_free`` or ``free`` in ``model_info``.
    """
    info = row.get("model_info") or {}
    upstream = str((row.get("litellm_params") or {}).get("model") or "").lower()
    tags = {str(tag).lower() for tag in (info.get("tags") or [])}
    return upstream.endswith(":free") or info.get("is_free") is True or "free" in tags


def free_models(rows: list[dict]) -> set[str]:
    """Model groups that cost nothing, judged by an explicit free marker.

    Cost fields cannot answer this in either direction, which is why they are not
    consulted: a registration that was never priced declares zero, and a free
    tier may still carry a price. Only a marker counts, and a group qualifies
    only when *every* deployment the router could reach carries it -- one paid
    sibling makes the group paid, because the router may reach that one.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["model_name"], []).append(row)

    free = set()
    for name, deployments in groups.items():
        live = [
            row for row in deployments
            if not (row.get("model_info") or {}).get("blocked")
        ] or deployments
        if all(declares_free(row) for row in live):
            free.add(name)
    return free


# A decision model answers a schema of typed questions in one forward pass and
# returns calibrated probabilities. It is not a chat or streaming model, so it
# can never fill a routing tier however cheap it is.
DECISION_MODES = {"decision", "decisions", "evaluation"}
DECISION_ENDPOINTS = ("/v1/decisions", "/decisions", "/v1/systemone", "/systemone")
DECISION_WORDS = ("decision model",)


def is_decision(row: dict) -> bool:
    """The registration says this model decides rather than chats.

    A chat model can also expose a decisions endpoint (``gpt-6-luna`` lists
    ``/v1/chat/completions`` and ``/v1/decisions``), so an endpoint alone proves
    nothing: the decision endpoints have to be the model's whole surface.
    """
    info = row.get("model_info") or {}
    if str(info.get("mode") or "").lower() in DECISION_MODES:
        return True
    endpoints = [str(e).lower() for e in (info.get("supported_endpoints") or [])]
    if any(marker in " ".join(endpoints) for marker in DECISION_ENDPOINTS) and not any(
        "chat/completions" in endpoint for endpoint in endpoints
    ):
        return True
    text = f"{info.get('description') or ''} {info.get('availability') or ''}".lower()
    return any(word in text for word in DECISION_WORDS) and "not a chat" in text


def decision_models(rows: list[dict]) -> list[str]:
    """Groups that are decision models, reported instead of looking broken.

    They are legitimate to register and cheap to run, but a chat tier can never
    use one: the request shape is a decision schema, not a conversation. Naming
    them keeps them out of the "blocked model" bucket and out of every tier.
    """
    return sorted(
        name for name in {row["model_name"] for row in rows}
        if any(is_decision(row) for row in rows if row["model_name"] == name)
    )


def catalog_entry(upstream: str, catalog: dict | None) -> dict | None:
    """The external catalogue's row for a deployment's upstream model string.

    Registration metadata is written by whoever added the deployment, so an
    optimistic ``/v1/decisions`` in it is a claim, not a fact. The catalogue is
    the reference: it carries ``mode: evaluation`` for Jev and chat for the
    OpenRouter router, and that difference is exactly what decides whether a
    deployment can classify.
    """
    if not catalog or not upstream:
        return None
    for key in (upstream, upstream.removesuffix(":free")):
        if key in catalog:
            return catalog[key]
    # A route may be registered under a provider prefix the catalogue spells
    # differently (``openai/gpt-x`` vs ``gpt-x``): match on the model tail.
    tail = upstream.removesuffix(":free").split("/")[-1]
    for key, entry in catalog.items():
        if key.split("/")[-1] == tail:
            return entry
    return None


def supports_decisions(row: dict, catalog: dict | None = None) -> bool:
    """The deployment can answer a decision request, chat or not.

    ``is_decision`` answers a different question: whether the model is *only* a
    decision model, which is what keeps one out of every tier. A chat model that
    also exposes ``/v1/decisions`` (``gpt-6-luna``) is a perfectly good
    classifier, so the capability has to be read separately from the exclusion.

    When the external catalogue knows the upstream model, its verdict wins: a
    ``chat`` entry is not decision-capable however many endpoints the
    registration lists, and an ``evaluation``/``decision`` entry is.
    """
    info = row.get("model_info") or {}
    entry = catalog_entry((row.get("litellm_params") or {}).get("model") or "", catalog)
    if entry is not None:
        if str(entry.get("mode") or "").lower() in DECISION_MODES:
            return True
        endpoints = [str(e).lower() for e in (entry.get("supported_endpoints") or [])]
        return any(marker in " ".join(endpoints) for marker in DECISION_ENDPOINTS)
    if str(info.get("mode") or "").lower() in DECISION_MODES:
        return True
    endpoints = [str(e).lower() for e in (info.get("supported_endpoints") or [])]
    if any(marker in " ".join(endpoints) for marker in DECISION_ENDPOINTS):
        return True
    text = f"{info.get('description') or ''} {info.get('availability') or ''}".lower()
    return any(word in text for word in DECISION_WORDS) and "not a chat" in text


def decision_catalog(catalog: dict | None) -> list[dict]:
    """Upstream models the catalogue itself marks as decision-capable.

    This is how a supported classifier is *found* before anything is registered:
    the catalogue's ``mode`` is the reference, so a deployment only has to exist
    for the model to be usable.
    """
    out = []
    for key, entry in sorted((catalog or {}).items()):
        mode = str((entry or {}).get("mode") or "").lower()
        if mode not in DECISION_MODES:
            continue
        out.append({
            "model": key,
            "mode": mode,
            "max_input_tokens": (entry or {}).get("max_input_tokens"),
        })
    return out


def classifier_candidates(rows: list[dict], catalog: dict | None = None) -> list[dict]:
    """Every group that can answer a decision request, and whether it can now.

    A decision model is reported as ``exclusive``; a chat model that also serves
    ``/v1/decisions`` is reported too, because it can classify even though it
    must never fill a tier.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["model_name"], []).append(row)
    only = set(decision_models(rows))
    out = []
    for name, group in sorted(groups.items()):
        if not any(supports_decisions(row, catalog) for row in group):
            continue
        out.append({
            "model": name,
            "exclusive": name in only,
            "usable": any(
                not (row.get("model_info") or {}).get("blocked") for row in group
            ),
        })
    return out


def decision_classifier(rows: list[dict], catalog: dict | None = None) -> str | None:
    """A model this gateway can actually use as its decision classifier.

    The router asks one question -- which tier -- and a decision endpoint answers
    that in one cheap forward pass instead of a chat completion. Models that are
    only decisions come first; a chat model that also serves decisions is the
    second choice. The 272k solver floor does not apply: the classifier reads a
    short prompt, not the request, so its window cannot bound what the router
    advertises.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["model_name"], []).append(row)
    only = set(decision_models(rows))
    capable = [
        name for name, group in groups.items()
        if any(supports_decisions(row, catalog) for row in group)
    ]
    for name in sorted(capable, key=lambda name: (name not in only, name)):
        if any(not (row.get("model_info") or {}).get("blocked") for row in groups[name]):
            return name
    return None


def free_gaps(rows: list[dict], scored) -> list[dict]:
    """Free groups the router cannot use yet, each with the reason.

    A free group only earns its place in a chain once the router can actually
    select it, so a group that is registered but blocked, below the context
    floor, or simply unmeasured would otherwise disappear without a trace.
    This reports them instead: the model name and why it is not in the pool.
    """
    free = free_models(rows)
    if not free:
        return []
    known = set(scored or ())
    gaps = []
    for name in sorted(free):
        if name in known:
            continue
        usable, reason = group_state(rows, name)
        gaps.append({"model": name, "reason": "unmeasured" if usable else reason})
    return gaps


def declared_effort(rows: list[dict], name: str) -> str | None:
    """The effort this group runs at when a request names none.

    Only a deployment-level ``reasoning_effort`` states it. Sibling deployments
    that disagree do not describe one default, so nothing is claimed then.
    """
    values = {
        (row.get("litellm_params") or {}).get("reasoning_effort")
        for row in rows if row["model_name"] == name
    }
    values.discard(None)
    values.discard("")
    return values.pop() if len(values) == 1 else None


def effort_floor(variants: dict) -> float | None:
    """Index at the weakest measured effort.

    A request that names no effort may be served at any rung, so the lowest one
    is the quality the router can actually promise on its default path.
    """
    indexes = [
        entry.get("index") for entry in (variants or {}).values()
        if entry.get("index")
    ]
    return min(indexes) if indexes else None


def pick_default(rows: list[dict], candidates: dict[str, dict], preferred) -> tuple:
    """Fallback model for a request the classifier cannot place in a tier.

    The default path carries no ``tier_model_configs`` entry, so it runs at the
    model's own default effort instead of the rung its best score was measured
    at. A declared deployment default is scored at that rung; without one the
    model is scored at its measured floor, which is what the router can promise
    there. Ties keep the caller's preference order.
    """
    ranked = []
    for order, name in enumerate(preferred):
        variants = candidates.get(name) or {}
        effort = declared_effort(rows, name)
        entry = variants.get(effort) if effort else None
        if entry and entry.get("index"):
            index, basis = entry["index"], "declared"
        else:
            index, basis, effort = effort_floor(variants), "floor", None
        ranked.append((index or 0, -order, name, basis, effort))
    best = max(ranked)
    return best[2], {
        "basis": best[3],
        "effort": best[4],
        "index": round(best[0], 1) if best[0] else None,
    }


def pick_micro(candidates: dict[str, dict], taken=(), limit: int = 2, effort: str = "low"):
    """Cheapest remaining candidates, so light tiers need no supplied list.

    A light tier exists to be cheap and fast, so the small models are the ones
    the pool already proved usable and that cost least for an extraction-sized
    run. Input and output prices enter separately (``assign.cost``): a light
    tier is mostly prompt, so a low output price alone proves nothing.
    """
    skip = set(taken)
    priced = []
    for name, variants in candidates.items():
        if name in skip:
            continue
        result = score(variants, effort)
        if not result:
            continue
        index, _, entry = result
        priced.append((name, cost(entry, "extract"), index))
    priced.sort(key=lambda item: (item[1], -item[2]))
    return [name for name, _, _ in priced[:limit]]
