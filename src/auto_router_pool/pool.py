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
DECISION_MODES = {"decision", "evaluation"}
DECISION_ENDPOINTS = ("/v1/decisions", "/decisions", "/v1/systemone", "/systemone")
DECISION_WORDS = ("decision model",)


def is_decision(row: dict) -> bool:
    """The registration says this model decides rather than chats."""
    info = row.get("model_info") or {}
    if str(info.get("mode") or "").lower() in DECISION_MODES:
        return True
    endpoints = " ".join(str(e).lower() for e in (info.get("supported_endpoints") or []))
    if any(marker in endpoints for marker in DECISION_ENDPOINTS):
        return True
    text = f"{info.get('description') or ''} {info.get('availability') or ''}".lower()
    return any(word in text for word in DECISION_WORDS)


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
