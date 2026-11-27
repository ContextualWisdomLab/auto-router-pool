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
