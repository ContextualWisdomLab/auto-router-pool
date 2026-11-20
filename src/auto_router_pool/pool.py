"""Validity, scoring and role selection for the auto-router pool."""
from __future__ import annotations

import hashlib
import json


def fingerprint(rows: list[dict], names: list[str]) -> str:
    """Catalogue identity plus blocked flags.

    Changes when a model is added, removed, blocked or unblocked upstream, which
    is the only reason to spend Artificial Analysis traffic.
    """
    blocked = sorted(
        (row["model_name"], bool((row.get("model_info") or {}).get("blocked")))
        for row in rows
    )
    payload = json.dumps([sorted(names), blocked], sort_keys=True).encode()
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


def group_state(rows: list[dict], name: str):
    """``(usable, reason)`` based on registration and blocked flags only."""
    group = [row for row in rows if row["model_name"] == name]
    if not group:
        return False, "unregistered"
    if all((row.get("model_info") or {}).get("blocked") for row in group):
        return False, "all_blocked"
    return True, "registered"


def score(variants: dict, effort: str):
    """``(index, index per blended dollar, entry)`` for one effort."""
    entry = variants.get(effort) or variants.get("max")
    if not entry or not entry.get("index") or not entry.get("in") or not entry.get("out"):
        return None
    blended = (entry["in"] + entry["out"]) / 2
    return entry["index"], entry["index"] / blended, entry


def pick(
    candidates: dict[str, dict],
    effort: str,
    max_in: float,
    max_out: float,
    *,
    mode: str = "quality",
    tolerance: float = 2.0,
):
    """Choose a model under a price cap.

    ``quality`` maximises the index and then prefers the cheapest option within
    ``tolerance`` points, which keeps a marginal +0.5 from buying a 4x price.
    ``value`` maximises index per blended dollar.
    """
    scored = []
    for name, variants in candidates.items():
        result = score(variants, effort)
        if not result:
            continue
        index, value, entry = result
        if entry["in"] > max_in or entry["out"] > max_out:
            continue
        scored.append((name, index, value, (entry["in"] + entry["out"]) / 2, entry))
    if not scored:
        return None
    if mode == "value":
        best = max(scored, key=lambda item: item[2])
    else:
        top = max(item[1] for item in scored)
        best = min(
            (item for item in scored if item[1] >= top - tolerance),
            key=lambda item: item[3],
        )
    return best[0], best[2], best[1], best[4]
