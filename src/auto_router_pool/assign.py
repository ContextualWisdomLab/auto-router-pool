"""Score-equal placement, and the tier count that placement implies.

A tier's entries are fallbacks for the same request, so the router's choice
between them is only sound if they measure the same. Measured on the Artificial
Analysis index they do not: at their peak efforts the three frontier models of
one roster sit at 62.19, 57.26 and 54.66, so roughly eight index points of the
tier's quality depend on which fallback the router happens to reach.

Placement therefore treats a tier as a small integer program::

    maximise  T
    s.t.      one rung per model              x[m,e] in {0,1}
              index[m,e] >= T                 every entry reaches the target
              max index - min index <= band   entries are interchangeable
    then      minimise price, then effort     cheapest rung that qualifies

The programme is tiny -- one binary per rung of a five-point ladder -- so it is
solved exactly by enumerating ``T`` over the measured ladder points: for a fixed
target a model is placeable iff one of its rungs falls inside the band, and its
own choice is then independent of the others. ``place`` returns the widest chain
of interchangeable entries, then the highest score it can assume.

How many tiers a router needs is the same question asked once more: each tier can
only assume a score its members reach, and members must stay inside ``band`` of
each other, so covering the achievable range with the fewest tiers is an
interval-cover problem. ``schedule`` solves it by the greedy sweep, which is
exact for intervals, and drops tiers whose entry set repeats.
"""
from __future__ import annotations

# Index points a chain may differ by and still be treated as one score. Chosen
# just above the gap between neighbouring rungs (xhigh -> high is ~1.5 on the
# roster this was measured on) so a fallback cannot silently drop a rung.
SPREAD = 3.0


def _rungs(variants: dict) -> list[dict]:
    """``[{effort, index, cost}]`` sorted by measured index."""
    rungs = []
    for effort, entry in (variants or {}).items():
        index = entry.get("index")
        if index is None:
            continue
        cost = (entry.get("in") or 0) + (entry.get("out") or 0)
        rungs.append({"effort": effort, "index": index, "cost": cost})
    return sorted(rungs, key=lambda rung: rung["index"])


def _cheapest(rungs: list[dict], target: float, band: float):
    """Cheapest rung inside ``[target, target + band]``; price ties go lower."""
    inside = [rung for rung in rungs if target <= rung["index"] <= target + band]
    return min(inside, key=lambda rung: (rung["cost"], rung["index"])) if inside else None


def ceiling(variants: dict) -> float | None:
    """Highest measured index of one model, or ``None`` without a ladder."""
    rungs = _rungs(variants)
    return rungs[-1]["index"] if rungs else None


def place(models, ladders: dict, band: float = SPREAD) -> dict:
    """Equal-score placement of ``models`` given their measured ladders.

    Returns ``target`` (the chain's assumed score), ``entries`` as
    ``(model, effort, index)`` in the order the models were given, the realised
    ``spread``, the models ``skipped`` because no rung stayed inside the band,
    and ``unrated`` models that have no measured ladder at all.
    """
    rungs = {model: _rungs(ladders.get(model) or {}) for model in models}
    rated = [model for model in models if rungs[model]]
    unrated = [model for model in models if not rungs[model]]
    empty = {
        "target": None,
        "band": band,
        "entries": [],
        "spread": 0.0,
        "skipped": [],
        "unrated": unrated,
    }
    if len(rated) < 2:
        return empty

    # Every measured rung is a candidate target; the best one places the most
    # models, then assumes the highest score, then the tightest and cheapest set.
    best = None
    for target in sorted({rung["index"] for model in rated for rung in rungs[model]}):
        chosen, cost = [], 0.0
        for model in rated:
            rung = _cheapest(rungs[model], target, band)
            if rung is None:
                continue
            cost += rung["cost"]
            chosen.append((model, rung["effort"], rung["index"]))
        if len(chosen) < 2:
            continue
        indexes = [index for _, _, index in chosen]
        score = (len(chosen), target, -(max(indexes) - min(indexes)), -cost)
        if best is None or score > best[0]:
            best = (score, target, chosen)
    if best is None:
        return {**empty, "skipped": list(rated)}

    _, target, entries = best
    placed = [model for model, _, _ in entries]
    indexes = [index for _, _, index in entries]
    return {
        "target": target,
        "band": band,
        "entries": entries,
        "spread": max(indexes) - min(indexes),
        "skipped": [model for model in rated if model not in placed],
        "unrated": unrated,
    }


def schedule(models, ladders: dict, band: float = SPREAD) -> list[dict]:
    """Fewest score-equal tiers covering every measured quality level.

    Each tier is ``{target, covered, entries}``: the score it assumes, the
    highest score it can serve inside the band, and its interchangeable entries.
    Tiers repeat nothing -- a repeat would mean two names for one score -- so the
    length of the result is the tier count the data supports for this band.
    """
    rungs = {model: _rungs(ladders.get(model) or {}) for model in models}
    levels = sorted({
        rung["index"] for model in models for rung in rungs[model]
    })
    tiers: list[dict] = []
    index = 0
    while index < len(levels):
        target = levels[index]
        entries = []
        for model in models:
            rung = _cheapest(rungs[model], target, band)
            if rung is None:
                continue
            entries.append((model, rung["effort"], rung["index"]))
        if not entries:
            index += 1
            continue
        # A tier serves any score within ``band`` of the one it assumes.
        covered = target + band
        if tiers and tiers[-1]["entries"] == entries:
            tiers[-1]["covered"] = covered
        else:
            tiers.append({"target": target, "covered": covered, "entries": entries})
        while index < len(levels) and levels[index] <= covered:
            index += 1
    return tiers
