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


# Role floors in measured index points, not dollars. A role picks the cheapest
# model that still reaches its floor, so prices are an output of the roster.
PREMIUM_TOL = 1.0
PRIMARY_TOL = 2.0
RUNNER_TOL = 6.0

ROLE_EFFORT = {
    "premium": "xhigh",
    "primary": "high",
    "runner_up": "high",
    "budget": "medium",
}
ROLE_TOL = {"premium": PREMIUM_TOL, "primary": PRIMARY_TOL, "runner_up": RUNNER_TOL}


def _options(rungs: dict, effort: str, skip=()) -> list[tuple]:
    """``(name, rung)`` for every model measured at ``effort``.

    A model whose ladder was only measured at one point still competes: it is
    scored at that rung instead of vanishing from the assignment, which is what
    the single-effort candidates in a thin catalogue need.
    """
    options = []
    for name, choices in rungs.items():
        if name in skip or not choices:
            continue
        rung = next((r for r in choices if r["effort"] == effort), None)
        options.append((name, rung or choices[-1]))
    return options


def _cheapest_above(options: list[tuple], tolerance: float):
    """Cheapest option within ``tolerance`` of the best index; price first."""
    if not options:
        return None
    top = max(rung["index"] for _, rung in options)
    floor = top - tolerance
    here = [(name, rung) for name, rung in options if rung["index"] >= floor]
    return min(here, key=lambda item: (item[1]["cost"], -item[1]["index"]))


def roles(ladders: dict) -> dict[str, tuple | None]:
    """Role assignment as a linear program over the measured ladders.

    ::\n
        minimise   blended price of the four roles
        s.t.       index[role] >= best_available(role) - tolerance
                   one model per role, all roles distinct

    Floors are index points taken from the roster itself, so no price cap is
    ever written down and a price change cannot silently demote a role: the
    cheapest model that still reaches the floor wins, and the floor moves with
    the measurement. ``budget`` has no quality floor -- it maximises index per
    blended dollar, which is the same statement with the price on the other side.
    """
    rungs = {name: _rungs(variants or {}) for name, variants in (ladders or {}).items()}
    chosen: dict[str, tuple | None] = {}

    def solve(role: str, exclude=()):
        return _cheapest_above(
            _options(rungs, ROLE_EFFORT[role], set(exclude)), ROLE_TOL[role]
        )

    def keep(role: str, best):
        chosen[role] = (
            None if best is None else (best[0], None, best[1]["index"], best[1])
        )
        return best[0] if best else None

    premium = keep("premium", solve("premium"))
    # The roles are distinct where the roster allows it: primary must not be the
    # premium model, and runner_up must be neither. A catalogue too thin for that
    # falls back to the previous role instead of failing the whole run.
    primary = keep(
        "primary",
        solve("primary", [premium]) or solve("primary"),
    )
    runner = keep(
        "runner_up",
        solve("runner_up", [premium, primary]) or solve("runner_up", [primary]),
    )

    cheap = [
        (item[1]["index"] / item[1]["cost"], item)
        for item in _options(rungs, ROLE_EFFORT["budget"])
        if item[1]["cost"] > 0
    ]
    best_value = max(cheap, key=lambda pair: (pair[0], pair[1][1]["index"]))[1] if cheap else None
    chosen["budget"] = (
        None if best_value is None
        else (best_value[0], None, best_value[1]["index"], best_value[1])
    )
    return chosen


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


def grow(
    tier_map: dict,
    ladders: dict,
    limit: int | None = None,
    band: float = SPREAD,
    windows: dict | None = None,
) -> dict:
    """Tier sizes as the solution of the same program, not a chosen number.

    Per tier the program is::

        maximise  |S|                             how many interchangeable fallbacks
        s.t.      index[m,e] in [target, target + band]   same score as the tier
                  window[m] >= window[tier]                no context regression

    The feasible set is monotone in ``|S|`` and every candidate is independent,
    so the optimum is simply every model that satisfies both constraints -- the
    roster decides how wide a tier gets. ``limit`` only exists for callers that
    want a smaller chain. Existing entries keep their place and effort.
    """
    rungs = {
        name: _rungs(variants or {}) for name, variants in (ladders or {}).items()
    }
    grown = {}
    for tier, entries in tier_map.items():
        kept = [(model, effort) for model, effort in entries if model]
        if not rungs or (limit is not None and len(kept) >= limit):
            grown[tier] = kept
            continue

        indexes, window = [], None
        for model, effort in kept:
            choices = rungs.get(model) or []
            here = next((r for r in choices if r["effort"] == effort), None)
            if here is None and choices:
                here = choices[-1]
            if here:
                indexes.append(here["index"])
            if windows and model in windows:
                window = windows[model] if window is None else min(window, windows[model])
        if not indexes:
            grown[tier] = kept
            continue

        target = min(indexes)
        have = {model for model, _ in kept}
        extra = []
        for name, choices in rungs.items():
            if name in have:
                continue
            if window is not None and windows and windows.get(name, 0) < window:
                continue
            rung = _cheapest(choices, target, band)
            if rung is None:
                continue
            extra.append((rung["cost"], -rung["index"], name, rung["effort"]))
        extra.sort()
        room = len(extra) if limit is None else min(limit - len(kept), len(extra))
        grown[tier] = kept + [(name, effort) for _, _, name, effort in extra[:room]]
    return grown


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
