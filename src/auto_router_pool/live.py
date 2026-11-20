"""Keep fallback chains live and ending on a self-selected resort.

The resort is never hardcoded: it is the cheapest model from a family the auto
tiers do not already use, with an unblocked deployment and a window above the
272k pool floor. An incumbent keeps the role while it still qualifies, so a
harmless price change cannot churn the graph. No inference is performed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .cli import Settings
from .client import Client, models
from .pool import MIN_INPUT_TOKENS
from .tiers import ROUTERS

FAMILIES = (("gpt-", "openai"), ("claude-", "anthropic"),
            ("gemini-", "gemini"), ("grok-", "xai"))


def family(name: str) -> str | None:
    for prefix, label in FAMILIES:
        if name.startswith(prefix):
            return label
    return None


def edges_from(entries) -> dict[str, list[str]]:
    out = {}
    for entry in entries or []:
        source, targets = next(iter(entry.items()))
        out[source] = list(targets)
    return out


def group_facts(rows: list[dict]) -> dict[str, dict]:
    """Deployment count, unblocked count and largest declared window per group."""
    facts: dict[str, dict] = {}
    for row in rows:
        name = row["model_name"]
        info = row.get("model_info") or {}
        fact = facts.setdefault(name, {"unblocked": 0, "total": 0, "window": 0})
        fact["total"] += 1
        if not info.get("blocked"):
            fact["unblocked"] += 1
        cap = info.get("max_input_tokens")
        if type(cap) is int and cap > 0:
            fact["window"] = max(fact["window"], cap)
    return facts


def usable(name: str, facts: dict) -> bool:
    """A target must be routable at all: live and above the 272k floor."""
    fact = facts.get(name)
    return bool(fact and fact["unblocked"] > 0 and fact["window"] >= MIN_INPUT_TOKENS)


def tier_members(rows: list[dict]) -> set[str]:
    """Models the managed routers can actually select."""
    members: set[str] = set()
    for row in rows:
        if row["model_name"] not in ROUTERS:
            continue
        config = (row.get("litellm_params") or {}).get("complexity_router_config") or {}
        for names in (config.get("tiers") or {}).values():
            members.update(names)
    return members


def blended_price(prices: dict, name: str) -> float | None:
    entry = prices.get(name) or {}
    pin, pout = entry.get("input_cost_per_token"), entry.get("output_cost_per_token")
    if type(pin) not in (int, float) or type(pout) not in (int, float):
        return None
    return pin + pout


def choose_terminal(
    rows: list[dict],
    prices: dict,
    solvers: set[str],
    current: str | None = None,
) -> tuple[str | None, dict]:
    """Pick the last-resort model, returning it with the facts that justify it."""
    facts = group_facts(rows)
    used = {family(name) for name in solvers}

    def qualifies(name: str) -> bool:
        fact = facts.get(name)
        return (
            fact is not None
            and fact["unblocked"] > 0
            and fact["window"] >= MIN_INPUT_TOKENS
            and name not in solvers
            and not name.startswith(("__", "auto"))
        )

    candidates = [name for name in facts if qualifies(name)]
    if not candidates:
        return None, {}

    def rank(name: str):
        price = blended_price(prices, name)
        return (
            family(name) is None or family(name) in used,
            price is None,
            price or 0.0,
            -facts[name]["unblocked"],
            name,
        )

    best = min(candidates, key=rank)
    # Stability: keep the incumbent while it still ranks in the same class.
    if current in candidates and rank(current)[:2] == rank(best)[:2]:
        best = current
    return best, facts[best]


def required(terminal: str) -> dict[str, list[str]]:
    """Chains whose resort hop matters beyond price similarity.

    The resort itself is absent on purpose: it keeps no chain of its own.
    """
    return {
        "gemini-4-argon": [terminal, "claude-sonnet-5-5", "gpt-6.1-sol"],
        "gpt-6.1-terra": ["claude-sonnet-5-5", terminal, "gpt-6.1-sol"],
        "claude-haiku-5-5": [terminal, "gpt-6-luna", "claude-sonnet-5-5"],
    }


def sanitize(rows: list[dict], settings: dict, prices: dict, incumbent: str | None = None) -> dict:
    """Return fallbacks with dead targets removed and every chain ending live."""
    facts = group_facts(rows)
    alive = {name: fact["unblocked"] > 0 for name, fact in facts.items()}
    edges = edges_from(settings.get("fallbacks"))
    if incumbent is None:
        counts = Counter(target for targets in edges.values() for target in targets)
        incumbent = counts.most_common(1)[0][0] if counts else None
    terminal, fact = choose_terminal(rows, prices, tier_members(rows), incumbent)
    if not terminal:
        # Fail closed: a chain ending on a dead model is worse than no write.
        raise ValueError("no live model qualifies as a terminal fallback resort")

    edges[terminal] = []
    changes = []
    for source, wanted in required(terminal).items():
        current = edges.get(source, [])
        # Exact on purpose: stale resort hops and sub-floor targets do not linger.
        filtered = [target for target in wanted if usable(target, facts) and target != source]
        if not filtered and source != terminal:
            filtered = [terminal]
        if filtered != current:
            changes.append({"source": source, "before": current, "after": filtered})
            edges[source] = filtered

    for source, targets in list(edges.items()):
        healthy = [target for target in targets if usable(target, facts) and target != source]
        if targets and not healthy and source != terminal:
            changes.append({"source": source, "before": targets, "after": [terminal],
                            "reason": "all_targets_blocked"})
            edges[source] = [terminal]
        elif healthy != [target for target in targets if target != source]:
            new = healthy or ([terminal] if source != terminal else [])
            if new != targets:
                changes.append({"source": source, "before": targets, "after": new,
                                "reason": "strip_blocked"})
                edges[source] = new

    return {
        "terminal": terminal,
        "terminal_family": family(terminal),
        "terminal_unblocked": fact["unblocked"],
        "terminal_window": fact["window"],
        "fallbacks": [
            {source: edges[source]} for source in sorted(edges) if edges[source]
        ],
        "changes": changes,
        "alive": alive,
    }


def verify(fallbacks: list[dict], alive: dict, terminal: str) -> None:
    graph = edges_from(fallbacks)
    for source in required(terminal):
        if not any(alive.get(target) for target in graph.get(source, [])):
            raise ValueError(f"{source} still has no alive fallback")
    if graph.get(terminal):
        raise ValueError("terminal resort kept its own chain")


def seed_unblock(client, rows: list[dict], seeds: dict) -> list[str]:
    """Unblock the seeded deployment of a group that is otherwise fully blocked."""
    id_to_row = {row["model_info"]["id"]: row for row in rows}
    unblocked = []
    for model_name, deployment_id in seeds.items():
        if model_name == "reason":
            continue
        row = id_to_row.get(deployment_id)
        if not row or row["model_name"] != model_name:
            continue
        if not (row.get("model_info") or {}).get("blocked"):
            continue
        if any(not (r.get("model_info") or {}).get("blocked")
               for r in rows if r["model_name"] == model_name):
            continue
        client.post(f"/model/{deployment_id}/update", {"model_info": {"blocked": False}})
        unblocked.append(model_name)
    return unblocked


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Keep every fallback chain live.")
    parser.add_argument("--apply", action="store_true", help="write (default: preview)")
    parser.add_argument("--base-url", default="http://127.0.0.1:4000")
    parser.add_argument("--env-file", type=Path, default=None, help="file with KEY=VALUE secrets")
    parser.add_argument("--prices", type=Path, default=None, help="pinned LiteLLM price map")
    parser.add_argument("--seed-file", type=Path, default=None,
                        help='JSON {"group": "deployment-id"} to unblock when fully blocked')
    args = parser.parse_args(argv)

    settings = Settings.from_env(args.env_file)
    settings.litellm_base = args.base_url
    if not settings.key:
        print(json.dumps({"error": "missing environment", "names": ["LITELLM_MASTER_KEY"]}))
        return 1
    client = Client(settings.litellm_base, settings.key)
    rows = models(client)
    seeded: list[str] = []
    if args.seed_file and args.seed_file.exists():
        seeded = seed_unblock(client, rows, json.loads(args.seed_file.read_text()))
        if seeded:
            # Re-read so a seeded deployment counts towards the terminal proof.
            rows = models(client)
    state = client.get("/router/settings")
    if state.get("error_status"):
        print(json.dumps({"error": "router settings unavailable",
                          "status": state["error_status"]}))
        return 1
    prices = json.loads(args.prices.read_text()) if args.prices and args.prices.exists() else {}
    result = sanitize(rows, state.get("current_values", state), prices)
    report = {
        "terminal": result["terminal"],
        "terminal_family": result["terminal_family"],
        "terminal_unblocked": result["terminal_unblocked"],
        "terminal_window": result["terminal_window"],
        "seeded": seeded,
    }
    if not args.apply or not result["changes"]:
        print(json.dumps({"applied": False, "changed": len(result["changes"]), **report}))
        return 0

    client.post("/config/update", {"router_settings": {"fallbacks": result["fallbacks"]}})
    after = client.get("/router/settings")
    if after.get("error_status"):
        print(json.dumps({"error": "readback failed", "status": after["error_status"]}))
        return 1
    actual = edges_from(after.get("current_values", after).get("fallbacks"))
    verify(after.get("current_values", after).get("fallbacks"), result["alive"], result["terminal"])
    print(json.dumps({
        "applied": True,
        "changed": len(result["changes"]),
        **report,
        "resort_chain": actual.get(result["terminal"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
