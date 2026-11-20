"""Diff, validate, patch, verify and roll back router configuration."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from .tiers import build


def plan(rows: list[dict], routers: dict, default: str):
    """Return only the routers whose tiers or efforts actually differ."""
    by_name = {row["model_name"]: row for row in rows}
    changes = []
    for name, tier_map in routers.items():
        row = by_name.get(name)
        if not row:
            continue
        config = copy.deepcopy(row["litellm_params"]["complexity_router_config"])
        tiers, configs = build(tier_map)
        if config.get("tiers") == tiers and config.get("tier_model_configs") == configs:
            continue
        config["tiers"] = tiers
        config["tier_model_configs"] = configs
        config["default_model"] = default
        changes.append(
            {
                "model_name": name,
                "id": row["model_info"]["id"],
                "previous": copy.deepcopy(row["litellm_params"]),
                "config": config,
                "litellm_params": {
                    **row["litellm_params"],
                    "complexity_router_config": config,
                    "complexity_router_default_model": default,
                },
            }
        )
    return changes


def validate(client, changes: list[dict]):
    for change in changes:
        result = client.post(
            "/auto_router/validate_complexity_router_config",
            {"complexity_router_config": change["config"]},
        )
        if result.get("valid") is not True:
            raise ValueError(f"server rejected {change['model_name']}")


def backup(changes: list[dict], root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"auto-pool-before-{stamp}.json"
    path.write_text(
        json.dumps(
            [
                {"model_name": c["model_name"], "previous": c["previous"],
                 "config": c["config"]}
                for c in changes
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return path


def apply(client, changes: list[dict]):
    """Patch every changed router, then re-read and verify; roll back on drift."""
    for change in changes:
        client.patch(
            f"/model/{change['id']}/update",
            {"litellm_params": change["litellm_params"]},
        )

    after = {row["model_name"]: row for row in client.get("/model/info").get("data", [])}
    for change in changes:
        got = after.get(change["model_name"], {}).get("litellm_params", {}).get(
            "complexity_router_config", {}
        )
        if got.get("tiers") != change["config"]["tiers"]:
            for old in changes:
                client.patch(
                    f"/model/{old['id']}/update",
                    {"litellm_params": old["previous"]},
                )
            raise RuntimeError("verification failed; rolled back")


def set_auto_fallback(client, chain: list[str]):
    """Replace only the ``auto`` edge, preserving every other fallback."""
    settings = client.get("/router/settings").get("current_values", {})
    edges: dict[str, list[str]] = {}
    for entry in settings.get("fallbacks") or []:
        source, targets = next(iter(entry.items()))
        edges[source] = list(targets)
    edges["auto"] = list(chain)
    fallbacks = [{src: targets} for src, targets in sorted(edges.items()) if targets]
    client.post("/config/update", {"router_settings": {"fallbacks": fallbacks}})


def smoke(client) -> dict:
    result = client.post(
        "/v1/chat/completions",
        {
            "model": "auto",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "Return the word ok."}],
        },
    )
    if result.get("error_status"):
        return {"smoke": "failed", "status": result["error_status"]}
    return {
        "smoke": "ok",
        "finish": result.get("choices", [{}])[0].get("finish_reason"),
    }
