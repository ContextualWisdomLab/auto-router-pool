"""Align block flags with what the probe says, instead of a separate cron.

A router can only route to a deployment it is not blocking, so a stale block and a
missing block cost the same: one removes capacity, the other sends traffic to a
model that cannot answer. This sweep probes each managed model group once and
writes the result back.

Ownership is explicit. Only deployments this sweep blocked carry
``auto_pool_quarantined``; an account-level block written by a credential sync is
left alone, because the sweep cannot prove an account is fine from one probe of
one model.
"""
from __future__ import annotations

from .pool import MIN_INPUT_TOKENS, input_window

MARKER = "auto_pool_quarantined"


def sweep(client, rows: list[dict], names, probe, apply: bool = False) -> list[dict]:
    """Probe every named group and align its block flags with the answer.

    ``probe(name)`` returns ``(ok, reason)``. A failing group keeps its last live
    deployment: blocking every deployment would remove the group from routing
    instead of narrowing it, which is a worse outcome than one failing call.
    """
    actions: list[dict] = []
    for name in sorted(set(names)):
        group = [row for row in rows if row["model_name"] == name]
        if not group:
            continue
        # Read the declared window, not the block flags: a group this sweep
        # blocked itself has to stay eligible, or nothing can ever lift the block.
        if input_window(rows, name) < MIN_INPUT_TOKENS:
            actions.append(
                {"model": name, "action": "skipped", "reason": "context_below_floor"}
            )
            continue

        ok, detail = probe(name)
        live = [
            row for row in group
            if not (row.get("model_info") or {}).get("blocked")
        ]
        if ok:
            for row in group:
                info = row.get("model_info") or {}
                if not info.get(MARKER):
                    continue
                if apply:
                    client.post("/model/unblock", {"model_id": info["id"]})
                    client.patch(
                        f"/model/{info['id']}/update",
                        {"model_info": {MARKER: False}},
                    )
                info["blocked"] = False
                actions.append(
                    {"model": name, "deployment": info["id"], "action": "unblocked"}
                )
            continue

        if len(live) < 2:
            actions.append(
                {
                    "model": name,
                    "action": "kept",
                    "reason": f"{detail};last_deployment" if live else detail,
                }
            )
            continue

        # Narrow, never empty: the first live deployment is kept so a transient
        # failure cannot take a whole group out of routing.
        for row in live[1:]:
            info = row.get("model_info") or {}
            if apply:
                client.post("/model/block", {"model_id": info["id"]})
                client.patch(
                    f"/model/{info['id']}/update",
                    {"model_info": {MARKER: True}},
                )
            info[MARKER] = True
            info["blocked"] = True
            actions.append(
                {
                    "model": name,
                    "deployment": info["id"],
                    "action": "blocked",
                    "reason": detail,
                }
            )
    return actions
