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
BUDGET_MARKER = "auto_pool_budget_blocked"


def budget_sweep(client, rows: list[dict], health, apply: bool = False) -> list[dict]:
    """Align whole credentials with LiteLLM's budget verdict.

    A budget is enforced per key, so one budget-exceeded deployment proves every
    deployment sharing that credential is unusable -- including ones a sibling
    model group would otherwise hide, because a probe of the group may be served
    by a different key. ``health`` is called with a deployment id; the health
    route answers for blocked deployments too, so nothing has to be unblocked
    just to learn the credential's state.

    Only this sweep's own marker is lifted on recovery. A deployment blocked for
    another reason without the marker is adopted into the budget marker (the
    proof is direct), so its recovery can be automated as well.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        credential = (row.get("litellm_params") or {}).get("litellm_credential_name")
        if credential and (row.get("model_info") or {}).get("id"):
            groups.setdefault(credential, []).append(row)

    actions: list[dict] = []
    for credential in sorted(groups):
        group = groups[credential]
        canary = min(
            group,
            key=lambda row: (
                str((row.get("model_info") or {}).get("mode") or "chat")
                not in ("chat", "completion"),
                bool((row.get("model_info") or {}).get("blocked")),
                (row.get("litellm_params") or {}).get("order", 1),
            ),
        )
        identifier = canary["model_info"]["id"]
        state = health(identifier)
        if state == "budget_exceeded":
            for row in group:
                info = row.get("model_info") or {}
                deployment = info.get("id")
                was_blocked = bool(info.get("blocked"))
                # Already blocked by this sweep with its marker: nothing to prove
                # or write again, so a repeated run stays silent.
                if was_blocked and info.get(BUDGET_MARKER):
                    continue
                if apply:
                    if not was_blocked:
                        result = client.post("/model/block", {"model_id": deployment})
                        if result.get("error_status") is not None:
                            actions.append({
                                "model": row["model_name"], "deployment": deployment,
                                "action": "failed",
                                "detail": f"block_{result['error_status']}",
                            })
                            continue
                    if not info.get(BUDGET_MARKER):
                        client.patch(
                            f"/model/{deployment}/update",
                            {"model_info": {BUDGET_MARKER: True}},
                        )
                info["blocked"] = True
                info[BUDGET_MARKER] = True
                actions.append({
                    "model": row["model_name"], "deployment": deployment,
                    "action": "adopted" if was_blocked else "blocked",
                    "reason": "budget_exceeded",
                })
        elif state == "ok":
            for row in group:
                info = row.get("model_info") or {}
                if not info.get(BUDGET_MARKER):
                    continue
                deployment = info["id"]
                if apply:
                    client.post("/model/unblock", {"model_id": deployment})
                    client.patch(
                        f"/model/{deployment}/update",
                        {"model_info": {BUDGET_MARKER: False}},
                    )
                info["blocked"] = False
                info[BUDGET_MARKER] = False
                actions.append({
                    "model": row["model_name"], "deployment": deployment,
                    "action": "unblocked", "reason": "budget_recovered",
                })
    return actions


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
