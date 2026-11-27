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

import json
import subprocess
from datetime import datetime, timedelta, timezone

from .extra_catalogs import litellm_container
from .pool import MIN_INPUT_TOKENS, group_state, input_window

MARKER = "auto_pool_quarantined"
BUDGET_MARKER = "auto_pool_budget_blocked"
LIVE_FAILURE_MARKER = "auto_pool_live_failure_blocked"
LIVE_FAILURE_AT = "auto_pool_live_failure_at"
LIVE_FAILURE_REASON = "auto_pool_live_failure_reason"
LIVE_FAILURE_MIN_ERRORS = 5
LIVE_FAILURE_MIN_RATE = 0.80
LIVE_FAILURE_WINDOW_MINUTES = 30
LIVE_FAILURE_RETRY_HOURS = 24


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

SPEND_FAILURE_QUERY = """
import json, os, sys
from urllib.parse import urlsplit, urlunsplit
import psycopg

window_minutes = int(sys.argv[1])
u = urlsplit(os.environ["DATABASE_URL"])
sql = (
    'SELECT model_id, model, call_type, '
    "COUNT(*) FILTER (WHERE status = 'success') AS ok, "
    "COUNT(*) FILTER ("
    "  WHERE status = 'failure' "
    "    AND COALESCE(metadata->'error_information'->>'error_code', '') <> '429' "
    "    AND COALESCE(metadata->'error_information'->>'error_message', '') NOT ILIKE '%%prompt is too long%%' "
    "    AND COALESCE(metadata->'error_information'->>'error_message', '') NOT ILIKE '%%context_length_exceeded%%' "
    "    AND COALESCE(metadata->'error_information'->>'error_message', '') NOT ILIKE '%%maximum context length%%'"
    ") AS bad "
    'FROM "LiteLLM_SpendLogs" '
    # SpendLogs stores UTC as a timestamp without a time zone.
    'WHERE "startTime" >= '
    "(CURRENT_TIMESTAMP AT TIME ZONE 'UTC') - make_interval(mins => %s) "
    "  AND model_id IS NOT NULL AND model_id <> '' "
    "  AND call_type IN ('anthropic_messages', 'acompletion', 'aresponses') "
    "  AND NOT ("
    "    status = 'success' "
    "    AND COALESCE(prompt_tokens, 0) <= 20 "
    "    AND COALESCE(completion_tokens, 0) <= 5"
    "  ) "
    'GROUP BY model_id, model, call_type'
)
with psycopg.connect(
    urlunsplit((u.scheme, u.netloc, u.path, "", "")),
    connect_timeout=5,
    options="-c statement_timeout=5000 -c default_transaction_read_only=on",
) as conn:
    rows = conn.execute(sql, (window_minutes,)).fetchall()
print(json.dumps([
    {
        "model_id": str(model_id),
        "model": str(model or ""),
        "call_type": str(call_type or ""),
        "ok": int(ok or 0),
        "bad": int(bad or 0),
    }
    for model_id, model, call_type, ok, bad in rows
]))
"""

def spend_failure_stats(
    window_minutes: int = LIVE_FAILURE_WINDOW_MINUTES,
    runner=subprocess.run,
) -> list[dict]:
    """Read recent per-(deployment, call_type) real-traffic outcomes from SpendLogs.

    Fails closed (empty list) when Docker or PostgreSQL is unreachable so a
    monitoring outage never blocks router maintenance.
    """
    try:
        completed = runner(
            ["docker", "exec", "-i", litellm_container(), "python", "-", str(int(window_minutes))],
            input=SPEND_FAILURE_QUERY,
            text=True,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    try:
        parsed = json.loads(completed.stdout)
    except ValueError:
        return []
    return parsed if isinstance(parsed, list) else []


def _parse_iso(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _eligible_live_models(rows: list[dict]) -> set[str]:
    names = {
        row["model_name"]
        for row in rows
        if not (row.get("model_info") or {}).get("blocked")
        and str((row.get("model_info") or {}).get("mode") or "chat") in ("chat", "completion")
        and not str(row.get("model_name") or "").startswith(("__", "auto"))
    }
    return {name for name in names if group_state(rows, name)[0]}


def live_failure_sweep(
    client,
    rows: list[dict],
    managed_names,
    stats: list[dict],
    *,
    apply: bool = False,
    now: datetime | None = None,
    min_errors: int = LIVE_FAILURE_MIN_ERRORS,
    min_rate: float = LIVE_FAILURE_MIN_RATE,
    retry_hours: int = LIVE_FAILURE_RETRY_HOURS,
) -> list[dict]:
    """Quarantine managed deployments failing persistently in real traffic.

    A 1-token non-streaming chat probe can return 200 while real streaming calls
    (such as /v1/messages) fail mid-stream after HTTP 200. Because LiteLLM only
    cools down pre-stream errors, these deployments otherwise stay in ``auto``
    indefinitely.

    Uses ``LIVE_FAILURE_MARKER`` (distinct from ``MARKER``) so ``sweep()``'s
    synthetic probe cannot prematurely unblock it and oscillate every run.
    Recovery waits ``retry_hours`` before lifting our own block; manual operator
    blocks without ``LIVE_FAILURE_MARKER`` are never touched.
    """
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    now_iso = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    actions: list[dict] = []

    # Step 1: lift only our own live-failure blocks after the retry cooldown.
    for row in rows:
        info = row.get("model_info") or {}
        if not info.get("blocked") or not info.get(LIVE_FAILURE_MARKER):
            continue
        if info.get(BUDGET_MARKER) or info.get(MARKER):
            continue
        blocked_at = _parse_iso(info.get(LIVE_FAILURE_AT))
        if blocked_at is None:
            continue
        if current_time - blocked_at < timedelta(hours=retry_hours):
            continue
        deployment = info.get("id")
        if not deployment:
            continue
        if apply:
            unblock = client.post("/model/unblock", {"model_id": deployment})
            if unblock.get("error_status") is not None:
                actions.append({
                    "model": row["model_name"],
                    "deployment": deployment,
                    "action": "failed",
                    "detail": f"unblock_{unblock['error_status']}",
                })
                continue
            client.patch(
                f"/model/{deployment}/update",
                {
                    "model_info": {
                        LIVE_FAILURE_MARKER: False,
                        LIVE_FAILURE_AT: None,
                        LIVE_FAILURE_REASON: None,
                    }
                },
            )
        info["blocked"] = False
        info[LIVE_FAILURE_MARKER] = False
        info[LIVE_FAILURE_AT] = None
        info[LIVE_FAILURE_REASON] = None
        actions.append({
            "model": row["model_name"],
            "deployment": deployment,
            "action": "unblocked",
            "reason": "live_failure_cooldown_elapsed",
        })

    # Step 2: quarantine managed unblocked deployments failing on live surfaces.
    managed = set(managed_names)
    by_deployment: dict[tuple[str, str], list[dict]] = {}
    for entry in stats or []:
        mid = str(entry.get("model_id") or "")
        model = str(entry.get("model") or "")
        if mid and model:
            by_deployment.setdefault((mid, model), []).append(entry)

    for row in rows:
        name = row.get("model_name")
        if name not in managed:
            continue
        info = row.get("model_info") or {}
        if info.get("blocked"):
            continue
        deployment = info.get("id")
        current_model = str((row.get("litellm_params") or {}).get("model") or "")
        if not deployment or not current_model:
            continue
        surfaces = by_deployment.get((deployment, current_model), [])
        breach = None
        for entry in surfaces:
            ok = int(entry.get("ok") or 0)
            bad = int(entry.get("bad") or 0)
            total = ok + bad
            if bad >= min_errors and total > 0 and (bad / total) >= min_rate:
                breach = f"{entry.get('call_type', 'unknown')}:{bad}/{total}"
                break
        if breach is None:
            continue

        # Never empty the entire eligible chat pool.
        live_siblings = [
            other for other in rows
            if other["model_name"] == name
            and not (other.get("model_info") or {}).get("blocked")
            and (other.get("model_info") or {}).get("id") != deployment
        ]
        if not live_siblings:
            remaining = _eligible_live_models(rows) - {name}
            if not remaining:
                actions.append({
                    "model": name,
                    "deployment": deployment,
                    "action": "kept",
                    "reason": f"{breach};last_eligible_model",
                })
                continue

        if apply:
            result = client.post("/model/block", {"model_id": deployment})
            if result.get("error_status") is not None:
                actions.append({
                    "model": name,
                    "deployment": deployment,
                    "action": "failed",
                    "detail": f"block_{result['error_status']}",
                })
                continue
            patch = client.patch(
                f"/model/{deployment}/update",
                {
                    "model_info": {
                        LIVE_FAILURE_MARKER: True,
                        LIVE_FAILURE_AT: now_iso,
                        LIVE_FAILURE_REASON: breach,
                    }
                },
            )
            if patch.get("error_status") is not None:
                # Only undo the block we just applied. Without the marker,
                # a later sweep cannot tell this apart from a manual block.
                rollback = client.post("/model/unblock", {"model_id": deployment})
                detail = f"patch_{patch['error_status']}"
                if rollback.get("error_status") is not None:
                    info["blocked"] = True
                    detail += f";rollback_unblock_{rollback['error_status']}"
                actions.append({
                    "model": name,
                    "deployment": deployment,
                    "action": "failed",
                    "detail": detail,
                })
                continue
        info["blocked"] = True
        info[LIVE_FAILURE_MARKER] = True
        info[LIVE_FAILURE_AT] = now_iso
        info[LIVE_FAILURE_REASON] = breach
        actions.append({
            "model": name,
            "deployment": deployment,
            "action": "blocked",
            "reason": breach,
        })

    return actions
