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

import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from .extra_catalogs import (
    atomic,
    config_api_keys,
    litellm_container,
    retired_names,
)
from .pool import MIN_INPUT_TOKENS, input_window

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
                still_blocked = any(info.get(marker) for marker in (MARKER, LIVE_FAILURE_MARKER))
                if apply:
                    if not still_blocked:
                        client.post("/model/unblock", {"model_id": deployment})
                    client.patch(
                        f"/model/{deployment}/update",
                        {"model_info": {BUDGET_MARKER: False}},
                    )
                info["blocked"] = still_blocked
                info[BUDGET_MARKER] = False
                actions.append({
                    "model": row["model_name"], "deployment": deployment,
                    "action": "cleared" if still_blocked else "unblocked",
                    "reason": "budget_recovered",
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
        active_window = input_window(group, name)
        recovery_window = input_window(group, name, include_blocked=True)
        has_recoverable_block = any(
            (row.get("model_info") or {}).get(MARKER) for row in group
        )
        if active_window < MIN_INPUT_TOKENS and (
            not has_recoverable_block or recovery_window < MIN_INPUT_TOKENS
        ):
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
                still_blocked = any(info.get(marker) for marker in (BUDGET_MARKER, LIVE_FAILURE_MARKER))
                if apply:
                    if not still_blocked:
                        client.post("/model/unblock", {"model_id": info["id"]})
                    client.patch(
                        f"/model/{info['id']}/update",
                        {"model_info": {MARKER: False}},
                    )
                info[MARKER] = False
                info["blocked"] = still_blocked
                actions.append({
                    "model": name, "deployment": info["id"],
                    "action": "cleared" if still_blocked else "unblocked",
                })
            continue

        if active_window < MIN_INPUT_TOKENS:
            actions.append(
                {"model": name, "action": "skipped", "reason": "context_below_floor"}
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
    'WHERE "startTime" >= CURRENT_TIMESTAMP - make_interval(mins => %s) '
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
    eligible = [
        row
        for row in rows
        if not (row.get("model_info") or {}).get("blocked")
        and str((row.get("model_info") or {}).get("mode") or "chat") in ("chat", "completion")
        and not str(row.get("model_name") or "").startswith(("__", "auto"))
    ]
    names = {row["model_name"] for row in eligible}
    return {name for name in names if input_window(eligible, name) >= MIN_INPUT_TOKENS}


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
        if (
            not info.get("blocked") or not info.get(LIVE_FAILURE_MARKER)
            or info.get(BUDGET_MARKER) or info.get(MARKER)
        ):
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
        remaining = _eligible_live_models([other for other in rows if other is not row])
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
            metadata = client.patch(
                f"/model/{deployment}/update",
                {
                    "model_info": {
                        LIVE_FAILURE_MARKER: True,
                        LIVE_FAILURE_AT: now_iso,
                        LIVE_FAILURE_REASON: breach,
                    }
                },
            )
            if metadata.get("error_status") is not None:
                rollback = client.post("/model/unblock", {"model_id": deployment})
                rollback_error = rollback.get("error_status")
                info["blocked"] = rollback_error is not None
                actions.append({
                    "model": name,
                    "deployment": deployment,
                    "action": "failed",
                    "detail": f"metadata_{metadata['error_status']};" + (
                        "block_rolled_back" if rollback_error is None
                        else f"rollback_{rollback_error}"
                    ),
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


CLIPROXY_CREDENTIAL = "cliproxyapi-shared"
CLIPROXY_MANUAL_BLOCK = "auto_pool_manual_block"
CLIPROXY_UNAVAILABLE_RETRY_SECONDS = 6 * 3600
CLIPROXY_COOLDOWN_RESET_MIN_INTERVAL = 600
CLIPROXY_SERVICE = "agent-gateway-core.service"
CLIPROXY_RECOVERY_MODES = frozenset({"chat", "completion", "responses"})


def _parse_health(raw) -> tuple[str, str]:
    if isinstance(raw, tuple) and len(raw) == 2:
        return str(raw[0]), str(raw[1] or "").lower()
    return str(raw), ""


def _is_cliproxy_cooldown(verdict: str, detail: str) -> bool:
    if verdict == "model_cooldown":
        return True
    if verdict == "rate_limited" and any(
        marker in detail for marker in ("cooling down", "model_cooldown")
    ):
        return True
    return False


def _wait_cliproxy_ready(base: str, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    url = base.rstrip("/") + "/v1/models"
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(urllib.request.Request(url), timeout=2.0)
            return True
        except urllib.error.HTTPError:
            # 401/403 proves the HTTP server is up and serving requests.
            return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.3)
    return False


def reset_cliproxy_cooldowns(
    cliproxy_base: str = "http://127.0.0.1:8317",
    *,
    management_key: str = "",
    service: str = CLIPROXY_SERVICE,
    runner=subprocess.run,
    ready_check=None,
) -> str | None:
    """Clear stale in-memory CLIProxyAPI cooldowns so probes hit upstream live.

    Prefers the zero-downtime management API when ``management_key`` is set;
    otherwise restarts the local systemd user service (which clears RAM-only
    force-cooldowns when ``save-cooldown-status`` is false) and waits until the
    listener accepts connections again.
    """
    base = cliproxy_base.rstrip("/")
    if management_key:
        for path in (
            "/v0/management/routing/cooldown/reset",
            "/v8/management/routing/cooldown/reset",
        ):
            req = urllib.request.Request(
                base + path,
                data=b"{}",
                headers={
                    "Authorization": "Bearer " + management_key,
                    "X-Management-Key": management_key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if 200 <= getattr(resp, "status", 200) < 300:
                        return "management_api"
            except (OSError, urllib.error.URLError):
                continue

    if service:
        try:
            proc = runner(
                ["systemctl", "--user", "restart", service],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if getattr(proc, "returncode", 1) == 0:
            check = ready_check or (lambda: _wait_cliproxy_ready(base))
            check()
            return "service_restart"
    return None


def oauth_fingerprint(home: Path) -> str:
    """Hash OAuth credential file metadata without ever reading token contents.

    Adding, replacing, or refreshing a CLIProxyAPI credential file under
    ``<home>/oauth`` changes this fingerprint, immediately invalidating any
    failure backoff so recovered accounts are re-probed on the very next sync.
    """
    oauth_dir = Path(home) / "oauth"
    if not oauth_dir.is_dir():
        return ""
    entries = []
    for path in sorted(oauth_dir.iterdir(), key=lambda p: p.name):
        if not path.is_file() or not path.name.endswith(".json"):
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        entries.append((path.name, st.st_mtime_ns, st.st_size))
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def cliproxy_catalog_ids(
    home: Path,
    base: str = "http://127.0.0.1:8317",
    api_key: str | None = None,
    timeout: float = 10.0,
) -> set[str] | None:
    """Return upstream model IDs currently advertised by CLIProxyAPI.

    Returns ``None`` if CLIProxyAPI is not configured or unreachable, so the
    recovery sweep fails closed without probing a dead proxy.
    """
    key = (api_key or "").strip()
    if not key:
        cfg = Path(home) / "core-config.yaml"
        if cfg.exists():
            try:
                keys = config_api_keys(cfg.read_text())
                key = keys[0] if keys else ""
            except OSError:
                key = ""
    if not key:
        return None
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/models",
        headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except (OSError, urllib.error.URLError, ValueError):
        return None
    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None
    return {
        str(item["id"]).strip()
        for item in items
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip()
    }


def cliproxy_recovery_sweep(
    client,
    rows: list[dict],
    health,
    *,
    home: Path | None = None,
    cliproxy_base: str = "http://127.0.0.1:8317",
    cliproxy_key: str | None = None,
    management_key: str = "",
    catalog_ids: set[str] | None = None,
    retired: set[str] | None = None,
    apply: bool = False,
    workers: int = 4,
    now: float | None = None,
    reset_cooldowns=None,
    reset_min_interval: float = CLIPROXY_COOLDOWN_RESET_MIN_INTERVAL,
) -> list[dict]:
    """Re-probe blocked CLIProxyAPI deployments and unblock recovered ones.

    CLIProxyAPI multiplexes multiple upstream OAuth accounts behind the single
    ``cliproxyapi-shared`` credential. When an account hits a usage outage, its
    deployments are blocked (either by quarantine, triage, or an earlier health
    pass). Once new credentials are added or quotas reset, this sweep verifies
    recovery via LiteLLM's ``/health?model_id=...`` (which probes blocked
    deployments directly) and lifts the block automatically.

    Safety boundaries:
    - Never touches deployments outside ``cliproxyapi-shared``.
    - Never lifts ``auto_pool_live_failure_blocked`` (owned by 24h live-failure
      quarantine), ``auto_pool_manual_block``, retired models, internal ``__*``
      aliases, or ``*-blocked`` aliases.
    - Deduplicates probes by ``(litellm_params.model, api_base)``.
    - Rate-limit/cooldown failures are retried every sync so quota resets are
      picked up promptly; permanent non-rate-limit errors (e.g. account
      ``not_found``) back off for 6 hours, and any change in ``<home>/oauth``
      immediately clears all backoff.
    """
    retired_set = retired if retired is not None else (
        retired_names(home) if home is not None else set()
    )

    candidates: list[dict] = []
    for row in rows:
        params = row.get("litellm_params") or {}
        info = row.get("model_info") or {}
        if params.get("litellm_credential_name") != CLIPROXY_CREDENTIAL:
            continue
        if not info.get("blocked") or not info.get("id"):
            continue
        name = str(row.get("model_name") or "")
        if not name or name.startswith("__") or name.endswith("-blocked"):
            continue
        if name in retired_set:
            continue
        if info.get(LIVE_FAILURE_MARKER) or info.get(CLIPROXY_MANUAL_BLOCK):
            continue
        mode = str(info.get("mode") or "chat").lower()
        if mode not in CLIPROXY_RECOVERY_MODES:
            continue
        if not str(params.get("model") or "").strip():
            continue
        candidates.append(row)

    if not candidates:
        return []

    if catalog_ids is None:
        if home is None:
            return []
        catalog_ids = cliproxy_catalog_ids(
            home, base=cliproxy_base, api_key=cliproxy_key
        )
        if not catalog_ids:
            return []

    now_ts = time.time() if now is None else float(now)
    state_path = Path(home) / "cliproxy-recovery-state.json" if home is not None else None
    state = {}
    if state_path and state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except (OSError, ValueError):
            state = {}
    current_oauth_fp = oauth_fingerprint(home) if home is not None else ""
    oauth_changed = bool(current_oauth_fp) and current_oauth_fp != state.get("oauth_fingerprint")
    backoff: dict[str, dict] = {} if oauth_changed else dict(state.get("backoff") or {})
    # Prune expired backoff entries.
    backoff = {
        k: v for k, v in backoff.items()
        if isinstance(v, dict)
        and now_ts - float(v.get("ts") or 0) < CLIPROXY_UNAVAILABLE_RETRY_SECONDS
    }

    groups: dict[tuple[str, str], list[dict]] = {}
    for row in candidates:
        params = row.get("litellm_params") or {}
        model_str = str(params.get("model") or "").strip()
        upstream = model_str.split("/", 1)[-1] if "/" in model_str else model_str
        if upstream not in catalog_ids:
            continue
        api_base = str(params.get("api_base") or "").strip()
        key = (model_str, api_base)
        state_key = f"{model_str}@{api_base}"
        if state_key in backoff:
            continue
        groups.setdefault(key, []).append(row)

    if not groups:
        if state_path and apply:
            atomic(state_path, {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "oauth_fingerprint": current_oauth_fp,
                "backoff": backoff,
            })
        return []

    def _probe_keys(keys_to_probe: list[tuple[str, str]]) -> dict[tuple[str, str], tuple[str, str]]:
        rep_ids = [groups[k][0]["model_info"]["id"] for k in keys_to_probe]
        if workers > 1 and len(rep_ids) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(rep_ids))) as pool:
                raw_results = list(pool.map(health, rep_ids))
        else:
            raw_results = [health(mid) for mid in rep_ids]
        return {k: _parse_health(r) for k, r in zip(keys_to_probe, raw_results)}

    ordered_keys = sorted(groups)
    probed = _probe_keys(ordered_keys)

    # CLIProxyAPI caches multi-day force-cooldowns in RAM, so probes return 429
    # model_cooldown in 1ms without ever calling the upstream API. When any
    # blocked deployment hits model_cooldown, clear CLIProxyAPI's stale cooldown
    # once per sync window and immediately re-probe those models live.
    cooldown_keys = [
        k for k in ordered_keys if _is_cliproxy_cooldown(*probed[k])
    ]
    last_reset_ts = float(state.get("last_cooldown_reset_ts") or 0.0)
    reset_method = None
    if (
        cooldown_keys
        and apply
        and (oauth_changed or (now_ts - last_reset_ts) >= reset_min_interval)
    ):
        resetter = reset_cooldowns or (
            lambda: reset_cliproxy_cooldowns(
                cliproxy_base, management_key=management_key
            )
        )
        reset_method = resetter()
        if reset_method:
            last_reset_ts = now_ts
            probed.update(_probe_keys(cooldown_keys))

    actions: list[dict] = []
    for key in ordered_keys:
        verdict, _detail = probed[key]
        model_str, api_base = key
        state_key = f"{model_str}@{api_base}"
        if verdict == "ok":
            backoff.pop(state_key, None)
            for row in groups[key]:
                info = row.get("model_info") or {}
                deployment = info["id"]
                if apply:
                    res = client.post("/model/unblock", {"model_id": deployment})
                    if res.get("error_status") is not None:
                        actions.append({
                            "model": row["model_name"],
                            "deployment": deployment,
                            "upstream": model_str,
                            "action": "failed",
                            "detail": f"unblock_{res['error_status']}",
                        })
                        continue
                    clear_fields = {}
                    if info.get(MARKER):
                        clear_fields[MARKER] = False
                    if info.get(BUDGET_MARKER):
                        clear_fields[BUDGET_MARKER] = False
                    if clear_fields:
                        client.patch(
                            f"/model/{deployment}/update",
                            {"model_info": clear_fields},
                        )
                info["blocked"] = False
                if info.get(MARKER):
                    info[MARKER] = False
                if info.get(BUDGET_MARKER):
                    info[BUDGET_MARKER] = False
                reason = (
                    f"cliproxy_recovered_after_{reset_method}"
                    if reset_method and key in cooldown_keys
                    else "cliproxy_recovered"
                )
                actions.append({
                    "model": row["model_name"],
                    "deployment": deployment,
                    "upstream": model_str,
                    "action": "unblocked" if apply else "would_unblock",
                    "reason": reason,
                })
        elif verdict == "unhealthy":
            # Permanent account/model errors (e.g. not_found) back off for 6h;
            # rate_limited / cooldown / unknown are NOT backed off so quota resets
            # are caught on the next sync.
            backoff[state_key] = {"ts": now_ts, "verdict": verdict}

    if state_path and apply:
        atomic(state_path, {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "oauth_fingerprint": current_oauth_fp,
            "last_cooldown_reset_ts": last_reset_ts,
            "backoff": backoff,
        })
    return actions
