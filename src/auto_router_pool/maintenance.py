"""Operational upkeep that used to be separate cron scripts.

Three jobs with different cadences but one owner: keep providers that reject
``prompt_cache_key`` from receiving it, keep pooled deployments' cooldown short
and clear a cooldown only when two real answers prove recovery, and produce a
bounded weekly health report. None of them changes routing.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import urllib.parse

from . import catalog
from .extra_catalogs import atomic

REDIS_CONTAINER = os.environ.get("AGENT_GATEWAY_REDIS_CONTAINER", "agent-gateway-redis")
REDIS_CONF = Path(os.environ.get(
    "AGENT_GATEWAY_REDIS_CONF",
    Path.home() / ".local/share/agent-gateway-redis/redis/redis.conf",
))
COOLDOWN_CAP = 30
COOLDOWN_NAMES = (
    "gpt-6-luna", "gpt-6.1-sol", "gpt-6-sol", "claude-opus-5-5",
    "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1",
    "gpt-6-astra", "gpt-5.5",
)
COOLDOWN_PROBES = (
    ("348 + 259를 계산하고 숫자만 출력하세요.", "607"),
    ("Output exactly the lowercase word: ready", "ready"),
)
HEALTH_MODES = {"chat", "responses"}
MAX_HEALTH_ROWS = 1024


def prompt_cache_drop_pass(client, rows: list[dict], apply: bool = False) -> list[dict]:
    """Patch deployments that would be sent a parameter their provider refuses."""
    changes = []
    for row in rows:
        params = row.get("litellm_params") or {}
        dropped = catalog.prompt_cache_drop(params)
        if dropped is params:
            continue
        identifier = (row.get("model_info") or {}).get("id")
        if not identifier:
            continue
        if apply:
            result = client.patch(
                f"/model/{urllib.parse.quote(identifier, safe='')}/update",
                {"litellm_params": {
                    "additional_drop_params": dropped["additional_drop_params"],
                }},
            )
            if result.get("error_status") is not None:
                changes.append({
                    "model": row["model_name"], "deployment": identifier,
                    "action": "failed", "detail": f"error_{result['error_status']}",
                })
                continue
        changes.append({
            "model": row["model_name"], "deployment": identifier,
            "action": "updated" if apply else "would_update",
        })
    return changes


def provider_prefix_pass(client, rows: list[dict], apply: bool = False) -> list[dict]:
    """Ensure every deployment's model string names a provider.

    LiteLLM routes by the ``provider/model`` form; a bare model name such as
    ``gemini-3.8-flash-high`` has no transport and fails with "unknown
    provider". The gateway is OpenAI-compatible, so a missing prefix is
    repaired to ``openai/`` rather than guessed from the model name.
    """
    changes = []
    for row in rows:
        params = row.get("litellm_params") or {}
        model = str(params.get("model") or "")
        if not model or "/" in model:
            continue
        identifier = (row.get("model_info") or {}).get("id")
        if not identifier:
            continue
        repaired = "openai/" + model
        if apply:
            result = client.patch(
                f"/model/{urllib.parse.quote(identifier, safe='')}/update",
                {"litellm_params": {"model": repaired}},
            )
            if result.get("error_status") is not None:
                changes.append({
                    "model": row["model_name"], "deployment": identifier,
                    "action": "failed", "detail": f"error_{result['error_status']}",
                })
                continue
        changes.append({
            "model": row["model_name"], "deployment": identifier,
            "action": "updated" if apply else "would_update",
            "from": model, "to": repaired,
        })
    return changes


def context_sync_pass(client, rows: list[dict], prices: dict | None = None,
                      apply: bool = False) -> list[dict]:
    """Fill missing context windows from LiteLLM's own model catalogue.

    Deployments registered without an upstream catalogue entry keep no
    ``max_input_tokens``, so the pool treats them as below the 272k floor.
    LiteLLM's price/context map is keyed by model name; a tail match is enough
    because the gateway often stores a provider-prefixed variant of the same
    model. Only a missing window is touched, so an operator's explicit value
    wins.
    """
    by_tail: dict[str, int] = {}
    for key, entry in (prices or {}).items():
        if not isinstance(entry, dict):
            continue
        value = entry.get("max_input_tokens")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            continue
        by_tail.setdefault(str(key).split("/")[-1], value)

    changes = []
    for row in rows:
        info = row.get("model_info") or {}
        if isinstance(info.get("max_input_tokens"), int) and not isinstance(
            info.get("max_input_tokens"), bool
        ) and info.get("max_input_tokens", 0) > 0:
            continue
        identifier = info.get("id")
        if not identifier:
            continue
        tail = str(row.get("model_name") or "").split("/")[-1]
        value = by_tail.get(tail)
        if value is None:
            continue
        if apply:
            result = client.patch(
                f"/model/{urllib.parse.quote(identifier, safe='')}/update",
                {"model_info": {"max_input_tokens": value}},
            )
            if result.get("error_status") is not None:
                changes.append({
                    "model": row["model_name"], "deployment": identifier,
                    "action": "failed", "detail": f"error_{result['error_status']}",
                })
                continue
        changes.append({
            "model": row["model_name"], "deployment": identifier,
            "action": "updated" if apply else "would_update",
            "max_input_tokens": value,
        })
    return changes


def pooled(params: dict) -> bool:
    """A CLIProxyAPI/Cursor bridge deployment whose cooldown this host owns."""
    from urllib.parse import urlsplit

    url = urlsplit(params.get("api_base") or "")
    if url.hostname not in ("127.0.0.1", "192.168.68.3"):
        return False
    credential = params.get("litellm_credential_name")
    if url.port == 8317 and credential == "cliproxyapi-shared":
        return "gemini" not in str(params.get("model") or "").lower()
    if url.port == 8319 and credential == "cursor-chat-bridge":
        return True
    return False


def redis_config(path: Path | None = None) -> dict:
    path = path or REDIS_CONF
    config = {}
    for line in path.read_text().splitlines():
        fields = shlex.split(line, comments=True)
        if fields:
            config[fields[0]] = fields[1:]
    return config


def redis_command(*args, config: dict | None = None,
                  container: str = REDIS_CONTAINER, timeout: int = 15):
    """One Redis command through the same TLS container the gateway uses."""
    config = config or redis_config(REDIS_CONF)
    commands = "AUTH " + json.dumps(config["requirepass"][0]) + "\n"
    commands += " ".join(json.dumps(str(arg)) for arg in args) + "\n"
    proc = subprocess.run(
        [
            "docker", "exec", "-i", container, "redis-cli", "--json",
            "--tls", "--cacert", config["tls-ca-cert-file"][0],
            "-p", config["tls-port"][0],
        ],
        input=commands, text=True, capture_output=True, timeout=timeout,
    )
    lines = proc.stdout.splitlines()
    if proc.returncode or len(lines) != 2 or json.loads(lines[0]) != "OK":
        raise RuntimeError("redis command failed")
    return json.loads(lines[1])


def _probe_fixture(client, model: str, prompt: str, expected: str) -> dict:
    result = client.post(
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    if result.get("error_status") is not None:
        return {"passed": False, "reason": f"unavailable_{result['error_status']}"}
    try:
        content = result["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return {"passed": False, "reason": "malformed_answer"}
    return {"passed": expected in content.strip().lower()}


def repair_cooldowns(client, rows: list[dict], home: Path, apply: bool = False) -> dict:
    """Cap pooled cooldowns, and clear one only after two real answers."""
    selected = {
        (row.get("model_info") or {}).get("id"): row
        for row in rows
        if pooled(row.get("litellm_params") or {})
        and not (row.get("model_info") or {}).get("blocked")
        and (row.get("model_info") or {}).get("id")
    }
    to_patch = [
        row for row in selected.values()
        if row["model_info"].get("cooldown_time") != COOLDOWN_CAP
    ]
    changed = []
    if to_patch and apply:
        backup = home / (
            "pooled-cooldown-before-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + ".json"
        )
        atomic(backup, to_patch)
    for row in to_patch:
        identifier = row["model_info"]["id"]
        if apply:
            result = client.patch(f"/model/{identifier}/update", {
                "model_info": {"cooldown_time": COOLDOWN_CAP},
            })
            if result.get("error_status") is not None:
                continue
        changed.append(identifier)

    verified = True
    if changed and apply:
        fresh = client.get("/model/info")
        data = fresh.get("data") if isinstance(fresh, dict) else None
        if isinstance(data, list):
            current = {row["model_info"]["id"]: row for row in data if row.get("model_info")}
            verified = all(
                current.get(identifier, {}).get("model_info", {}).get("cooldown_time") == COOLDOWN_CAP
                for identifier in changed
            )

    results = []
    if not apply:
        return {
            "pooled_deployments": len(selected),
            "changed_this_run": len(to_patch),
            "would_change": [row["model_info"]["id"] for row in to_patch],
            "cleared": 0,
        }

    try:
        config = redis_config(REDIS_CONF)
    except (OSError, KeyError, IndexError):
        return {
            "pooled_deployments": len(selected),
            "changed_this_run": len(changed),
            "verified": verified,
            "cleared": 0,
            "error": "redis_config_unavailable",
        }

    for name in COOLDOWN_NAMES:
        candidates = [
            row for row in selected.values()
            if row["model_name"] == name and row["model_info"]["id"] in selected
        ]
        if not candidates:
            continue
        row = min(candidates, key=lambda item: (item.get("litellm_params") or {}).get("order", 1))
        identifier = row["model_info"]["id"]
        cache_key = "deployment:" + identifier + ":cooldown"
        record = {"model": name, "deployment_id": identifier, "cleared": False}
        try:
            raw = redis_command("GET", cache_key, config=config)
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
            record["error"] = type(error).__name__
            results.append(record)
            continue
        if raw is None:
            record["cooldown_present"] = False
            results.append(record)
            continue
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            try:
                import ast
                value = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                record["error"] = "cooldown_unparseable"
                results.append(record)
                continue
        if not isinstance(value, dict) or not isinstance(value.get("cooldown_time"), (int, float)):
            record["error"] = "cooldown_unparseable"
            results.append(record)
            continue
        record.update(
            cooldown_present=True,
            previous_cooldown_seconds=value.get("cooldown_time"),
        )
        try:
            record["previous_ttl_seconds"] = redis_command("TTL", cache_key, config=config)
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError):
            pass
        checks = [
            _probe_fixture(client, name, prompt, expected)
            for prompt, expected in COOLDOWN_PROBES
        ]
        record["probes"] = checks
        if all(check["passed"] for check in checks):
            script = (
                "if redis.call('GET', KEYS[1]) == ARGV[1] then "
                "return redis.call('DEL', KEYS[1]) else return 0 end"
            )
            try:
                record["cleared"] = redis_command(
                    "EVAL", script, 1, cache_key, raw, config=config
                ) == 1
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
                record["error"] = type(error).__name__
        results.append(record)
    return {
        "pooled_deployments": len(selected),
        "changed_this_run": len(changed),
        "verified": verified,
        "cleared": sum(record["cleared"] for record in results),
        "results": results,
    }


def _health(client, deployment: dict, redact: str = "") -> dict:
    record = dict(deployment)
    if deployment["mode"] not in HEALTH_MODES:
        return {**record, "tested": False, "reason": "media/decisions excluded from text health"}
    started = datetime.datetime.now(datetime.timezone.utc)
    url = "/health?" + urllib.parse.urlencode({"model_id": deployment["id"]})
    data = client.get(url, timeout=85)
    if data.get("error_status") not in (None, 429):
        return {**record, "tested": False, "healthy": None,
                "reason": "health API error", "http_status": data.get("error_status")}
    body = data.get("error_body")
    if not isinstance(body, dict):
        body = data
    if "unhealthy_endpoints" not in body and "healthy_endpoints" not in body:
        return {**record, "tested": False, "healthy": None,
                "reason": "health API could not test deployment"}

    def scoped(key):
        rows = body.get(key) or []
        selected = [row for row in rows if row.get("model_id") == deployment["id"]]
        if selected:
            return selected
        if any(row.get("model_id") for row in rows):
            return []
        return rows

    healthy = scoped("healthy_endpoints")
    unhealthy = scoped("unhealthy_endpoints")
    if healthy:
        record.update(tested=True, healthy=True)
    elif unhealthy:
        error = str(unhealthy[0].get("error", "")).split("\nstack trace:", 1)[0]
        if redact:
            error = error.replace(redact, "[REDACTED]")
        reason = next((
            label for label in (
                "permission_denied", "payment_required", "rate_limit",
                "insufficient_quota", "quota", "balance exhausted",
                "access_terminated_error", "failed_precondition",
                "invalid_argument", "not_found", "timeout", "unsupported",
                "context_window", "auth_unavailable", "401", "403", "429",
            ) if label in error.lower()
        ), "upstream_error")
        status = unhealthy[0].get("exception_status")
        restricted = status in {402, 403, 429} or any(
            term in error.lower()
            for term in (
                "permission_denied", "payment_required", "quota",
                "rate_limit", "rate limit", "balance exhausted",
                "access_terminated_error",
            )
        )
        category = (
            "account_restriction" if restricted
            else "upstream_precondition" if reason == "failed_precondition"
            else "request_error" if status in {400, 404, 422}
            else "upstream_error"
        )
        record.update(
            tested=True, healthy=False, exception_status=status,
            reason=reason, category=category,
        )
    else:
        record.update(tested=False, healthy=None,
                      reason="health API did not return the requested deployment")
    record["elapsed_seconds"] = round(
        (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds(), 2
    )
    return record


def health_report(client, rows: list[dict], home: Path, workers: int = 2,
                  redact: str = "") -> dict:
    """Weekly bounded report; a run never blocks routing or writes a block flag."""
    deployments = [
        {
            "model_name": row["model_name"],
            "upstream": row["litellm_params"]["model"].split("/", 1)[-1],
            "id": row["model_info"]["id"],
            "mode": row["model_info"].get("mode", "unknown"),
            "transport": row["litellm_params"].get("custom_llm_provider"),
        }
        for row in rows
        if (row.get("model_info") or {}).get("id")
    ]
    if len(deployments) > MAX_HEALTH_ROWS:
        raise ValueError("deployment count exceeds the health-check budget")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda item: _health(client, item, redact), deployments))
    summary = {
        "healthy": sum(row.get("healthy") is True for row in results),
        "unhealthy": sum(row.get("healthy") is False for row in results),
        "not_tested": sum(not row["tested"] for row in results),
        "account_restrictions": sum(
            row.get("category") == "account_restriction" for row in results
        ),
        "request_errors": sum(row.get("category") == "request_error" for row in results),
        "other_upstream_errors": sum(row.get("category") == "upstream_error" for row in results),
        "total": len(results),
    }
    report = {
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "summary": summary,
        "results": results,
    }
    atomic(home / "deployment-health-results.json", report)
    return summary
