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

import math

from . import catalog
from .extra_catalogs import atomic
from .tiers import ROUTERS

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
    ``max_input_tokens``, so the pool treats them as below the 128k floor.
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


EFFORT_SUFFIXES = ("-high", "-low", "-medium", "-max", "-xhigh", "-mini")


def effort_suffix_pass(client, rows: list[dict], prices: dict | None = None,
                      apply: bool = False) -> list[dict]:
    """Drop a trailing effort suffix from a model string.

    A registration sometimes stores an effort variant (``gemini-3.8-flash-high``)
    as if it were a model. LiteLLM then has no such upstream model and returns
    404 / "unknown provider". When the unsuffixed model is known to exist, the
    suffix is stripped so the deployment calls the real model.
    """
    known: set[str] = set()
    for key in (prices or {}):
        known.add(str(key).split("/")[-1])
    for row in rows:
        known.add(str(row.get("model_name") or "").split("/")[-1])

    changes = []
    for row in rows:
        params = row.get("litellm_params") or {}
        model = str(params.get("model") or "")
        if not model or "/" not in model:
            continue
        prefix, separator, tail = model.rpartition("/")
        base_tail = None
        for suffix in EFFORT_SUFFIXES:
            if tail.endswith(suffix):
                candidate = tail[: -len(suffix)]
                if candidate in known:
                    base_tail = candidate
                    break
        if base_tail is None:
            continue
        identifier = (row.get("model_info") or {}).get("id")
        if not identifier:
            continue
        repaired = prefix + "/" + base_tail
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


LOCAL_ZERO_COST_PROVIDERS = frozenset({"lm_studio", "ollama", "ollama_chat"})
TOKEN_COST_FIELDS = ("input_cost_per_token", "output_cost_per_token")
MODALITY_COST_FIELDS = (
    "input_cost_per_image",
    "output_cost_per_image",
    "input_cost_per_second",
    "output_cost_per_second",
    "input_cost_per_audio_token",
    "output_cost_per_audio_token",
    "input_cost_per_audio_per_second",
)
ALL_COST_FIELDS = TOKEN_COST_FIELDS + MODALITY_COST_FIELDS
ROUTING_NAME_SUFFIXES = ("-openrouter", "-nim")


def _finite_nonneg(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    val = float(value)
    if not math.isfinite(val) or val < 0:
        return None
    return val


def _extract_costs(entry: dict) -> dict[str, float]:
    costs: dict[str, float] = {}
    for field in ALL_COST_FIELDS:
        val = _finite_nonneg(entry.get(field))
        if val is not None:
            costs[field] = val
    return costs


def _has_any_cost(info: dict) -> bool:
    return any(_finite_nonneg(info.get(field)) is not None for field in ALL_COST_FIELDS)


def _price_lookup(row: dict, exact_index: dict[str, tuple[str, dict]],
                  tail_index: dict[str, list[tuple[str, dict]]]) -> tuple[str | None, dict[str, float] | None]:
    params = row.get("litellm_params") or {}
    info = row.get("model_info") or {}
    model_str = str(params.get("model") or "").strip()
    model_name = str(row.get("model_name") or "").strip()
    provider = str(params.get("custom_llm_provider") or "").strip().lower()
    if not provider and "/" in model_str:
        provider = model_str.split("/", 1)[0].lower()
    mode = str(info.get("mode") or "chat").lower()

    # 1. Local inference providers and explicit :free upstream tags are $0.0.
    if provider in LOCAL_ZERO_COST_PROVIDERS or model_str.lower().endswith(":free"):
        return None, {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}

    # 2. Exact matches against LiteLLM price map.
    candidates: list[tuple[int, str, dict]] = []
    for priority, key in (
        (0, model_str.lower()),
        (1, model_name.lower()),
        (2, str(info.get("base_model") or "").strip().lower()),
    ):
        if key and key in exact_index:
            orig_key, entry = exact_index[key]
            candidates.append((priority, orig_key, entry))

    # 3. Tail matches (including stripping routing suffixes like -openrouter / -nim).
    tails: list[str] = []
    for raw in (model_str, model_name):
        if not raw:
            continue
        t = raw.rsplit("/", 1)[-1].split(":", 1)[0].lower()
        if t and t not in tails:
            tails.append(t)
        for suf in ROUTING_NAME_SUFFIXES:
            if t.endswith(suf):
                stripped = t[: -len(suf)]
                if stripped and stripped not in tails:
                    tails.append(stripped)

    for tail in tails:
        for orig_key, entry in tail_index.get(tail, ()):
            entry_prov = str(entry.get("litellm_provider") or "").lower()
            entry_mode = str(entry.get("mode") or "chat").lower()
            score = 10
            if entry_prov and provider and entry_prov == provider:
                score -= 4
            if entry_mode == mode:
                score -= 2
            if entry_prov in ("openai", "anthropic", "gemini", "xai", "openrouter"):
                score -= 1
            candidates.append((score, orig_key, entry))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: (item[0], item[1]))
    for _, orig_key, entry in candidates:
        costs = _extract_costs(entry)
        if not costs:
            continue
        if mode in ("chat", "responses", "embedding") and not any(k in costs for k in TOKEN_COST_FIELDS):
            continue
        return orig_key, costs
    return None, None


def price_sync_pass(client, rows: list[dict], prices: dict | None = None,
                    apply: bool = False) -> list[dict]:
    """Backfill missing pricing metadata on deployments from local/free rules and LiteLLM prices.

    Only deployments missing all cost fields are patched, so an operator's
    explicit pricing always wins. Internal auto-router groups are skipped.
    """
    exact_index: dict[str, tuple[str, dict]] = {}
    tail_index: dict[str, list[tuple[str, dict]]] = {}
    for key, entry in (prices or {}).items():
        if not isinstance(entry, dict):
            continue
        if not _extract_costs(entry):
            continue
        low = str(key).lower()
        exact_index.setdefault(low, (str(key), entry))
        tail = low.rsplit("/", 1)[-1]
        tail_index.setdefault(tail, []).append((str(key), entry))

    changes = []
    for row in rows:
        model_name = str(row.get("model_name") or "")
        params = row.get("litellm_params") or {}
        model_str = str(params.get("model") or "")
        if (
            not model_name
            or model_name in ROUTERS
            or model_name.startswith("__")
            or model_str.startswith("auto_router/")
        ):
            continue
        info = row.get("model_info") or {}
        if _has_any_cost(params) or _has_any_cost(info):
            continue
        identifier = info.get("id")
        if not identifier:
            continue

        matched_key, costs = _price_lookup(row, exact_index, tail_index)
        if not costs:
            continue

        patch_body: dict = {"litellm_params": dict(costs)}
        if matched_key and not info.get("base_model"):
            patch_body["model_info"] = {"base_model": matched_key}

        if apply:
            result = client.patch(
                f"/model/{urllib.parse.quote(identifier, safe='')}/update",
                patch_body,
            )
            if result.get("error_status") is not None:
                changes.append({
                    "model": model_name, "deployment": identifier,
                    "action": "failed", "detail": f"error_{result['error_status']}",
                })
                continue
        entry_rec = {
            "model": model_name, "deployment": identifier,
            "action": "updated" if apply else "would_update",
            "costs": costs,
        }
        if matched_key:
            entry_rec["price_key"] = matched_key
        changes.append(entry_rec)
    return changes


CHAT_ONLY_API_HOSTS = frozenset({"integrate.api.nvidia.com"})


def chat_only_transport_pass(client, rows: list[dict], apply: bool = False) -> list[dict]:
    """Use ``hosted_vllm`` for chat-only OpenAI-compatible hosts.

    LiteLLM routes Anthropic ``/v1/messages`` requests for ``openai/``
    deployments through the OpenAI Responses API (``/v1/responses``) and
    rejects ``reasoning_effort`` for unknown OpenAI model names. Hosts such as
    NVIDIA NIM only expose ``/v1/chat/completions``, so registering them as
    ``openai/`` breaks ``/v1/messages`` with 404. Moving them to
    ``hosted_vllm/`` keeps ``/v1/chat/completions`` and supports reasoning
    parameters on both OpenAI and Anthropic surfaces.
    """
    changes = []
    for row in rows:
        params = row.get("litellm_params") or {}
        api_base = str(params.get("api_base") or "")
        host = (urllib.parse.urlparse(api_base).hostname or "").casefold()
        if host not in CHAT_ONLY_API_HOSTS:
            continue
        model = str(params.get("model") or "")
        custom_provider = str(params.get("custom_llm_provider") or "")
        info = row.get("model_info") or {}
        info_provider = str(info.get("litellm_provider") or "")
        needs_model = model.startswith("openai/")
        needs_provider = custom_provider == "openai" or info_provider == "openai"
        if not needs_model and not needs_provider:
            continue
        identifier = info.get("id")
        if not identifier:
            continue
        repaired = "hosted_vllm/" + model[len("openai/"):] if needs_model else model
        payload: dict = {
            "litellm_params": {
                "model": repaired,
                "custom_llm_provider": "hosted_vllm",
            }
        }
        if info_provider == "openai":
            payload["model_info"] = {"litellm_provider": "hosted_vllm"}
        if apply:
            result = client.patch(
                f"/model/{urllib.parse.quote(identifier, safe='')}/update",
                payload,
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
