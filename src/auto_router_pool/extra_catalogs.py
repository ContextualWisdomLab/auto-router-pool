"""Provider catalogues other than the Experiential one.

Each family has its own discovery credential and its own safe transport, so each
keeps its own plan. The shared rules are the ones that matter: only deployments
this package owns are written, a route owned by another credential is preserved,
and no block flag is touched -- the quarantine sweep owns those, because one
probe cannot prove an account's credit state.
"""
from __future__ import annotations

import datetime
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from . import catalog
from . import client as client_mod

PRICE_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
CURSOR_BRIDGE = os.environ.get("CURSOR_BRIDGE_BASE", "http://192.168.68.3:8319")
CURSOR_CLOUD_PATH = "/cursor-cloud/v1/models"
CURSOR_CREDENTIAL = "cursor-chat-bridge"
CURSOR_OWNER = "cursor-chat-base-sync-v1"
CURSOR_PROVIDERS = {
    "openai", "anthropic", "gemini", "xai", "moonshot", "zai",
    "cognition", "deepseek", "cursor",
}
CURSOR_MAX = 1024

ANTHROPIC_CREDENTIAL = "anthropic-api"
ANTHROPIC_API = "https://api.anthropic.com"
ANTHROPIC_OWNER = "anthropic-api-sync-v1"
ANTHROPIC_MAX = 1024

LMSTUDIO_CREDENTIAL = os.environ.get("LMSTUDIO_CREDENTIAL", "lmstudio")
LMSTUDIO_OWNER = "lmstudio-sync-v1"
LMSTUDIO_MAX = 1024

ARGON_PUBLIC = "gemini-4-argon"
ARGON_UPSTREAMS = ("gemini-4-argon", "gemini-4-argon-high")
ARGON_CREDENTIAL = "cliproxyapi-shared"
ARGON_OWNER = "cliproxy-argon-sync-v1"
ARGON_INPUT_LIMIT = 1048576
ARGON_OUTPUT_LIMIT = 1048576
ARGON_MAX = 1024
ARGON_COPY_FIELDS = (
    "input_cost_per_token", "output_cost_per_token",
    "supports_function_calling", "supports_parallel_function_calling",
    "supports_vision", "supports_pdf_input", "supports_reasoning",
    "supports_response_schema", "supports_native_streaming",
    "supports_system_messages", "supports_tool_choice", "supports_web_search",
    "supports_url_context", "reasoning_effort_levels", "default_reasoning_effort",
    "thinking_always_on", "supported_openai_params", "supported_endpoints",
    "supported_modalities", "supported_output_modalities", "litellm_provider",
)


def atomic(path: Path, value) -> None:
    """Write JSON through a sibling temp file, so a reader never sees half."""
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# --------------------------------------------------------------------------
# Cursor Cloud: a read-only snapshot, because those are agent catalog entries,
# not synchronous chat deployments.


def sync_cursor_cloud(client, home: Path) -> dict:
    payload = client.get(CURSOR_CLOUD_PATH)
    if payload.get("error_status") is not None:
        return {"error": f"http_{payload['error_status']}"}
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("expected the real Cursor cloud catalog, not local models")
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate Cursor cloud model ids")
    path = home / "cursor-cloud-model-catalog.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    old = {item.get("id") for item in previous.get("items") or []}
    new = set(ids)
    atomic(path, {
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "items": items,
        "added": sorted(new - old),
        "removed": sorted(old - new),
        "kind": "Cursor Cloud Agent catalog, not synchronous chat deployments",
    })
    return {"models": len(ids), "added": sorted(new - old), "removed": sorted(old - new)}


# --------------------------------------------------------------------------
# Cursor chat bridge: price-verified base routes behind the OAuth bridge.


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a loopback bearer to a redirected host.
        raise RuntimeError("unexpected HTTP redirect")


def request(url: str, token: str | None = None, body=None, method=None, timeout=60):
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        url,
        headers=headers,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(req, timeout=timeout) as response:
            raw = response.read(32 * 1024 * 1024 + 1)
            if len(raw) > 32 * 1024 * 1024:
                raise ValueError("JSON response exceeds 32 MiB")
            return json.loads(raw)
    except urllib.error.HTTPError as error:
        # Never print a response body, which may echo a credential.
        raise RuntimeError("HTTP " + str(error.code) + " from " + urllib.parse.urlsplit(url).path) from None


def rows_from(catalog_payload) -> list[dict]:
    if not isinstance(catalog_payload, dict) or not isinstance(catalog_payload.get("data"), list):
        raise ValueError("expected an OpenAI model list with a data array")
    rows = catalog_payload["data"]
    if not rows or any(
        not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]
        for row in rows
    ):
        raise ValueError("empty or invalid chat discovery")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate discovered model ids")
    return rows


def retired_names(home: Path) -> set[str]:
    path = home / "retired-commercial-models.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    if not isinstance(data, list) or any(not isinstance(item, str) or not item for item in data):
        raise ValueError("retired commercial model list is invalid")
    return set(data)


def cursor_matches(catalog_payload, prices: dict, retired: set[str] | None = None):
    """Price-verified public bases, and everything rejected with a reason."""
    if not isinstance(prices, dict):
        raise ValueError("expected a pricing dictionary")
    retired = retired or set()
    index = {}
    for key, price in prices.items():
        if not isinstance(price, dict) or price.get("litellm_provider") not in CURSOR_PROVIDERS:
            continue
        publisher = price["litellm_provider"]
        # A publisher-prefixed key is accepted; foreign hosting prices are not.
        if "/" in key and key.split("/", 1)[0] != publisher:
            continue
        index.setdefault(key.rsplit("/", 1)[-1], []).append((key, price))

    approved, audit = [], []
    for row in rows_from(catalog_payload):
        upstream = row["id"]
        public = upstream.removeprefix("cursor-")
        if public in retired:
            audit.append({"upstream": upstream, "reason": "retired non-current commercial model"})
            continue
        if public.endswith((
            "-high", "-low", "-medium", "-max", "-thinking", "-fast",
            "-highspeed", "-xhigh", "-minimal", "-none",
        )):
            audit.append({"upstream": upstream, "reason": "reasoning/speed alias, not a base"})
            continue
        candidates = index.get(public, [])
        hint = {"google": "gemini", "zhipu": "zai"}.get(row.get("owned_by"), row.get("owned_by"))
        if hint in CURSOR_PROVIDERS and hint != "cursor":
            candidates = [item for item in candidates if item[1]["litellm_provider"] == hint]
        publishers = {price["litellm_provider"] for _, price in candidates}
        if len(publishers) != 1:
            audit.append({"upstream": upstream, "reason": "missing or ambiguous publisher pricing"})
            continue
        candidates.sort(key=lambda item: (item[0] != public, item[0]))
        key, price = candidates[0]
        if price.get("mode") != "chat":
            audit.append({"upstream": upstream, "reason": "not a chat pricing entry"})
            continue
        rates = [price.get(field) for field in ("input_cost_per_token", "output_cost_per_token")]
        if any(
            isinstance(rate, bool) or not isinstance(rate, (int, float))
            or not math.isfinite(rate) or rate <= 0
            for rate in rates
        ):
            audit.append({"upstream": upstream, "reason": "missing positive input/output rates"})
            continue
        caps = [price.get(field) for field in ("max_input_tokens", "max_output_tokens")]
        if any(isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0 for cap in caps):
            audit.append({"upstream": upstream, "reason": "missing verified capacity"})
            continue
        limits = {field: price[field] for field in ("max_input_tokens", "max_output_tokens")}
        for field in limits:
            supplied = row.get(field)
            if isinstance(supplied, int) and not isinstance(supplied, bool) and supplied > 0:
                limits[field] = min(limits[field], supplied)
        approved.append({
            "upstream": upstream,
            "public": public,
            "pricing_key": key,
            "publisher": price["litellm_provider"],
            "bridge_limits": limits,
        })

    # Prefer the canonical wire base over an explicitly cursor-prefixed duplicate.
    selected = {}
    for match in sorted(approved, key=lambda item: (item["upstream"] != item["public"], item["upstream"])):
        if match["public"] in selected:
            audit.append({"upstream": match["upstream"], "reason": "duplicate exact base route"})
        else:
            selected[match["public"]] = match
    return list(selected.values()), audit


def cursor_identity(match: dict) -> str:
    return match["upstream"] + "|" + match["public"]


def cursor_payload(match: dict, prices: dict) -> dict:
    provider = match["publisher"]
    endpoint = {"anthropic": CURSOR_BRIDGE, "gemini": CURSOR_BRIDGE + "/v1beta"}.get(
        provider, CURSOR_BRIDGE + "/v1"
    )
    # Cursor-native Composer has no assumed provider handler; only verified
    # rates may admit it, and its protocol stays OpenAI-compatible.
    handler = "openai" if provider == "cursor" else provider
    price = prices[match["pricing_key"]]
    info = {
        key: value for key, value in price.items()
        if key.startswith(("supports_", "supported_", "max_", "input_cost_", "output_cost_", "cache_"))
    }
    limits = match.get("bridge_limits", {"max_input_tokens": 200000, "max_output_tokens": 64000})
    for field, ceiling in limits.items():
        info[field] = min(info[field], ceiling)
    info["max_tokens"] = min(price.get("max_tokens", info["max_output_tokens"]), info["max_output_tokens"])
    info.update({
        "base_model": match["pricing_key"],
        "mode": "chat",
        "cursor_chat_sync": CURSOR_OWNER,
        "cursor_chat_identity": cursor_identity(match),
        "pricing_estimate": "Publisher API estimate; not Cursor invoice",
        "cap_origin": "Bridge current registry defaults, conservatively bounded; not upstream guarantee",
    })
    params = catalog.prompt_cache_drop({
        "model": handler + "/" + match["upstream"],
        "custom_llm_provider": handler,
        "api_base": endpoint,
        "litellm_credential_name": CURSOR_CREDENTIAL,
        "order": 3,
    })
    return {"model_name": match["public"], "litellm_params": params, "model_info": info}


def cursor_reconcile(approved, prices, existing, state, api, save):
    """Create, update or preserve one owned deployment per verified base."""
    if not isinstance(existing, list) or len(existing) > CURSOR_MAX:
        raise ValueError("invalid registry or registry exceeds its ceiling")
    deployments = state.setdefault("deployments", {})
    created = updated = unchanged = manual = 0
    capacity = len(existing)
    for match in approved:
        ident = cursor_identity(match)
        expected = cursor_payload(match, prices)
        wire = expected["litellm_params"]["model"]
        same = [
            row for row in existing
            if row.get("model_name") == match["public"]
            and row.get("litellm_params", {}).get("model") == wire
            and row.get("litellm_params", {}).get("api_base") == expected["litellm_params"]["api_base"]
        ]
        candidates = [
            row for row in same
            if row.get("model_info", {}).get("cursor_chat_sync") == CURSOR_OWNER
            and row["model_info"].get("cursor_chat_identity") == ident
        ]
        tracked = deployments.get(ident, {})
        if tracked.get("id"):
            candidates = [row for row in candidates if row["model_info"].get("id") == tracked["id"]]
        if len(candidates) > 1:
            raise ValueError("multiple owned deployments for " + match["public"])
        if not candidates and same:
            manual += 1  # Never adopt or overwrite a manually configured bridge route.
            continue
        if candidates:
            row = candidates[0]
            identifier = row["model_info"]["id"]
            deployments[ident] = {
                "id": identifier, "upstream": match["upstream"], "public": match["public"],
            }
            save(state)
            equal = (
                all(row.get("litellm_params", {}).get(k) == v
                    for k, v in expected["litellm_params"].items())
                and all(row["model_info"].get(k) == v
                        for k, v in expected["model_info"].items())
            )
            if equal:
                unchanged += 1
                continue
            expected["model_info"]["id"] = identifier
            api("/model/" + urllib.parse.quote(identifier, safe="") + "/update", expected, "PATCH")
            updated += 1
        else:
            if capacity >= CURSOR_MAX:
                raise ValueError("registry capacity reached; no capacity settings changed")
            # Persist intent before POST, so an interrupted successful POST can
            # be recovered next run by its exact owner + identity markers.
            deployments[ident] = {"upstream": match["upstream"], "public": match["public"], "pending": True}
            save(state)
            result = api("/model/new", expected, "POST")
            identifier = (result.get("model_info") or {}).get("id") or result.get("model_id")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("registration response has no deployment id; rerun to recover")
            deployments[ident] = {
                "id": identifier, "upstream": match["upstream"], "public": match["public"],
            }
            save(state)
            capacity += 1
            created += 1
    return {"created": created, "updated": updated, "unchanged": unchanged, "manual_preserved": manual}


def cursor_family(identifier: str):
    """Mirror cursorVariant in the bridge, never fuzzy matching."""
    base, effort, fast = identifier.removeprefix("cursor-"), False, False
    while "-" in base:
        head, suffix = base.rsplit("-", 1)
        if suffix == "fast":
            fast = True
        elif suffix == "thinking":
            pass
        elif suffix in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            if effort:
                break
            effort = True
            if suffix == "high" and head.endswith("-extra"):
                head = head.removesuffix("-extra")
        else:
            break
        base = head
    return base, fast


def authoritative_observation(evidence, now=None):
    now = datetime.datetime.now(datetime.timezone.utc).timestamp() if now is None else now
    if not isinstance(evidence, dict) or evidence.get("authoritative") is not True:
        return None
    if evidence.get("source", "live") != "live" or evidence.get("last_attempt_ok", True) is not True:
        return None
    ids = evidence.get("model_ids")
    if (
        not isinstance(ids, list) or not ids or len(ids) > 512
        or any(not isinstance(item, str) or not item or len(item) > 256 for item in ids)
        or len(set(ids)) != len(ids)
    ):
        return None
    try:
        stamp = datetime.datetime.fromisoformat(evidence["observed_at"].replace("Z", "+00:00"))
        observed = stamp.timestamp() if stamp.tzinfo else None
        if observed is not None and 0 <= now - observed <= 3600:
            return observed, ids
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    return None


def cursor_retirement(catalog_payload, existing, state, api, save, evidence=None, now=None):
    """Fresh raw account families govern retirement; aliases cannot prove it."""
    observation = authoritative_observation(evidence, now)
    verified = observation is not None
    observed, ids = observation if verified else (None, [row["id"] for row in rows_from(catalog_payload)])
    present = {cursor_family(identifier)[0] for identifier in ids}
    pending, deleted = [], []
    deployments = state["deployments"]
    for ident, tracked in list(deployments.items()):
        families = {
            cursor_family(tracked.get("upstream", ""))[0],
            cursor_family(tracked.get("public", ""))[0],
        }
        if families & present:
            tracked.pop("retirement", None)
            continue
        row = next((item for item in existing if item.get("model_info", {}).get("id") == tracked.get("id")), None)
        if row is None:
            continue
        info, params = row.get("model_info", {}), row.get("litellm_params", {})
        expected_ident = tracked.get("upstream", "") + "|" + tracked.get("public", "")
        owned = (
            ident == expected_ident
            and info.get("cursor_chat_sync") == CURSOR_OWNER
            and info.get("cursor_chat_identity") == ident
            and row.get("model_name") == tracked.get("public")
            and params.get("model", "").split("/", 1)[-1] == tracked.get("upstream")
            and params.get("litellm_credential_name") == CURSOR_CREDENTIAL
            and params.get("api_base") in {CURSOR_BRIDGE, CURSOR_BRIDGE + "/v1", CURSOR_BRIDGE + "/v1beta"}
        )
        if not owned:
            continue
        if not verified:
            tracked.pop("retirement", None)
            pending.append({
                "upstream": tracked["upstream"],
                "reason": "missing model; no fresh authoritative retirement proof",
            })
            continue
        history = tracked.setdefault("retirement", {
            "first_seen": observed, "last_seen": observed, "observations": 1,
        })
        if observed - history["last_seen"] >= 3600:
            history["last_seen"] = observed
            history["observations"] += 1
        if history["observations"] >= 3 and observed - history["first_seen"] >= 86400:
            save(state)  # Persist evidence and exact ownership before deleting.
            api("/model/delete", {"id": tracked["id"]}, "POST")
            deleted.append({"upstream": tracked["upstream"], "public": tracked["public"], "id": tracked["id"]})
            del deployments[ident]
            save(state)
        else:
            pending.append({
                "upstream": tracked["upstream"],
                "reason": "authoritative missing model; waiting for three hourly observations over 24 hours",
            })
    save(state)
    return {"retirement_pending": pending, "retired": deleted, "deletions": len(deleted)}


def sync_cursor_chat(client, home: Path, rows: list[dict], apply: bool = False,
                     evidence=None) -> dict:
    key_path = Path(os.environ.get(
        "CURSOR_BRIDGE_KEY", home.parent / "cursor-chat-bridge" / "downstream.key"
    ))
    if not key_path.exists():
        return {"skipped": "no_bridge_key"}
    downstream = key_path.read_text().strip()
    if not downstream:
        return {"skipped": "empty_bridge_key"}
    catalog_payload = request(CURSOR_BRIDGE + "/v1/models", downstream)
    prices = request(PRICE_URL)
    approved, audit = cursor_matches(catalog_payload, prices, retired_names(home))
    report = {
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "discovered": len(rows_from(catalog_payload)),
        "public_bases": len(approved),
        "matches": approved,
        "audit": audit,
        "price_source": PRICE_URL,
        "priority": 3,
        "deletions": 0,
    }
    if apply:
        state_path = home / "cursor-chat-model-sync-state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            "owner": CURSOR_OWNER, "deployments": {},
        }
        if state.get("owner") != CURSOR_OWNER or not isinstance(state.get("deployments"), dict):
            raise ValueError("invalid ownership state")

        def save(value):
            atomic(state_path, value)

        def api(path, body=None, method=None):
            if method == "POST":
                return client.post(path, body)
            if method == "PATCH":
                return client.patch(path, body)
            return client.get(path)

        report.update(cursor_reconcile(approved, prices, rows, state, api, save))
        report.update(cursor_retirement(catalog_payload, rows, state, api, save, evidence))
        atomic(home / "cursor-chat-model-sync-audit.json", report)
    return report


# --------------------------------------------------------------------------
# Anthropic official catalog: the credential lives in the gateway database, so
# discovery runs inside the LiteLLM container and never prints the key.


def litellm_container() -> str:
    configured = os.environ.get("LITELLM_CONTAINER")
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=15,
        )
        names = sorted(
            name for name in result.stdout.split()
            if name.startswith("agent-gateway-litellm")
        )
        if names:
            return names[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return "agent-gateway-litellm-1"


ANTHROPIC_DISCOVERY = r'''
import os, json, hashlib, base64, urllib.parse, urllib.request
import psycopg, nacl.secret
from urllib.parse import urlsplit, urlunsplit
u = urlsplit(os.environ["DATABASE_URL"])
with psycopg.connect(urlunsplit((u.scheme, u.netloc, u.path, "", "")), connect_timeout=5,
                     options="-c statement_timeout=5000 -c default_transaction_read_only=on") as conn:
    rows = conn.execute("SELECT credential_values FROM \"LiteLLM_CredentialsTable\" WHERE credential_name=%s",
                        ("anthropic-api",)).fetchall()
if len(rows) != 1: raise SystemExit(2)
salt = os.environ.get("LITELLM_SALT_KEY") or os.environ["LITELLM_MASTER_KEY"]
box = nacl.secret.SecretBox(hashlib.sha256(salt.encode()).digest())
def dec(s):
    if not isinstance(s, str): return s
    if s.startswith("v2:gcm:"):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        b = base64.urlsafe_b64decode(s[7:])
        return AESGCM(hashlib.sha256(salt.encode()).digest()).decrypt(b[:12], b[12:], None).decode()
    try: return box.decrypt(base64.urlsafe_b64decode(s)).decode()
    except Exception: return s
values = {k: dec(v) for k, v in rows[0][0].items()}
if values.get("api_base", "").rstrip("/") != "https://api.anthropic.com": raise SystemExit(3)
after, pages = None, []
while True:
    url = "https://api.anthropic.com/v1/models?limit=100" + (("&after=" + urllib.parse.quote(after)) if after else "")
    req = urllib.request.Request(url, headers={"x-api-key": values["api_key"], "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=25) as response: page = json.load(response)
    pages.extend(page.get("data") or [])
    if not page.get("has_more"): break
    after = page.get("last_id")
    if not after or len(pages) > 512: raise SystemExit(4)
print(json.dumps(pages))
'''


def anthropic_catalog() -> list[dict]:
    completed = subprocess.run(
        ["docker", "exec", "-i", litellm_container(), "python", "-"],
        input=ANTHROPIC_DISCOVERY, text=True, capture_output=True, timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError("Anthropic catalog discovery failed with status " + str(completed.returncode))
    models = json.loads(completed.stdout)
    if not isinstance(models, list) or not models:
        raise ValueError("Anthropic catalog is empty")
    return models


def plan_anthropic(models, existing, prices, retired):
    if not isinstance(existing, list) or len(existing) > ANTHROPIC_MAX:
        raise ValueError("invalid registry or registry exceeds its ceiling")
    seen, planned, creates = set(), [], 0
    for model in models:
        identifier = model.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("invalid or duplicate Anthropic model id")
        seen.add(identifier)
        if model.get("lifecycle") != "active" or identifier in retired:
            continue
        limits = {}
        for source, target in (("max_input_tokens", "max_input_tokens"), ("max_tokens", "max_output_tokens")):
            value = model.get(source)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("invalid token limit for " + identifier)
            limits[target] = value
        same = [row for row in existing if row.get("model_name") == identifier]
        owned = [row for row in same if row.get("model_info", {}).get("anthropic_api_sync") == ANTHROPIC_OWNER]
        manual = [
            row for row in same
            if row.get("litellm_params", {}).get("litellm_credential_name") == ANTHROPIC_CREDENTIAL
            and row not in owned
        ]
        if len(owned) > 1 or manual:
            if manual and not owned:
                continue
            raise ValueError("ambiguous official route for " + identifier)
        previous = owned[0] if owned else None
        if previous and previous["model_info"].get("anthropic_api_model_id") != identifier:
            raise ValueError("owned route identity mismatch for " + identifier)
        orders = []
        for row in same:
            if row is previous:
                continue
            order = row.get("litellm_params", {}).get("order", 1)
            if isinstance(order, bool) or not isinstance(order, int):
                raise ValueError("invalid routing order for " + identifier)
            orders.append(order)
        order = previous["litellm_params"].get("order") if previous else (max(orders) if orders else 0) + 1
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("invalid official routing order for " + identifier)
        info = dict(previous["model_info"]) if previous else {}
        info.update({"mode": "chat", "anthropic_api_sync": ANTHROPIC_OWNER,
                     "anthropic_api_model_id": identifier, **limits})
        price = prices.get(identifier, {})
        if price.get("litellm_provider") == "anthropic":
            info["base_model"] = identifier
            for field in ("input_cost_per_token", "output_cost_per_token"):
                if isinstance(price.get(field), (int, float)) and not isinstance(price[field], bool):
                    info[field] = price[field]
        params = catalog.prompt_cache_drop({
            "model": "anthropic/" + identifier,
            "custom_llm_provider": "anthropic",
            "api_base": ANTHROPIC_API,
            "litellm_credential_name": ANTHROPIC_CREDENTIAL,
            "order": order,
        })
        if any(params.get(field) for field in ("api_key", "headers", "extra_headers")):
            raise ValueError("official route contains a static credential for " + identifier)
        body = {"model_name": identifier, "litellm_params": params, "model_info": info}
        if previous:
            same_fields = all(
                previous[section].get(key) == value
                for section in ("litellm_params", "model_info")
                for key, value in body[section].items()
            )
            action = "unchanged" if same_fields else "update"
        else:
            action = "create"
            creates += 1
        planned.append((action, body))
    if len(existing) + creates > ANTHROPIC_MAX:
        raise ValueError("registry ceiling would be exceeded")
    return planned


def sync_anthropic(client, home: Path, rows: list[dict], prices: dict, apply: bool = False) -> dict:
    models = anthropic_catalog()
    planned = plan_anthropic(models, rows, prices, retired_names(home))
    report = {
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "catalog_models": len(models),
        "planned": {
            action: sum(item == action for item, _ in planned)
            for action in ("create", "update", "unchanged")
        },
        "retired_or_inactive_not_added": sorted(
            model["id"] for model in models
            if model.get("id") not in {body["model_name"] for _, body in planned}
        ),
        "no_models_removed": True,
    }
    atomic(home / "anthropic-api-sync-preview.json", [{"action": action, **body} for action, body in planned])
    if apply:
        created = updated = 0
        for action, body in planned:
            if action == "create":
                result = client.post("/model/new", body)
                if result.get("error_status") is not None:
                    raise RuntimeError("gateway refused an Anthropic registration")
                created += 1
            elif action == "update":
                identifier = body["model_info"]["id"]
                result = client.patch(f"/model/{urllib.parse.quote(identifier, safe='')}/update", body)
                if result.get("error_status") is not None:
                    raise RuntimeError("gateway refused an Anthropic update")
                updated += 1
        report["created"] = created
        report["updated"] = updated
    atomic(home / "anthropic-api-sync-report.json", report)
    return report


# --------------------------------------------------------------------------
# CLIProxyAPI / Antigravity route for gemini-4-argon.


def proxy_models(base: str) -> list[str]:
    config_path = Path(os.environ.get("CLIPROXY_CONFIG", Path.home() / ".config/agent-gateway" / "core-config.yaml"))
    keys = config_api_keys(config_path.read_text())
    if not keys:
        raise ValueError("CLIProxyAPI client key is missing")
    request_obj = urllib.request.Request(
        base.rstrip("/") + "/v1/models",
        headers={"Authorization": "Bearer " + keys[0]},
    )
    with urllib.request.urlopen(request_obj, timeout=20) as response:
        payload = json.load(response)
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError("CLIProxyAPI model list is invalid")
    found = []
    for row in rows:
        identifier = row.get("id") if isinstance(row, dict) else None
        if isinstance(identifier, str) and identifier:
            found.append(identifier)
    return found


def config_api_keys(text: str) -> list[str]:
    """The proxy client keys from a JSON or YAML config file.

    The gateway writes ``core-config.yaml`` as JSON despite the extension, so
    reading it structurally is both simpler and safer than a YAML dependency.
    """
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        keys = payload.get("api-keys") or payload.get("api_keys")
        if isinstance(keys, list):
            return [key for key in keys if isinstance(key, str) and key]
        return []
    return _yaml_api_keys(text)


def _yaml_api_keys(text: str) -> list[str]:
    """First list under ``api-keys`` without depending on a YAML library."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() not in ("api-keys:", "api_keys:"):
            continue
        found = []
        for candidate in lines[index + 1:]:
            stripped = candidate.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not stripped.startswith("- "):
                break
            value = stripped[2:].strip().strip("\"'")
            if value:
                found.append(value)
        return found
    return []


def choose_argon(found) -> str | None:
    present = set(found)
    for identifier in ARGON_UPSTREAMS:
        if identifier in present:
            return identifier
    return None


def argon_sibling_info(existing):
    rows = [
        row for row in existing
        if row.get("model_name") == ARGON_PUBLIC
        and row.get("model_info", {}).get("cliproxy_argon_sync") != ARGON_OWNER
    ]
    if not rows:
        return {}
    info = rows[0].get("model_info") or {}
    return {
        field: info[field] for field in ARGON_COPY_FIELDS
        if info.get(field) is not None
    }


def plan_argon(existing, upstream):
    if not isinstance(existing, list) or len(existing) > ARGON_MAX:
        raise ValueError("invalid registry or registry exceeds its ceiling")
    same = [row for row in existing if row.get("model_name") == ARGON_PUBLIC]
    owned = [row for row in same if row.get("model_info", {}).get("cliproxy_argon_sync") == ARGON_OWNER]
    if len(owned) > 1:
        raise ValueError("ambiguous CLIProxyAPI route for " + ARGON_PUBLIC)
    if upstream is None:
        return "absent", None
    previous = owned[0] if owned else None
    if previous and not isinstance(previous.get("model_info", {}).get("id"), str):
        raise ValueError("owned route has no deployment id")
    orders = []
    for row in same:
        if row is previous:
            continue
        order = row.get("litellm_params", {}).get("order", 1)
        if isinstance(order, bool) or not isinstance(order, int):
            raise ValueError("invalid routing order for " + ARGON_PUBLIC)
        orders.append(order)
    order = 1 if 1 not in orders else (max(orders) if orders else 0) + 1
    if previous and 1 in orders:
        current = previous.get("litellm_params", {}).get("order")
        if isinstance(current, int) and not isinstance(current, bool) and current >= 1:
            order = current
    info = argon_sibling_info(existing)
    if previous:
        preserved = {
            key: value for key, value in previous.get("model_info", {}).items()
            if key in ARGON_COPY_FIELDS and value is not None
        }
        info = {**info, **preserved}
    info.update({
        "mode": "chat",
        "base_model": "gemini/" + ARGON_PUBLIC,
        "max_input_tokens": ARGON_INPUT_LIMIT,
        "max_output_tokens": ARGON_OUTPUT_LIMIT,
        "max_tokens": ARGON_OUTPUT_LIMIT,
        "cliproxy_argon_sync": ARGON_OWNER,
        "cliproxy_upstream_model": upstream,
    })
    if previous:
        info["id"] = previous["model_info"]["id"]
    base = os.environ.get("CLIPROXY_BASE", "http://192.168.68.3:8317")
    params = catalog.prompt_cache_drop({
        "model": "gemini/" + upstream,
        "custom_llm_provider": "gemini",
        "api_base": base.rstrip("/") + "/v1beta",
        "litellm_credential_name": ARGON_CREDENTIAL,
        "order": order,
    })
    if any(params.get(field) for field in ("api_key", "headers", "extra_headers")):
        raise ValueError("CLIProxyAPI route contains a static credential")
    body = {"model_name": ARGON_PUBLIC, "litellm_params": params, "model_info": info}
    if not previous:
        if len(existing) + 1 > ARGON_MAX:
            raise ValueError("registry ceiling would be exceeded")
        return "create", body
    same_route = all(
        previous.get("litellm_params", {}).get(key) == value
        for key, value in params.items()
    )
    same_info = all(
        previous.get("model_info", {}).get(key) == value
        for key, value in info.items()
    )
    return ("unchanged" if same_route and same_info else "update"), body


def sync_argon(client, home: Path, rows: list[dict], apply: bool = False) -> dict:
    base = os.environ.get("CLIPROXY_BASE", "http://192.168.68.3:8317")
    found = proxy_models(base)
    upstream = choose_argon(found)
    action, body = plan_argon(rows, upstream)
    report = {
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "catalog_models": len(found),
        "upstream_model": upstream,
        "action": action,
        "public_model": ARGON_PUBLIC,
        "other_credentials_preserved": True,
        "no_models_removed": True,
    }
    if body is not None:
        atomic(home / "cliproxy-argon-sync-preview.json", {"action": action, **body})
    if apply and action == "create":
        result = client.post("/model/new", body)
        if result.get("error_status") is not None:
            raise RuntimeError("gateway refused the CLIProxyAPI registration")
        identifier = (result.get("model_info") or {}).get("id") or result.get("model_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("registration response has no deployment id")
        atomic(home / "cliproxy-argon-sync-state.json",
               {"deployment_id": identifier, "upstream_model": upstream})
        report["deployment_id"] = identifier
    elif apply and action == "update":
        identifier = body["model_info"]["id"]
        result = client.patch(f"/model/{urllib.parse.quote(identifier, safe='')}/update", body)
        if result.get("error_status") is not None:
            raise RuntimeError("gateway refused the CLIProxyAPI update")
        atomic(home / "cliproxy-argon-sync-state.json",
               {"deployment_id": identifier, "upstream_model": upstream})
        report["deployment_id"] = identifier
    atomic(home / "cliproxy-argon-sync-report.json", report)
    return report


# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# LM Studio local inference catalog: the credential (api_base + api_key) lives
# in the LiteLLM database, so discovery runs inside the LiteLLM container and
# never prints the key.


LMSTUDIO_DISCOVERY = r"""
import os, json, hashlib, base64, urllib.error, urllib.request
import psycopg, nacl.secret
from urllib.parse import urlsplit, urlunsplit
cred = os.environ.get("LMSTUDIO_CREDENTIAL", "lmstudio")
u = urlsplit(os.environ["DATABASE_URL"])
with psycopg.connect(urlunsplit((u.scheme, u.netloc, u.path, "", "")), connect_timeout=5,
                     options="-c statement_timeout=5000 -c default_transaction_read_only=on") as conn:
    rows = conn.execute(
        'SELECT credential_values FROM "LiteLLM_CredentialsTable" WHERE credential_name=%s',
        (cred,),
    ).fetchall()
if len(rows) != 1:
    raise SystemExit(2)
salt = os.environ.get("LITELLM_SALT_KEY") or os.environ["LITELLM_MASTER_KEY"]
box = nacl.secret.SecretBox(hashlib.sha256(salt.encode()).digest())
def dec(s):
    if not isinstance(s, str):
        return s
    if s.startswith("v2:gcm:"):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        b = base64.urlsafe_b64decode(s[7:])
        return AESGCM(hashlib.sha256(salt.encode()).digest()).decrypt(b[:12], b[12:], None).decode()
    try:
        return box.decrypt(base64.urlsafe_b64decode(s)).decode()
    except Exception:
        return s
values = {k: dec(v) for k, v in rows[0][0].items()}
base = str(values.get("api_base") or "").rstrip("/")
if not base:
    raise SystemExit(3)
root = base[:-3] if base.endswith("/v1") else base
key = str(values.get("api_key") or "").strip()
headers = {"Accept": "application/json"}
if key:
    headers["Authorization"] = "Bearer " + key
data = None
last_err = None
for url in (root + "/api/v0/models", base + "/models"):
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            page = json.load(response)
        items = page.get("data") if isinstance(page, dict) else None
        if isinstance(items, list):
            data = items
            break
    except (OSError, urllib.error.URLError, ValueError) as exc:
        last_err = str(exc)
if data is None:
    raise SystemExit("lmstudio discovery failed: " + str(last_err))
print(json.dumps({"api_base": base, "models": data}))
"""


def lmstudio_catalog(credential: str = LMSTUDIO_CREDENTIAL) -> tuple[str, list[dict]]:
    """Query LM Studio through the gateway container using its stored credential."""
    try:
        completed = subprocess.run(
            [
                "docker", "exec", "-i",
                "-e", f"LMSTUDIO_CREDENTIAL={credential}",
                litellm_container(), "python", "-",
            ],
            input=LMSTUDIO_DISCOVERY, text=True, capture_output=True, timeout=60,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("LM Studio catalog discovery timed out") from error
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()[:160]
        raise RuntimeError(
            f"LM Studio catalog discovery failed with status {completed.returncode}"
            + (f": {detail}" if detail else "")
        )
    payload = json.loads(completed.stdout)
    api_base = payload.get("api_base") if isinstance(payload, dict) else None
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(api_base, str) or not api_base or not isinstance(models, list):
        raise ValueError("LM Studio catalog response is malformed")
    return api_base.rstrip("/"), models


def lmstudio_public_name(upstream_id: str, colliding_tails: set[str]) -> str:
    """Slashless public model name; disambiguate only when two models share a tail."""
    tail = upstream_id.rsplit("/", 1)[-1]
    if tail in colliding_tails:
        return upstream_id.replace("/", "-")
    return tail


def plan_lmstudio(models: list[dict], existing: list[dict], api_base: str,
                  prices: dict | None = None,
                  credential: str = LMSTUDIO_CREDENTIAL) -> list[tuple[str, dict]]:
    """Plan deployments for advertised models whose serving mode can be determined."""
    if not isinstance(existing, list) or len(existing) > LMSTUDIO_MAX:
        raise ValueError("invalid registry or registry exceeds its ceiling")
    if not isinstance(api_base, str) or not api_base.strip():
        raise ValueError("LM Studio api_base is required")
    api_base = api_base.rstrip("/")
    prices = prices or {}

    valid_models: list[dict] = []
    seen_ids: set[str] = set()
    tail_counts: dict[str, int] = {}
    for model in models:
        if not isinstance(model, dict):
            continue
        identifier = model.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("invalid LM Studio model id")
        identifier = identifier.strip()
        if identifier in seen_ids:
            continue
        seen_ids.add(identifier)
        tail = identifier.rsplit("/", 1)[-1]
        tail_counts[tail] = tail_counts.get(tail, 0) + 1
        valid_models.append({**model, "id": identifier})

    colliding_tails = {tail for tail, count in tail_counts.items() if count > 1}
    by_name: dict[str, list[dict]] = {}
    owned_by_id: dict[str, list[dict]] = {}
    for row in existing:
        by_name.setdefault(row["model_name"], []).append(row)
        info = row.get("model_info") or {}
        upstream_id = info.get("lmstudio_model_id")
        if info.get("lmstudio_sync") == LMSTUDIO_OWNER and isinstance(upstream_id, str):
            owned_by_id.setdefault(upstream_id, []).append(row)

    planned, creates = [], 0
    allocated: set[str] = set()
    for model in valid_models:
        upstream_id = model["id"]
        owned = owned_by_id.get(upstream_id, [])
        if len(owned) > 1:
            # Keep the registry intact; an operator must resolve ambiguous ownership.
            raise ValueError("ambiguous LM Studio route for " + upstream_id)
        previous = owned[0] if owned else None

        raw_type = str(model.get("type") or "").strip().lower()
        if raw_type in ("embedding", "embeddings"):
            mode = "embedding"
        elif raw_type in ("llm", "vlm"):
            mode = "chat"
        elif not raw_type and "embed" in upstream_id.lower():
            mode = "embedding"
        else:
            # The OpenAI-compatible listing can omit type. Leave unknown routes alone.
            continue
        is_embedding = mode == "embedding"

        if previous:
            public = previous["model_name"]
        else:
            preferred = lmstudio_public_name(upstream_id, colliding_tails)
            occupants = by_name.get(preferred, [])
            collision = preferred in allocated or any(
                (row.get("model_info") or {}).get("lmstudio_sync") == LMSTUDIO_OWNER
                for row in occupants
            )
            manual = [
                row["litellm_params"].get("model")
                for row in occupants
                if (row.get("litellm_params") or {}).get("litellm_credential_name") == credential
                and (row.get("model_info") or {}).get("lmstudio_sync") != LMSTUDIO_OWNER
            ]
            if any(
                not isinstance(route, str) or not route.startswith("lm_studio/")
                or route == "lm_studio/" + upstream_id
                for route in manual
            ):
                continue  # Never adopt or duplicate a manually configured route.
            collision = collision or bool(manual)
            public = preferred
            # The plain tail may deliberately join another provider's group.
            if collision or (occupants and preferred != upstream_id.rsplit("/", 1)[-1]):
                public = upstream_id.replace("/", "-")
                stem, suffix = public, 2
                while public in by_name or public in allocated:
                    public = f"{stem}-{suffix}"
                    suffix += 1
            allocated.add(public)
        same = by_name.get(public, [])

        orders = []
        for row in same:
            if row is previous:
                continue
            order = row.get("litellm_params", {}).get("order", 1)
            if isinstance(order, bool) or not isinstance(order, int):
                continue
            orders.append(order)
        order = previous["litellm_params"].get("order") if previous else (max(orders) if orders else 0) + 1
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            order = (max(orders) if orders else 0) + 1

        info = dict(previous["model_info"]) if previous else {}
        info.update({
            "mode": mode,
            "litellm_provider": "lm_studio",
            "lmstudio_sync": LMSTUDIO_OWNER,
            "lmstudio_model_id": upstream_id,
        })
        ctx = model.get("max_context_length")
        if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0:
            info["max_input_tokens"] = ctx
            if not is_embedding:
                max_out = min(max(ctx // 4, 4096), 32768)
                info["max_output_tokens"] = max_out
                info["max_tokens"] = max_out
        elif not is_embedding and "max_output_tokens" not in info:
            info["max_output_tokens"] = 8192
            info["max_tokens"] = 8192

        if raw_type == "vlm":
            info["supports_vision"] = True
        caps = model.get("capabilities")
        if isinstance(caps, list) and "tool_use" in caps:
            info["supports_function_calling"] = True

        params = catalog.prompt_cache_drop({
            "model": "lm_studio/" + upstream_id,
            "custom_llm_provider": "lm_studio",
            "api_base": api_base,
            "litellm_credential_name": credential,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "order": order,
        })
        if any(params.get(field) for field in ("api_key", "headers", "extra_headers")):
            raise ValueError("LM Studio route contains a static credential for " + upstream_id)

        body = {"model_name": public, "litellm_params": params, "model_info": info}
        if previous:
            same_fields = all(
                previous[section].get(key) == value
                for section in ("litellm_params", "model_info")
                for key, value in body[section].items()
            )
            action = "unchanged" if same_fields else "update"
        else:
            action = "create"
            creates += 1
        planned.append((action, body))

    if len(existing) + creates > LMSTUDIO_MAX:
        raise ValueError("registry ceiling would be exceeded")
    return planned


def sync_lmstudio(client, home: Path, rows: list[dict], prices: dict | None = None,
                  apply: bool = False,
                  credential: str = LMSTUDIO_CREDENTIAL) -> dict:
    """Discover models on LM Studio and register missing deployments in LiteLLM."""
    api_base, models = lmstudio_catalog(credential=credential)
    planned = plan_lmstudio(models, rows, api_base, prices=prices, credential=credential)
    report = {
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "api_base": api_base,
        "credential": credential,
        "catalog_models": len(models),
        "planned": {
            action: sum(item == action for item, _ in planned)
            for action in ("create", "update", "unchanged")
        },
        "models": [body["model_name"] for _, body in planned],
        "no_models_removed": True,
    }
    atomic(home / "lmstudio-sync-preview.json", [{"action": action, **body} for action, body in planned])
    if apply:
        created = updated = 0
        for action, body in planned:
            if action == "create":
                result = client.post("/model/new", body)
                if result.get("error_status") is not None:
                    raise RuntimeError(
                        "gateway refused LM Studio registration for "
                        + body["model_name"]
                        + ": status "
                        + str(result["error_status"])
                    )
                created += 1
            elif action == "update":
                identifier = body["model_info"]["id"]
                result = client.patch(f"/model/{urllib.parse.quote(identifier, safe='')}/update", body)
                if result.get("error_status") is not None:
                    raise RuntimeError(
                        "gateway refused LM Studio update for "
                        + body["model_name"]
                        + ": status "
                        + str(result["error_status"])
                    )
                updated += 1
        report["created"] = created
        report["updated"] = updated
    atomic(home / "lmstudio-sync-report.json", report)
    return report




# --------------------------------------------------------------------------
# Generic unused LiteLLM credentials: discover and register usable models for
# any credential that has no manual deployments and no dedicated syncer.

CREDENTIAL_OWNER = "credential-auto-sync-v1"
CREDENTIAL_MAX = 1024
CREDENTIAL_PER_CAP = 32
CREDENTIAL_RETRY_TTL_SECONDS = 86400
DEDICATED_CREDENTIALS = frozenset({
    "experientiallabs-shared",
    "cliproxyapi-shared",
    "cursor-chat-bridge",
    "cursor-api",
    "anthropic-api",
    "lmstudio",
})
NON_CHAT_CREDENTIAL_PROVIDERS = frozenset({"cursor", "cursor-chat-bridge"})
CHAT_ONLY_HOSTS = frozenset({"integrate.api.nvidia.com"})
LOCAL_ZERO_COST_PROVIDERS = frozenset({"lm_studio", "ollama"})

# Utility, guard, safety, audio/image/video-only, or non-chat models that
# should not be auto-registered as general-purpose chat/embedding deployments.
CREDENTIAL_SKIP_SUBSTRINGS = (
    "nemoguard",
    "safety-guard",
    "content-safety",
    "video-detector",
    "gliner",
    "nvclip",
    "moderation",
    "whisper",
    "transcribe",
    "tts",
    "realtime",
    "audio",
    "image",
    "lyria",
    "robotics",
    "computer-use",
    "deep-research",
    "nano-banana",
    "gpt-live",
    "calibration",
    "riva-translate",
    "embedqa",
    "nemoretriever",
    "embed-qa",
    "rerank",
    "starcoder2",
    "antigravity",
    "-search-api",
    "-healthcare",
    "chat-latest",
    "flash-latest",
    "flash-lite-latest",
    "pro-latest",
)
DATED_SNAPSHOT_RE = re.compile(
    r"-(?:\d{4}-\d{2}-\d{2}|\d{2}-\d{4}|1106|0125|0613|0301|20\d{6})$"
)

CREDENTIAL_DISCOVERY = r"""
import base64, hashlib, json, os, urllib.error, urllib.request
from urllib.parse import urlsplit, urlunsplit
import psycopg, nacl.secret

targets = json.loads(os.environ.get("TARGET_CREDENTIALS_JSON", "[]"))
if not targets:
    print("[]")
    raise SystemExit(0)

u = urlsplit(os.environ["DATABASE_URL"])
with psycopg.connect(
    urlunsplit((u.scheme, u.netloc, u.path, "", "")),
    connect_timeout=5,
    options="-c statement_timeout=5000 -c default_transaction_read_only=on",
) as conn:
    rows = conn.execute(
        'SELECT credential_name, credential_values, credential_info '
        'FROM "LiteLLM_CredentialsTable" '
        'WHERE credential_name = ANY(%s) ORDER BY credential_name',
        (targets,),
    ).fetchall()

salt = os.environ.get("LITELLM_SALT_KEY") or os.environ["LITELLM_MASTER_KEY"]
box = nacl.secret.SecretBox(hashlib.sha256(salt.encode()).digest())

def dec(s):
    if not isinstance(s, str):
        return s
    if s.startswith("v2:gcm:"):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        b = base64.urlsafe_b64decode(s[7:])
        return AESGCM(hashlib.sha256(salt.encode()).digest()).decrypt(b[:12], b[12:], None).decode()
    try:
        return box.decrypt(base64.urlsafe_b64decode(s)).decode()
    except Exception:
        return s

results = []
for name, raw_vals, info in rows:
    vals = {k: dec(v) for k, v in (raw_vals or {}).items()}
    info = info or {}
    provider = str(vals.get("custom_llm_provider") or info.get("custom_llm_provider") or "").strip()
    p_low = provider.lower()
    api_base = str(vals.get("api_base") or "").strip().rstrip("/")
    api_key = str(vals.get("api_key") or "").strip()
    headers = {"Accept": "application/json", "User-Agent": "auto-router-pool/1.0"}
    record = {"credential": name, "provider": provider, "api_base": api_base, "models": []}

    try:
        if p_low in ("google_ai_studio", "gemini"):
            base = api_base or "https://generativelanguage.googleapis.com/v1beta"
            record["api_base"] = base
            if api_key:
                headers["x-goog-api-key"] = api_key
            req = urllib.request.Request(base + "/models?pageSize=200", headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = json.load(resp)
            models = []
            for item in body.get("models") or []:
                if not isinstance(item, dict):
                    continue
                mid = str(item.get("name") or "").removeprefix("models/").strip()
                if not mid:
                    continue
                methods = item.get("supportedGenerationMethods") or []
                if "generateContent" in methods:
                    mode = "chat"
                elif "embedContent" in methods:
                    mode = "embedding"
                else:
                    continue
                entry = {"id": mid, "mode": mode}
                if isinstance(item.get("inputTokenLimit"), int) and item["inputTokenLimit"] > 0:
                    entry["max_input_tokens"] = item["inputTokenLimit"]
                if isinstance(item.get("outputTokenLimit"), int) and item["outputTokenLimit"] > 0:
                    entry["max_output_tokens"] = item["outputTokenLimit"]
                models.append(entry)
            record["models"] = models
        elif p_low == "anthropic":
            base = api_base or "https://api.anthropic.com"
            record["api_base"] = base
            headers["anthropic-version"] = "2023-06-01"
            if api_key:
                headers["x-api-key"] = api_key
            req = urllib.request.Request(base + "/v1/models?limit=100", headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = json.load(resp)
            models = []
            for item in body.get("data") or []:
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                if item.get("lifecycle") and item.get("lifecycle") != "active":
                    continue
                entry = {"id": str(item["id"]).strip(), "mode": "chat"}
                if isinstance(item.get("max_input_tokens"), int) and item["max_input_tokens"] > 0:
                    entry["max_input_tokens"] = item["max_input_tokens"]
                if isinstance(item.get("max_tokens"), int) and item["max_tokens"] > 0:
                    entry["max_output_tokens"] = item["max_tokens"]
                models.append(entry)
            record["models"] = models
        elif p_low in ("nvidia_nim", "openrouter", "openai", "lm_studio", "ollama", "hosted_vllm", "vllm"):
            if p_low == "nvidia_nim":
                base = api_base or "https://integrate.api.nvidia.com/v1"
            elif p_low == "openrouter":
                base = api_base or "https://openrouter.ai/api/v1"
            elif p_low == "openai":
                base = api_base or "https://api.openai.com/v1"
            else:
                base = api_base
            if not base:
                record["error"] = "missing_api_base"
                results.append(record)
                continue
            record["api_base"] = base
            if api_key:
                headers["Authorization"] = "Bearer " + api_key
            if p_low == "lm_studio":
                root = base[:-3] if base.endswith("/v1") else base
                urls = [root + "/api/v0/models", base + "/models"]
            elif base.endswith("/v1"):
                urls = [base + "/models"]
            else:
                urls = [base + "/v1/models", base + "/models"]
            data = None
            last_err = None
            for url in urls:
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=20) as resp:
                        page = json.load(resp)
                    items = page.get("data") if isinstance(page, dict) else None
                    if isinstance(items, list):
                        data = items
                        break
                except urllib.error.HTTPError as exc:
                    body_txt = ""
                    try:
                        body_txt = exc.read().decode("utf-8", "ignore")[:180]
                    except Exception:
                        pass
                    last_err = f"http_{exc.code}: {body_txt}"
                except Exception as exc:
                    last_err = f"{type(exc).__name__}: {str(exc)[:160]}"
            if data is None:
                record["error"] = last_err or "discovery_failed"
            else:
                models = []
                for item in data:
                    if not isinstance(item, dict) or not item.get("id"):
                        continue
                    mid = str(item["id"]).strip()
                    raw_type = str(item.get("type") or "").lower()
                    if raw_type in ("embeddings", "embedding") or "embed" in mid.lower():
                        mode = "embedding"
                    else:
                        mode = "chat"
                    entry = {"id": mid, "mode": mode}
                    ctx = item.get("max_context_length") or item.get("context_length")
                    if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0:
                        entry["max_input_tokens"] = ctx
                    if isinstance(item.get("pricing"), dict):
                        entry["pricing"] = item["pricing"]
                    if isinstance(item.get("capabilities"), list):
                        entry["capabilities"] = item["capabilities"]
                    models.append(entry)
                record["models"] = models
        else:
            record["error"] = f"unsupported_provider:{provider}"
    except urllib.error.HTTPError as exc:
        body_txt = ""
        try:
            body_txt = exc.read().decode("utf-8", "ignore")[:180]
        except Exception:
            pass
        record["error"] = f"http_{exc.code}: {body_txt}"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    results.append(record)

print(json.dumps(results))
"""


def unused_credential_names(
    credentials: list[dict],
    rows: list[dict],
    skip: set[str] | frozenset[str] = DEDICATED_CREDENTIALS,
) -> list[str]:
    """Return credentials with no manual deployments and no dedicated syncer.

    A credential whose only deployments are already owned by ``CREDENTIAL_OWNER``
    stays eligible so subsequent hourly syncs remain idempotent and pick up new
    models. Credentials with any non-owned deployment (including manually
    blocked routes) are never touched, so operator blocks are always respected.
    """
    manual_creds: set[str] = set()
    for row in rows:
        params = row.get("litellm_params") or {}
        info = row.get("model_info") or {}
        cname = params.get("litellm_credential_name")
        if not isinstance(cname, str) or not cname:
            continue
        if info.get("credential_auto_sync") == CREDENTIAL_OWNER:
            continue
        manual_creds.add(cname)

    eligible: list[str] = []
    for cred in credentials:
        if not isinstance(cred, dict):
            continue
        name = str(cred.get("credential_name") or "").strip()
        if not name or name in skip or name in manual_creds:
            continue
        vals = cred.get("credential_values") or {}
        info = cred.get("credential_info") or {}
        provider = str(
            vals.get("custom_llm_provider") or info.get("custom_llm_provider") or ""
        ).strip().lower()
        if provider in NON_CHAT_CREDENTIAL_PROVIDERS:
            continue
        eligible.append(name)
    return sorted(set(eligible))


def discover_credentials(target_names: list[str]) -> list[dict]:
    """Discover catalogs for target credentials inside the LiteLLM container."""
    if not target_names:
        return []
    completed = subprocess.run(
        [
            "docker", "exec", "-i",
            "-e", f"TARGET_CREDENTIALS_JSON={json.dumps(target_names)}",
            litellm_container(), "python", "-",
        ],
        input=CREDENTIAL_DISCOVERY,
        text=True,
        capture_output=True,
        timeout=90,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()[:160]
        raise RuntimeError(
            f"credential catalog discovery failed with status {completed.returncode}"
            + (f": {detail}" if detail else "")
        )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise ValueError("credential catalog discovery response is malformed")
    return payload


def credential_transport(provider: str, api_base: str = "") -> tuple[str, str, str | None]:
    """Return ``(model_prefix, custom_llm_provider, effective_api_base)``."""
    p_low = (provider or "").strip().lower()
    base = (api_base or "").strip().rstrip("/")
    host = (urllib.parse.urlparse(base).hostname or "").casefold()
    if p_low == "nvidia_nim" or host in CHAT_ONLY_HOSTS:
        return "hosted_vllm", "hosted_vllm", base or "https://integrate.api.nvidia.com/v1"
    if p_low in ("google_ai_studio", "gemini"):
        return "gemini", "gemini", base or None
    if p_low == "openrouter":
        return "openrouter", "openrouter", None
    if p_low == "anthropic":
        return "anthropic", "anthropic", base or ANTHROPIC_API
    if p_low == "lm_studio":
        return "lm_studio", "lm_studio", base or None
    if p_low == "ollama":
        return "ollama", "ollama", base or None
    if p_low in ("hosted_vllm", "vllm"):
        return "hosted_vllm", "hosted_vllm", base or None
    return "openai", "openai", base or None


def credential_model_eligible(
    upstream_id: str,
    mode: str,
    retired: set[str],
    all_ids: set[str],
    provider: str = "",
) -> bool:
    """Filter out retired, utility/guard/media-only, and redundant dated models."""
    low = upstream_id.strip().lower()
    if not low or low.endswith(":batch"):
        return False
    tail = low.rsplit("/", 1)[-1].split(":", 1)[0]
    if upstream_id in retired or tail in retired:
        return False
    if any(token in low for token in CREDENTIAL_SKIP_SUBSTRINGS):
        return False
    if provider.strip().lower() == "openrouter" and ":free" not in low:
        return False
    if mode not in ("chat", "embedding"):
        return False
    # Skip dated snapshots when the undated alias is advertised alongside them.
    match = DATED_SNAPSHOT_RE.search(low)
    if match:
        base_id = low[: match.start()]
        if base_id in {item.lower() for item in all_ids}:
            return False
    return True


def _finite_cost(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num) or num < 0:
        return None
    return num


def _lookup_credential_price(
    upstream_id: str,
    public: str,
    prefix: str,
    provider: str,
    catalog_pricing: dict | None,
    prices: dict,
) -> tuple[str | None, dict[str, float], dict]:
    """Return ``(base_model, cost_fields, extra_limits)`` for a discovered model."""
    p_low = (provider or "").strip().lower()
    if p_low in LOCAL_ZERO_COST_PROVIDERS or upstream_id.lower().endswith(":free"):
        return None, {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}, {}

    if isinstance(catalog_pricing, dict):
        inp = _finite_cost(catalog_pricing.get("prompt"))
        out = _finite_cost(catalog_pricing.get("completion"))
        if inp is not None and out is not None:
            return None, {"input_cost_per_token": inp, "output_cost_per_token": out}, {}

    tail = upstream_id.rsplit("/", 1)[-1].split(":", 1)[0]
    for candidate in (
        f"{prefix}/{upstream_id}",
        f"{p_low}/{upstream_id}" if p_low else "",
        f"openrouter/{upstream_id}",
        upstream_id,
        public,
        tail,
    ):
        if not candidate:
            continue
        entry = prices.get(candidate)
        if not isinstance(entry, dict):
            continue
        costs: dict[str, float] = {}
        for field in ("input_cost_per_token", "output_cost_per_token"):
            val = _finite_cost(entry.get(field))
            if val is not None:
                costs[field] = val
        limits: dict[str, int] = {}
        for field in ("max_input_tokens", "max_output_tokens"):
            val = entry.get(field)
            if isinstance(val, int) and not isinstance(val, bool) and val > 0:
                limits[field] = val
        if costs:
            return candidate, costs, limits
    return None, {}, {}


def plan_credential_models(
    credential: str,
    provider: str,
    api_base: str,
    models: list[dict],
    existing: list[dict],
    prices: dict | None = None,
    retired: set[str] | None = None,
    skip_ids: set[str] | None = None,
    max_per_credential: int = CREDENTIAL_PER_CAP,
) -> list[tuple[str, dict]]:
    """Plan LiteLLM deployments for eligible models discovered on a credential."""
    if not isinstance(existing, list) or len(existing) > CREDENTIAL_MAX:
        raise ValueError("invalid registry or registry exceeds its ceiling")
    prices = prices or {}
    retired = retired or set()
    skip_ids = skip_ids or set()
    prefix, custom_provider, effective_base = credential_transport(provider, api_base)

    all_ids = {
        str(m.get("id") or "").strip()
        for m in models
        if isinstance(m, dict) and isinstance(m.get("id"), str) and m.get("id").strip()
    }

    candidates: list[dict] = []
    seen_ids: set[str] = set()
    tail_counts: dict[str, int] = {}
    for model in models:
        if not isinstance(model, dict):
            continue
        uid = str(model.get("id") or "").strip()
        if not uid or uid in seen_ids or uid in skip_ids:
            continue
        mode = str(model.get("mode") or "chat").lower()
        if not credential_model_eligible(uid, mode, retired, all_ids, provider=provider):
            continue
        seen_ids.add(uid)
        tail = uid.rsplit("/", 1)[-1].split(":", 1)[0]
        tail_counts[tail] = tail_counts.get(tail, 0) + 1
        candidates.append({**model, "id": uid, "mode": mode})

    # Prefer chat models ahead of embeddings, sorted deterministically, bounded by cap.
    candidates.sort(key=lambda item: (0 if item["mode"] == "chat" else 1, item["id"]))
    candidates = candidates[:max_per_credential]
    colliding_tails = {t for t, count in tail_counts.items() if count > 1}

    # Never re-introduce an upstream model on the same host when every existing
    # deployment for that exact (host, upstream_id) is already blocked (for
    # example, a provider-side broken streaming model manually blocked on a
    # sibling key on integrate.api.nvidia.com).
    eff_host = (urllib.parse.urlparse(effective_base or "").hostname or "").casefold()
    host_blocked: set[tuple[str, str]] = set()
    host_live: set[tuple[str, str]] = set()
    for row in existing:
        rp = row.get("litellm_params") or {}
        ri = row.get("model_info") or {}
        if rp.get("litellm_credential_name") == credential:
            continue
        rhost = (urllib.parse.urlparse(str(rp.get("api_base") or "")).hostname or "").casefold()
        rmodel = str(rp.get("model") or "").strip()
        ruid = rmodel.split("/", 1)[-1] if "/" in rmodel else rmodel
        if not rhost or not ruid:
            continue
        key = (rhost, ruid.casefold())
        if ri.get("blocked"):
            host_blocked.add(key)
        else:
            host_live.add(key)
    blocked_only_on_host = host_blocked - host_live

    planned: list[tuple[str, dict]] = []
    creates = 0
    for model in candidates:
        uid = model["id"]
        mode = model["mode"]
        tail = uid.rsplit("/", 1)[-1].split(":", 1)[0]
        public = uid.split(":", 1)[0].replace("/", "-") if tail in colliding_tails else tail

        same = [row for row in existing if row.get("model_name") == public]
        owned = [
            row for row in same
            if row.get("litellm_params", {}).get("litellm_credential_name") == credential
            and row.get("model_info", {}).get("credential_auto_sync") == CREDENTIAL_OWNER
        ]
        manual = [
            row for row in same
            if row.get("litellm_params", {}).get("litellm_credential_name") == credential
            and row not in owned
        ]
        if manual and not owned:
            continue
        if not owned and eff_host and (eff_host, uid.casefold()) in blocked_only_on_host:
            continue
        if len(owned) > 1:
            raise ValueError(f"ambiguous auto-synced route for {credential}/{uid}")
        previous = owned[0] if owned else None
        if previous and previous["model_info"].get("credential_auto_sync_model_id") != uid:
            raise ValueError(f"owned route identity mismatch for {credential}/{uid}")

        orders = []
        for row in same:
            if row is previous:
                continue
            raw_order = row.get("litellm_params", {}).get("order")
            order = 1 if raw_order is None else raw_order
            if isinstance(order, bool) or not isinstance(order, int):
                raise ValueError("invalid routing order for " + public)
            orders.append(order)
        prev_order = previous["litellm_params"].get("order") if previous else None
        order = prev_order if isinstance(prev_order, int) and not isinstance(prev_order, bool) else (
            (max(orders) if orders else 0) + 1
        )

        base_model, costs, price_limits = _lookup_credential_price(
            uid, public, prefix, provider, model.get("pricing"), prices,
        )
        info = dict(previous["model_info"]) if previous else {}
        info.update({
            "mode": mode,
            "credential_auto_sync": CREDENTIAL_OWNER,
            "credential_auto_sync_source": credential,
            "credential_auto_sync_model_id": uid,
        })
        if base_model and not info.get("base_model"):
            info["base_model"] = base_model
        for field in ("max_input_tokens", "max_output_tokens"):
            val = model.get(field) or price_limits.get(field)
            if isinstance(val, int) and not isinstance(val, bool) and val > 0:
                info[field] = val
        caps = model.get("capabilities") or []
        if isinstance(caps, list) and "tool_use" in caps:
            info["supports_function_calling"] = True

        params_raw = {
            "model": f"{prefix}/{uid}",
            "custom_llm_provider": custom_provider,
            "litellm_credential_name": credential,
            "order": order,
            **costs,
        }
        if effective_base and custom_provider not in ("openrouter", "gemini"):
            params_raw["api_base"] = effective_base
        params = catalog.prompt_cache_drop(params_raw)
        if any(params.get(field) for field in ("api_key", "headers", "extra_headers")):
            raise ValueError("credential route contains a static credential for " + uid)

        body = {"model_name": public, "litellm_params": params, "model_info": info}
        if previous:
            same_fields = all(
                previous[section].get(key) == value
                for section in ("litellm_params", "model_info")
                for key, value in body[section].items()
            )
            action = "unchanged" if same_fields else "update"
        else:
            action = "create"
            creates += 1
        planned.append((action, body))

    if len(existing) + creates > CREDENTIAL_MAX:
        raise ValueError("registry ceiling would be exceeded")
    return planned


def _probe_new_deployment(client, model_id: str, probe_fn=None) -> str:
    """Verify a newly registered deployment answers; retry once on replica lag."""
    fn = probe_fn or client_mod.deployment_health
    status = fn(client, model_id)
    if status == "unknown" and probe_fn is None:
        time.sleep(1.5)
        status = fn(client, model_id)
    return status


def sync_unused_credentials(
    client,
    home: Path,
    rows: list[dict],
    prices: dict | None = None,
    apply: bool = False,
    probe=None,
    discover=None,
    now: float | None = None,
) -> dict:
    """Discover and register usable models for unused LiteLLM credentials."""
    prices = prices or {}
    now_ts = time.time() if now is None else float(now)
    creds_resp = client.get("/credentials")
    if creds_resp.get("error_status") is not None:
        raise RuntimeError(f"gateway refused /credentials: status {creds_resp['error_status']}")
    credentials = creds_resp.get("credentials") or []
    targets = unused_credential_names(credentials, rows)

    state_path = home / "unused-credentials-sync-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    failed_models: dict[str, dict] = dict(state.get("failed_models") or {})
    failed_creds: dict[str, dict] = dict(state.get("failed_credentials") or {})

    # Prune expired cooldown entries.
    failed_models = {
        k: v for k, v in failed_models.items()
        if isinstance(v, dict) and now_ts - float(v.get("ts") or 0) < CREDENTIAL_RETRY_TTL_SECONDS
    }
    failed_creds = {
        k: v for k, v in failed_creds.items()
        if isinstance(v, dict) and now_ts - float(v.get("ts") or 0) < CREDENTIAL_RETRY_TTL_SECONDS
    }

    active_targets = [c for c in targets if c not in failed_creds]
    discovered = (discover or discover_credentials)(active_targets) if active_targets else []
    retired = retired_names(home)

    all_preview: list[dict] = []
    per_credential: dict[str, dict] = {}
    working_rows = list(rows)
    total_created = total_updated = total_reverted = 0

    for item in discovered:
        cred_name = str(item.get("credential") or "")
        provider = str(item.get("provider") or "")
        api_base = str(item.get("api_base") or "")
        if item.get("error"):
            per_credential[cred_name] = {"provider": provider, "error": item["error"]}
            continue
        models = item.get("models") or []
        skip_ids = {
            k.split("::", 1)[1]
            for k in failed_models
            if k.startswith(cred_name + "::")
        }
        try:
            planned = plan_credential_models(
                cred_name, provider, api_base, models, working_rows,
                prices=prices, retired=retired, skip_ids=skip_ids,
            )
        except ValueError as exc:
            per_credential[cred_name] = {"provider": provider, "error": str(exc)}
            continue

        all_preview.extend(
            {"credential": cred_name, "action": action, **body}
            for action, body in planned
        )
        summary = {
            "provider": provider,
            "api_base": api_base,
            "catalog_models": len(models),
            "skipped_cooldown": len(skip_ids),
            "planned": {
                action: sum(a == action for a, _ in planned)
                for action in ("create", "update", "unchanged")
            },
            "models": [body["model_name"] for _, body in planned],
        }

        if apply:
            created = updated = reverted = 0
            kept_models: list[str] = []
            reverted_models: list[dict] = []
            for action, body in planned:
                uid = body["model_info"]["credential_auto_sync_model_id"]
                mode = body["model_info"].get("mode", "chat")
                if action == "create":
                    res = client.post("/model/new", body)
                    if res.get("error_status") is not None:
                        reverted += 1
                        reverted_models.append({
                            "model": body["model_name"], "id": uid,
                            "reason": f"create_{res['error_status']}",
                        })
                        continue
                    model_id = res.get("model_id") or (res.get("model_info") or {}).get("id")
                    if mode == "chat" and model_id:
                        health = _probe_new_deployment(client, model_id, probe_fn=probe)
                        if health == "budget_exceeded":
                            client.post("/model/delete", {"id": model_id})
                            reverted += 1
                            reverted_models.append({
                                "model": body["model_name"], "id": uid, "reason": "budget_exceeded",
                            })
                            failed_creds[cred_name] = {"ts": now_ts, "reason": "budget_exceeded"}
                            break
                        if health not in ("ok", "rate_limited"):
                            client.post("/model/delete", {"id": model_id})
                            reverted += 1
                            reverted_models.append({
                                "model": body["model_name"], "id": uid, "reason": health,
                            })
                            failed_models[f"{cred_name}::{uid}"] = {"ts": now_ts, "reason": health}
                            continue
                    created += 1
                    kept_models.append(body["model_name"])
                    working_rows.append({
                        "model_name": body["model_name"],
                        "litellm_params": dict(body["litellm_params"]),
                        "model_info": {**body["model_info"], "id": model_id},
                    })
                elif action == "update":
                    identifier = body["model_info"]["id"]
                    res = client.patch(
                        f"/model/{urllib.parse.quote(identifier, safe='')}/update", body
                    )
                    if res.get("error_status") is not None:
                        raise RuntimeError(
                            f"gateway refused credential update for {body['model_name']}: "
                            f"status {res['error_status']}"
                        )
                    updated += 1
            summary.update({
                "created": created,
                "updated": updated,
                "reverted": reverted,
                "kept_models": kept_models,
                "reverted_models": reverted_models,
            })
            total_created += created
            total_updated += updated
            total_reverted += reverted

        per_credential[cred_name] = summary

    report = {
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "eligible_credentials": targets,
        "cooldown_credentials": sorted(c for c in targets if c in failed_creds),
        "credentials": per_credential,
        "no_models_removed": True,
    }
    if apply:
        report["created"] = total_created
        report["updated"] = total_updated
        report["reverted"] = total_reverted
        atomic(state_path, {
            "updated_at": report["checked_at"],
            "failed_models": failed_models,
            "failed_credentials": failed_creds,
        })
    atomic(home / "unused-credentials-sync-preview.json", all_preview)
    atomic(home / "unused-credentials-sync-report.json", report)
    return report


def sync_all(client, home: Path, rows: list[dict], prices: dict | None = None,
             apply: bool = False) -> dict:
    """Run every provider catalogue, keeping one failure from hiding the rest."""
    prices = prices or {}
    output = {}
    for name, call in (
        ("cursor_cloud", lambda: sync_cursor_cloud(client, home)),
        ("cursor_chat", lambda: sync_cursor_chat(client, home, rows, apply)),
        ("anthropic", lambda: sync_anthropic(client, home, rows, prices, apply)),
        ("argon", lambda: sync_argon(client, home, rows, apply)),
        ("lmstudio", lambda: sync_lmstudio(client, home, rows, prices, apply)),
        ("unused_credentials", lambda: sync_unused_credentials(client, home, rows, prices, apply)),
    ):
        try:
            output[name] = call()
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            output[name] = {"error": f"{type(error).__name__}: {str(error)[:160]}"}
    return output
