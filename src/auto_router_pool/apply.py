"""Diff, validate, patch, verify and roll back router configuration."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from .tiers import build

CLASSIFIER_TIMEOUT_MS = 30_000


def context_cap(rows: list[dict], tiers: dict) -> int | None:
    """Smallest declared input window among the models a router can select.

    A router must never advertise more context than the weakest solver a tier can
    hand a request to.  An undeclared window is unknown, so no safe floor can be
    proven: return ``None`` instead of guessing a larger number.
    """
    capacities: dict[str, int] = {}
    interactive: set[str] = set()
    for row in rows:
        info = row.get("model_info") or {}
        name = row["model_name"]
        # An image, video or embedding deployment never receives the routed chat
        # request, so its window cannot bound what the router advertises.
        if info.get("mode") not in (None, "chat", "completion"):
            continue
        interactive.add(name)
        value = info.get("max_input_tokens")
        if type(value) is int and value > 0:
            capacities[name] = min(capacities.get(name, value), value)
    names = {
        model for models in tiers.values() for model in models
        if model and model in interactive
    }
    if not names or any(name not in capacities for name in names):
        return None
    return min(capacities[name] for name in names)


def context_drift(rows: list[dict], names=()) -> list[str]:
    """Routers advertising more input than their own configured tiers allow.

    ``--watch`` normally exits on an unchanged catalogue; this keeps a context
    ceiling that drifted from its solvers visible without waiting for one.
    """
    watched = set(names)
    drift = []
    for row in rows:
        if watched and row["model_name"] not in watched:
            continue
        config = (row.get("litellm_params") or {}).get("complexity_router_config") or {}
        tiers = config.get("tiers")
        if not isinstance(tiers, dict) or not tiers:
            continue
        cap = context_cap(rows, tiers)
        if cap is not None and (row.get("model_info") or {}).get("max_input_tokens") != cap:
            drift.append(row["model_name"])
    return drift


def classifier_drift(rows: list[dict], names=()) -> list[str]:
    """Routers whose classifier is foreign to their own tiers or too impatient.

    A classifier pointing at a model the router no longer selects times out and
    silently disables tiering, so it counts as drift even when the tiers match.
    """
    watched = set(names)
    drift = []
    for row in rows:
        if watched and row["model_name"] not in watched:
            continue
        config = (row.get("litellm_params") or {}).get("complexity_router_config") or {}
        tiers = config.get("tiers")
        if not isinstance(tiers, dict) or not tiers:
            continue
        models = {model for entries in tiers.values() for model in entries if model}
        classifier = config.get("classifier_llm_config") or {}
        if (
            classifier.get("model") not in models
            or (classifier.get("timeout_ms") or 0) < CLASSIFIER_TIMEOUT_MS
        ):
            drift.append(row["model_name"])
    return drift


def plan(rows: list[dict], routers: dict, default: str, classifier: str | None = None,
         extras: dict | None = None):
    """Return the routers whose tiers, contract, context floor or classifier differ.

    ``extras`` carries the classifier contract (tier definitions, examples,
    prompt, fallback) per router. It must be written in the same patch as the
    tiers: LiteLLM refuses a tier map that does not cover every defined tier, so
    writing one without the other is rejected at load time.
    """
    by_name = {row["model_name"]: row for row in rows}
    changes = []
    for name, tier_map in routers.items():
        row = by_name.get(name)
        if not row:
            continue
        tiers, configs = build(tier_map)
        config = copy.deepcopy(row["litellm_params"]["complexity_router_config"])
        contract = (extras or {}).get(name) or {}
        if contract.get("tier_definitions"):
            # The engine refuses custom tiers together with the built-in severity
            # order: affinity ordering and the default-model fallback both assume
            # it. Custom tiers therefore drop affinity and fall back explicitly.
            config["session_affinity"] = False
            config.pop("classifier_fallback", None)
        contract_changed = any(config.get(key) != value for key, value in contract.items())
        params_changed = (
            config.get("tiers") != tiers
            or config.get("tier_model_configs") != configs
            or contract_changed
        )
        classifier_changed = False
        if classifier:
            current = config.get("classifier_llm_config") or {}
            classifier_changed = (
                current.get("model") != classifier
                or (current.get("timeout_ms") or 0) < CLASSIFIER_TIMEOUT_MS
            )
        cap = context_cap(rows, tiers)
        cap_changed = (
            cap is not None
            and (row.get("model_info") or {}).get("max_input_tokens") != cap
        )
        if not params_changed and not cap_changed and not classifier_changed:
            continue
        change = {
            "model_name": name,
            "id": row["model_info"]["id"],
            "previous": copy.deepcopy(row["litellm_params"]),
            "previous_max_input_tokens": (row.get("model_info") or {}).get(
                "max_input_tokens"
            ),
        }
        if classifier:
            change["classifier"] = classifier
        if params_changed or classifier_changed:
            config["tiers"] = tiers
            config["tier_model_configs"] = configs
            config["default_model"] = default
            config.update(contract)
            if classifier:
                current = dict(config.get("classifier_llm_config") or {})
                current["model"] = classifier
                current["timeout_ms"] = max(
                    int(current.get("timeout_ms") or 0), CLASSIFIER_TIMEOUT_MS
                )
                config["classifier_llm_config"] = current
            change["config"] = config
            change["litellm_params"] = {
                **row["litellm_params"],
                "complexity_router_config": config,
                "complexity_router_default_model": default,
            }
        if cap_changed:
            change["model_info"] = {"max_input_tokens": cap}
        changes.append(change)
    return changes


def validate(client, changes: list[dict]):
    for change in changes:
        if "config" not in change:
            continue
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
                 "config": c.get("config"),
                 "model_info": c.get("model_info"),
                 "previous_max_input_tokens": c.get("previous_max_input_tokens")}
                for c in changes
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return path


def _patch_body(change: dict, *, rollback: bool = False) -> dict:
    body = {}
    if "litellm_params" in change:
        body["litellm_params"] = change["previous"] if rollback else change["litellm_params"]
    if "model_info" in change:
        body["model_info"] = (
            {"max_input_tokens": change.get("previous_max_input_tokens")}
            if rollback
            else change["model_info"]
        )
    return body


def apply(client, changes: list[dict]):
    """Patch every changed router, then re-read and verify; roll back on drift."""
    for change in changes:
        client.patch(f"/model/{change['id']}/update", _patch_body(change))

    after = {row["model_name"]: row for row in client.get("/model/info").get("data", [])}
    for change in changes:
        got = after.get(change["model_name"], {})
        if "config" in change:
            actual = got.get("litellm_params", {}).get("complexity_router_config", {})
            classifier = change.get("classifier")
            actual_classifier = actual.get("classifier_llm_config") or {}
            contract_drift = any(
                actual.get(key) != change["config"][key]
                for key in ("tier_definitions", "classification_examples",
                            "classification_prompt", "fallback_tier",
                            "classification_mode", "session_affinity",
                            "classifier_fallback")
                if key in change["config"]
            )
            if contract_drift or actual.get("tiers") != change["config"]["tiers"] or (
                classifier
                and (
                    actual_classifier.get("model") != classifier
                    or (actual_classifier.get("timeout_ms") or 0)
                    < CLASSIFIER_TIMEOUT_MS
                )
            ):
                for old in changes:
                    client.patch(f"/model/{old['id']}/update", _patch_body(old, rollback=True))
                raise RuntimeError("verification failed; rolled back")
        if "model_info" in change:
            if (got.get("model_info") or {}).get("max_input_tokens") != change["model_info"][
                "max_input_tokens"
            ]:
                for old in changes:
                    client.patch(f"/model/{old['id']}/update", _patch_body(old, rollback=True))
                raise RuntimeError("verification failed; rolled back")


def set_auto_fallback(client, chain: list[str]):
    """Replace only the ``auto`` edge, preserving every other fallback."""
    result = client.get("/router/settings")
    if result.get("error_status") is not None:
        # A failed read must not look like an empty fallback list: the following
        # /config/update would replace every existing edge with only ``auto``.
        raise RuntimeError(
            f"router settings unavailable ({result['error_status']}); fallbacks unchanged"
        )
    settings = result.get("current_values", {})
    edges: dict[str, list[str]] = {}
    for entry in settings.get("fallbacks") or []:
        source, targets = next(iter(entry.items()))
        edges[source] = list(targets)
    # One model can win several roles at once (premium, primary, budget). A
    # repeated hop only retries the same deployment, so keep the first mention.
    edges["auto"] = list(dict.fromkeys(name for name in chain if name))
    fallbacks = [{src: targets} for src, targets in sorted(edges.items()) if targets]
    result = client.post("/config/update", {"router_settings": {"fallbacks": fallbacks}})
    if result.get("error_status") is not None:
        raise RuntimeError(f"fallback update failed ({result['error_status']})")


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
