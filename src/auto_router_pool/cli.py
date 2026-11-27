"""Command line entry point.

Nothing environment-specific is baked in: ports, model family, micro models and
the state directory all come from flags or environment variables, so this file
is safe to publish and reusable against any OpenAI-compatible catalogue. A
separate catalogue proxy (CLIProxyAPI) is optional: without one, candidate ids
come from LiteLLM's own catalogue.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import aa
from .assign import SPREAD, place, roles, schedule
from .apply import apply as apply_changes
from .apply import (
    backup,
    classifier_drift,
    context_drift,
    plan,
    set_auto_fallback,
    smoke,
    validate,
)
from .client import (
    Client,
    deployment_health,
    models,
    probe,
    probe_decision_choice,
    probe_with_usage,
)
from .apply import DECISION_CLASSIFIER
from .catalog import sync as sync_catalog
from .catalog import sync_batch
from .extra_catalogs import sync_all as sync_provider_catalogs
from .maintenance import (
    context_sync_pass,
    effort_suffix_pass,
    health_report,
    prompt_cache_drop_pass,
    provider_prefix_pass,
    repair_cooldowns,
)
from .modality_benchmarks import load as load_modality_benchmarks
from .pool import (
    MIN_INPUT_TOKENS,
    changed_scores,
    fingerprint,
    flaky_deployments,
    group_state,
    input_window,
    pick_default,
    pick_micro,
    score_hashes,
    stable_deployments,
    batch_models,
    classifier_candidates,
    decision_candidates,
    decision_catalog,
    decision_models,
    specialized_models,
    free_gaps,
    free_models,
)
from .quarantine import budget_sweep
from .quarantine import sweep as quarantine_sweep
from .registry import ensure_guests, load_guests, registered
from .tiers import FRONTIER, ROUTERS, policies, routers, top_effort


def default_home() -> Path:
    override = os.environ.get("AUTO_POOL_HOME")
    if override:
        return Path(override)
    return Path.home() / ".local" / "state" / "auto-router-pool"


@dataclass
class Settings:
    litellm_base: str = "http://127.0.0.1:4000"
    cliproxy_base: str = "http://127.0.0.1:8317"
    aa_base: str = aa.DEFAULT_BASE
    aa_ttl: float = aa.DEFAULT_TTL
    micro: tuple[str, ...] = ()
    home: Path = field(default_factory=default_home)
    key: str = ""
    cpkey: str = ""

    @classmethod
    def from_env(cls, env_file: Path | None = None):
        load_env(env_file)
        return cls(
            litellm_base=os.environ.get("LITELLM_BASE", cls.litellm_base),
            cliproxy_base=os.environ.get("CLIPROXY_BASE", cls.cliproxy_base),
            aa_base=os.environ.get("AA_BASE", cls.aa_base),
            aa_ttl=float(os.environ.get("AA_TTL", cls.aa_ttl)),
            key=os.environ.get("LITELLM_MASTER_KEY", ""),
            cpkey=os.environ.get("CPAPI_API_KEY", ""),
        )


def load_env(path: Path | None = None):
    """Read KEY=VALUE pairs without overwriting the real environment."""
    path = path or (os.environ.get("AUTO_POOL_ENV_FILE") or None)
    if not path:
        return
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"env file not found: {path}")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


def filtered(names, prefix: str = "", suffixes=()) -> list[str]:
    """Direct model ids, optionally narrowed by name prefix and suffix.

    Managed routers and internal advisory groups are never candidates, whichever
    catalogue they came from.
    """
    allowed = tuple(suffixes)
    found = []
    for name in names:
        if not name or "/" in name or name in ROUTERS or name.startswith("__"):
            continue
        if prefix and not name.startswith(prefix):
            continue
        if allowed and not name.endswith(allowed):
            continue
        found.append(name)
    return sorted(set(found))


def discover(client: Client, prefix: str = "", suffixes=()) -> list[str]:
    """Catalogue model ids from an optional catalogue proxy."""
    payload = client.get("/v1/models")
    return filtered([item["id"] for item in payload.get("data", [])], prefix, suffixes)


def catalogue(rows: list[dict], prefix: str = "", suffixes=()) -> list[str]:
    """Candidate ids from LiteLLM's own catalogue, used when no proxy is set.

    A row carrying a complexity router config is a router rather than a model,
    so it is never a candidate however it is named.
    """
    names = [
        row["model_name"]
        for row in rows
        if "complexity_router_config" not in (row.get("litellm_params") or {})
    ]
    return filtered(names, prefix, suffixes)


def selected_models(rows: list[dict]) -> list[str]:
    """Models the managed routers currently hand requests to."""
    names: set[str] = set()
    for row in rows:
        if row["model_name"] not in ROUTERS:
            continue
        config = (row.get("litellm_params") or {}).get("complexity_router_config") or {}
        for tier_names in (config.get("tiers") or {}).values():
            names.update(tier_names)
    return sorted(names)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_state(path: Path, fingerprint_value: str, hashes: dict[str, str]):
    path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint_value,
                "scores": hashes,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        + "\n"
    )


def csv(value) -> tuple[str, ...]:
    return tuple(part.strip() for part in (value or "").split(",") if part.strip())


def required_env(settings: "Settings") -> list[str]:
    """Only the LiteLLM admin key is mandatory; a catalogue proxy is optional."""
    return [name for name, value in (("LITELLM_MASTER_KEY", settings.key),) if not value]


# A single health call can catch a deployment just after a token bucket refilled,
# so flakiness is only reported when the same deployment both passes and 429s.
DEPLOYMENT_SAMPLES = 3


def triage_deployments(litellm, rows, name, *, apply=False):
    """Disable only deployments that are intermittently rate-limiting.

    Returns ``(blocked_ids, note)``. Blocking is skipped when the group has no
    other deployment (the model itself would disappear) or when no sibling was
    stable across every sample (the whole group is down, not one bad endpoint).
    """
    group = [
        row
        for row in rows
        if row["model_name"] == name
        and not (row.get("model_info") or {}).get("blocked")
    ]
    ids = []
    for row in group:
        model_id = (row.get("model_info") or {}).get("id")
        if model_id:
            ids.append(model_id)
    if len(ids) < 2:
        return [], "single_deployment"

    samples = {
        model_id: [
            deployment_health(litellm, model_id) for _ in range(DEPLOYMENT_SAMPLES)
        ]
        for model_id in ids
    }
    flaky = flaky_deployments(samples)
    if not flaky:
        return [], "none_flaky"
    if not stable_deployments(samples):
        return [], "no_stable_sibling"
    if not apply:
        return flaky, "would_block"

    blocked = []
    for model_id in flaky:
        result = litellm.post("/model/block", {"model_id": model_id})
        if result.get("error_status") is not None:
            continue
        for row in rows:
            info = row.get("model_info") or {}
            if info.get("id") == model_id:
                info["blocked"] = True
        blocked.append(model_id)
    return blocked, "blocked"


def usable_micro(litellm, rows, candidates, requested, *, probe_failures=None):
    """Micro models that can serve this run.

    Reuse failed candidate probes and skip probing scored candidates again.
    Other requested models are checked against registration and blocked flags,
    then probed to see whether they can serve a light tier.
    """
    chosen = []
    reasons = {}
    for name in requested:
        if probe_failures and name in probe_failures:
            reasons[name] = probe_failures[name]
            continue
        usable, reason = group_state(rows, name)
        if not usable:
            reasons[name] = reason
        elif name in candidates:
            chosen.append(name)
        else:
            ok, reason = probe(litellm, name)
            if ok:
                chosen.append(name)
            else:
                reasons[name] = reason
    return chosen, reasons


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes (default: preview)")
    parser.add_argument("--watch", action="store_true", help="skip refresh when catalogue and policy match")
    parser.add_argument("--refresh-scores", action="store_true", help="re-read every AA page")
    parser.add_argument("--smoke", action="store_true", help="one tiny auto completion after applying")
    parser.add_argument(
        "--register", action="store_true",
        help="register declared guest models that the catalogue does not carry",
    )
    parser.add_argument(
        "--guests", type=Path, default=None,
        help="guest declarations (default: <home>/auto-pool-guests.json)",
    )
    parser.add_argument(
        "--live-fallbacks", action="store_true",
        help="keep every fallback chain ending on a live model (own run, no tier write)",
    )
    parser.add_argument(
        "--quarantine", action="store_true",
        help="probe the routers' models and align their block flags with the answer",
    )
    parser.add_argument(
        "--sync-catalog", action="store_true",
        help="register upstream catalogue models the gateway does not carry yet",
    )
    parser.add_argument(
        "--sync-catalogs", action="store_true",
        help="sync the Cursor, Anthropic and CLIProxy catalogues and exit",
    )
    parser.add_argument(
        "--sync-batch", action="store_true",
        help="register public batch models as mode=batch and exit",
    )
    parser.add_argument(
        "--maintenance", action="store_true",
        help="apply missing prompt_cache_key drops and exit",
    )
    parser.add_argument(
        "--repair-cooldowns", action="store_true",
        help="cap pooled cooldowns, clear only proven recoveries, and exit",
    )
    parser.add_argument(
        "--health-report", action="store_true",
        help="write a bounded health report for every deployment and exit",
    )
    parser.add_argument(
        "--prices", type=Path, default=None,
        help="pinned LiteLLM price map for --live-fallbacks",
    )
    parser.add_argument(
        "--seed-file", type=Path, default=None,
        help='JSON {"group": "deployment-id"} to unblock for --live-fallbacks',
    )
    parser.add_argument("--env-file", type=Path, default=None, help="file with KEY=VALUE secrets")
    parser.add_argument("--home", type=Path, default=None, help="state, cache and backup directory")
    parser.add_argument("--prefix", default="", help="only models whose id starts with this")
    parser.add_argument("--suffixes", default="", help="comma separated id suffixes, e.g. sol,astra")
    parser.add_argument("--micro", default=None, help="comma separated small models for light tiers")
    parser.add_argument(
        "--catalogue-proxy", action="store_true",
        help="read candidate ids from CLIProxyAPI instead of LiteLLM's registered models",
    )
    return parser.parse_args(argv)


def run(args) -> int:
    settings = Settings.from_env(args.env_file)
    if args.home:
        settings.home = args.home
    if args.micro is not None:
        settings.micro = csv(args.micro)

    if args.live_fallbacks:
        # The fallback sanitizer ships in this package (``live.py``). Running it
        # through this flag leaves the host nothing to schedule but one script,
        # instead of a wrapper that re-declares the same paths and flags.
        from .live import main as live_main

        argv = ["--apply"] if args.apply else []
        if args.env_file:
            argv += ["--env-file", str(args.env_file)]
        argv += ["--base-url", settings.litellm_base]
        if args.prices:
            argv += ["--prices", str(args.prices)]
        if args.seed_file:
            argv += ["--seed-file", str(args.seed_file)]
        return live_main(argv)
    missing = required_env(settings)
    if missing:
        print(json.dumps({"error": "missing environment", "names": missing}))
        return 1

    settings.home.mkdir(parents=True, exist_ok=True)
    lock_path = settings.home / "auto-pool-refresh.lock"
    lock_path.touch()
    handle = lock_path.open("r")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(json.dumps({"skipped": "locked"}))
        return 0

    litellm = Client(settings.litellm_base, settings.key)

    # The provider-specific syncs and upkeep jobs keep their own cadence but no
    # longer need a script of their own: one cron entry now calls this program.
    if (args.sync_catalogs or args.sync_batch or args.maintenance
            or args.repair_cooldowns or args.health_report):
        rows = models(litellm)
        catalog_path = args.prices or (settings.home / "litellm-model-prices-main.json")
        prices = {}
        try:
            if catalog_path and Path(catalog_path).exists():
                prices = json.loads(Path(catalog_path).read_text())
        except (OSError, ValueError):
            prices = {}
        output = {}
        if args.sync_catalogs:
            output["catalogs"] = sync_provider_catalogs(
                litellm, settings.home, rows, prices, apply=args.apply
            )
        if args.sync_batch:
            output["batch_sync"] = sync_batch(
                litellm, rows, settings.home, prices=prices, apply=args.apply
            )
        if args.maintenance:
            output["prompt_cache_drop"] = prompt_cache_drop_pass(
                litellm, rows, apply=args.apply
            )
            output["provider_prefix"] = provider_prefix_pass(
                litellm, rows, apply=args.apply
            )
            output["context_sync"] = context_sync_pass(
                litellm, rows, prices, apply=args.apply
            )
            output["effort_suffix"] = effort_suffix_pass(
                litellm, rows, prices, apply=args.apply
            )
        if args.repair_cooldowns:
            output["cooldowns"] = repair_cooldowns(
                litellm, rows, settings.home, apply=args.apply
            )
        if args.health_report:
            output["health"] = health_report(
                litellm, rows, settings.home, redact=settings.key
            )
        print(json.dumps(output, ensure_ascii=False))
        return 0

    rows = models(litellm)

    # Registration metadata is a claim; the external price catalogue is the
    # reference for which upstream models can actually answer a decision request.
    catalog_path = args.prices or (settings.home / "litellm-model-prices-main.json")
    catalog = {}
    try:
        if catalog_path and Path(catalog_path).exists():
            catalog = json.loads(Path(catalog_path).read_text())
    except (OSError, ValueError):
        catalog = {}
    modality_scores = load_modality_benchmarks(settings.home)

    # A model missing from the registry is invisible to the router, and the
    # upstream catalogue is the only place a new one exists. Registration is
    # separate from the block sweep: only a probe can prove a deployment dead,
    # while an account's credit cannot be proven from one deployment.
    catalog_sync = None
    if args.sync_catalog:
        catalog_sync = sync_catalog(
            litellm, rows, settings.home, prices=catalog, apply=args.apply
        )
        if args.apply and catalog_sync.get("actions"):
            rows = models(litellm)  # the new groups are routable from here on

    # Guests are models the operator can use that no catalogue sync carries: a
    # free upstream router, a decision endpoint, a preview alias. They are
    # declared in a file and registered here, because a model the router does
    # not know about cannot be routed to at all.
    guests = load_guests(args.guests or (settings.home / "auto-pool-guests.json"))
    guest_results: list[dict] = []
    if guests and args.register:
        if args.apply:
            guest_results = ensure_guests(litellm, guests, rows)
            rows = models(litellm)  # the new groups are routable from here on
        else:
            guest_results = [
                {
                    "model": guest["model_name"],
                    "action": (
                        "present" if registered(rows, guest["model_name"])
                        else "would_register"
                    ),
                }
                for guest in guests
            ]

    # Blocks are capacity, so the same probe that chooses a model keeps the flags
    # honest: a group that answers is unblocked, one that cannot is narrowed.
    budget_quarantine: list[dict] = []
    quarantine: list[dict] = []
    if args.quarantine:
        # Budget is per key: one budget-exceeded answer blocks every deployment
        # sharing that credential, even when a sibling key still answers.
        budget_quarantine = budget_sweep(
            litellm, rows,
            lambda model_id: deployment_health(litellm, model_id),
            apply=args.apply,
        )
        if budget_quarantine and args.apply:
            rows = models(litellm)

        managed: set[str] = set()
        for row in rows:
            if row["model_name"] not in ROUTERS:
                continue
            config = (row.get("litellm_params") or {}).get(
                "complexity_router_config"
            ) or {}
            for entries in (config.get("tiers") or {}).values():
                managed.update(entries)
        # Only chat solvers: a media tier is judged by its own modality, and a
        # decision model is handled by the classifier probe, not a chat probe.
        managed = {
            name for name in managed
            if any(
                row["model_name"] == name
                and str((row.get("model_info") or {}).get("mode") or "chat")
                in ("chat", "completion")
                for row in rows
            )
        }
        quarantine = quarantine_sweep(
            litellm, rows, managed,
            lambda name: probe(litellm, name), apply=args.apply,
        )
        if quarantine and args.apply:
            rows = models(litellm)

    # LiteLLM's registrations are the routing truth: a model the proxy lists but
    # LiteLLM does not register can never serve a request. The proxy stays an
    # explicit opt-in for setups that treat its /v1/models as authoritative.
    if args.catalogue_proxy and settings.cpkey:
        cliproxy = Client(settings.cliproxy_base, settings.cpkey, timeout=30)
        names = discover(cliproxy, args.prefix, csv(args.suffixes))
        source = "cliproxy"
    else:
        names = catalogue(rows, args.prefix, csv(args.suffixes))
        source = "litellm"
    if not names:
        print(json.dumps({"error": "no models discovered"}))
        return 1
    current_fp = fingerprint(rows, names)
    quarantined: list[str] = []

    state_path = settings.home / "auto-pool-state.json"
    cache_path = settings.home / "auto-pool-aa-cache.json"
    pending_path = settings.home / "auto-pool-fallback-pending"
    fallback_pending = pending_path.exists()
    state = load_json(state_path)
    if (
        args.watch
        and not fallback_pending
        and state.get("fingerprint") == current_fp
        and not context_drift(rows, ROUTERS)
        and not classifier_drift(rows, ROUTERS, catalog)
    ):
        # Cooldowns and recoveries never touch the catalogue, so a fingerprint
        # match alone would keep a model that can no longer answer. Re-plan only
        # when a selected model is actually unavailable.
        unavailable = [
            name for name in selected_models(rows) if not probe(litellm, name)[0]
        ]
        if unavailable and args.apply:
            for name in unavailable:
                blocked, _ = triage_deployments(litellm, rows, name, apply=True)
                quarantined.extend(blocked)
            if quarantined:
                current_fp = fingerprint(rows, names)
                unavailable = [
                    name for name in selected_models(rows)
                    if not probe(litellm, name)[0]
                ]
        if not unavailable and not quarantined:
            print(json.dumps({"skipped": "no catalogue change", "fingerprint": current_fp[:12]}))
            return 0

    cache = load_json(cache_path)
    previous_scores = state.get("scores", {})

    status: dict[str, str] = {}
    probe_failures: dict[str, str] = {}
    observed_usage: dict[str, dict] = {}

    # A model can borrow another model's Artificial Analysis ladder and price
    # when it has no page of its own. Corrections come from model_info.base_model
    # (or aa_base_model / benchmark_base) on any deployment, or from
    # <home>/auto-pool-base-models.json: {"gateway-model": "base-model"}.
    base_models: dict[str, str] = {}
    override_path = settings.home / "auto-pool-base-models.json"
    if override_path.exists():
        try:
            loaded = json.loads(override_path.read_text())
        except (OSError, ValueError):
            loaded = None
        if isinstance(loaded, dict):
            base_models.update(
                {str(key): str(value) for key, value in loaded.items()}
            )
    for row in rows:
        info = row.get("model_info") or {}
        base = (
            info.get("base_model")
            or info.get("aa_base_model")
            or info.get("benchmark_base")
        )
        if base:
            base_models.setdefault(row["model_name"], str(base))
    candidates: dict[str, dict] = {}
    structural: list[str] = []

    def note_usage(observation) -> None:
        """Keep a probe's usage verdict without changing availability."""
        if isinstance(observation, dict):
            observed_usage[name] = observation

    for name in names:
        usable, reason = group_state(rows, name)
        if not usable:
            status[name] = reason
            continue
        ok, reason, observation = probe_with_usage(litellm, name)
        note_usage(observation)
        if not ok:
            blocked, note = triage_deployments(
                litellm, rows, name, apply=args.apply
            )
            if blocked and args.apply:
                quarantined.extend(blocked)
                ok, reason, observation = probe_with_usage(litellm, name)
                note_usage(observation)
            elif blocked:
                reason = f"{reason};would_block={','.join(blocked)}"
            elif note in ("single_deployment", "no_stable_sibling"):
                reason = f"{reason};{note}"
        status[name] = reason
        if not ok:
            probe_failures[name] = reason
            continue
        base = base_models.get(name, name)
        aa_slug = base.removesuffix(":free").split("/")[-1].replace(".", "-")
        variants, error = aa.get_variants(
            aa_slug,
            cache,
            base=settings.aa_base,
            ttl=settings.aa_ttl,
            force=args.refresh_scores,
        )
        if error in aa.STRUCTURAL and not variants:
            structural.append(name)
            status[name] = error
        elif not variants:
            status[name] = "no_aa_entry"
        else:
            candidates[name] = variants

    if quarantined and args.apply:
        # The block flags are part of the fingerprint, so the state must record
        # the catalogue as it stands after quarantine.
        current_fp = fingerprint(rows, names)

    if structural:
        # Deliberately no state write: the watcher keeps retrying instead of
        # pinning a fingerprint that would hide the broken model forever.
        print(
            json.dumps(
                {"result": "aa_unparseable", "models": structural, "status": status},
                ensure_ascii=False,
            )
        )
        return 1

    hashes = score_hashes(cache)
    updates = changed_scores(hashes, previous_scores)
    cache_path.write_text(json.dumps(cache))

    # Roles are solved, not capped: the cheapest model that reaches each role's
    # measured floor wins, so no price is written into the policy.
    assigned = roles(candidates)
    premium = assigned["premium"]
    primary = assigned["primary"]
    budget = assigned["budget"]
    if not (premium and primary and budget):
        print(json.dumps({"error": "insufficient valid candidates", "status": status}))
        return 1
    runner_up = assigned["runner_up"]
    if not runner_up:
        print(json.dumps({"error": "no runner-up candidate", "status": status}))
        return 1
    runner_up = runner_up[0]

    # Light tiers need small models; by default they come from the pool itself,
    # so nothing has to be named on the command line. ``--micro`` stays as an
    # explicit override, and a model that cannot serve this run is dropped from
    # the light tiers instead of being written back because its declared window
    # is large enough.
    if settings.micro:
        micro, micro_status = usable_micro(
            litellm, rows, candidates, settings.micro, probe_failures=probe_failures
        )
        status.update(micro_status)
    else:
        micro = pick_micro(
            candidates, [premium[0], primary[0], budget[0], runner_up]
        )
    # A group whose deployments cost nothing leads its model group, so the
    # router tries it before any paid sibling of the same score. The probe's
    # usage object is direct evidence: a stale :free marker does not outrank a
    # positive measured cost, and a measured zero-cost call can add a group the
    # marker missed.
    free = free_models(rows, observed_usage)
    top = {name: top_effort(variants) for name, variants in candidates.items()}
    frontier = place(
        [premium[0], primary[0], runner_up], candidates, free_models=free
    )
    # Fewest score-equal tiers that cover every measured level of this roster.
    tier_plan = schedule(
        [premium[0], primary[0], budget[0], runner_up], candidates
    )
    specialized = specialized_models(rows, catalog, modality_scores)
    # The complexity router has image and video tiers only; audio and OCR are
    # ranked in the summary but must not be handed to a text/media tier.
    media_specialized = {
        kind: specialized.get(kind, []) for kind in ("image", "video")
    }
    router_map = routers(
        primary[0], premium[0], budget[0], runner_up, micro, top,
        ladders=candidates,
        windows={name: input_window(rows, name) for name in candidates},
        specialized=media_specialized,
        free_models=free,
    )
    # The classifier contract travels with the tiers: LiteLLM rejects a tier map
    # that does not cover every defined tier, so both are written in one patch.
    # The default path has no tier config, so it answers at the model's own
    # default effort; rank by what each candidate can promise there.
    default, default_note = pick_default(
        rows, candidates, [primary[0], premium[0], runner_up]
    )
    # A decision model answers "which tier" in one forward pass, so it is the
    # classifier whenever the gateway can actually call one; the LLM judge is the
    # fallback, not the default.
    # Metadata only says a model should answer; the router only works if it
    # actually does, so each candidate is tried until one returns a real choice.
    # A decision model that fails its probe must not take the whole classifier
    # down when a sibling can still answer.
    # AA's Decision Index is the performance source: it scores decision-only
    # models on the same task shape the classifier asks, which the chat model
    # pages cannot. Missing scores are not fatal; price still orders those.
    decision_cache_path = settings.home / "auto-pool-decision-cache.json"
    decision_cache = load_json(decision_cache_path)
    capable = [
        item["model"] for item in classifier_candidates(rows, catalog)
        if item["usable"]
    ]
    decision_scores: dict[str, float] = {}
    if capable:
        benchmark, benchmark_errors = aa.get_decision_scores(
            capable, decision_cache,
            ttl=settings.aa_ttl, force=args.refresh_scores,
            aliases=base_models,
        )
        decision_scores = {
            name: entry["score"] for name, entry in benchmark.items()
            if entry.get("score") is not None
        }
        decision_cache_path.write_text(json.dumps(decision_cache))
        for name, reason in benchmark_errors.items():
            status.setdefault(name, f"benchmark_{reason}")

    decision = None
    for candidate in decision_candidates(rows, catalog, decision_scores):
        ok, reason = probe_decision_choice(litellm, candidate)
        if ok:
            decision = candidate
            break
        status[candidate] = f"classifier_{reason}"
    classifier = (
        {"type": DECISION_CLASSIFIER, "model": decision} if decision
        else {"type": "llm", "model": runner_up}
    )
    changes = plan(
        rows, router_map, default, classifier,
        extras=policies(router_map),
    )
    summary = {
        "catalogue": source,
        "pool": {
            "premium": premium[0],
            "primary": primary[0],
            "budget": budget[0],
            "runner_up": runner_up,
        },
        "index": {
            "premium": round(premium[2], 1),
            "primary": round(primary[2], 1),
            "budget": round(budget[2], 1),
        },
        "frontier": {
            "target": None if frontier["target"] is None else round(frontier["target"], 2),
            "spread": round(frontier["spread"], 2),
            "efforts": dict(router_map["auto"][FRONTIER]),
        },
        "specialized": specialized,
        # Free groups the pool cannot route yet, so a registered-but-blocked or
        # still-unmeasured free model is visible instead of silently absent.
        "free_gaps": free_gaps(rows, candidates, observed_usage),
        "free_usage": {
            "free": sorted(
                name for name, observation in observed_usage.items()
                if observation.get("free") is True
            ),
            "paid": sorted(
                name for name, observation in observed_usage.items()
                if observation.get("free") is False
            ),
            "unknown": sum(
                1 for observation in observed_usage.values()
                if observation.get("free") is None
            ),
        },
        # Decision models answer a schema, not a chat turn: never tier members.
        "decision_models": decision_models(rows),
        # Gateway models whose AA ladder/price was borrowed from a base model.
        "base_models": base_models,
        # Batch models are registered for /v1/batches and never chat tiers.
        "batch_models": batch_models(rows),
        "guests": guest_results,
        "catalog_sync": catalog_sync,
        "default_model": {"model": default, **default_note},
        "classifier": classifier,
        "quarantine": quarantine,
        "budget_quarantine": budget_quarantine,
        # Every group that could classify if it answered: a decision-only model,
        # or a chat model that also serves /v1/decisions.
        "classifier_candidates": classifier_candidates(
            rows, catalog, decision_scores
        ),
        # The catalogue's own decision models: what to look for (and register)
        # when no deployed deployment can classify yet.
        "decision_catalog": decision_catalog(catalog),
        "tier_plan": {
            "tiers": len(tier_plan),
            "band": SPREAD,
            "assignments": [
                {
                    "score": round(tier["target"], 2),
                    "until": round(tier["covered"], 2),
                    "entries": {
                        model: effort for model, effort, _ in tier["entries"]
                    },
                }
                for tier in tier_plan
            ],
        },
        "score_updates": updates,
        "quarantined": sorted(set(quarantined)),
        "status": status,
        "minimum_input_tokens": MIN_INPUT_TOKENS,
        "changed": len(changes),
        "max_input_tokens": {
            change["model_name"]: change["model_info"]["max_input_tokens"]
            for change in changes
            if change.get("model_info")
        },
    }

    if not changes and not (args.apply and fallback_pending):
        if not fallback_pending:
            save_state(state_path, current_fp, hashes)
        print(json.dumps({**summary, "result": "unchanged"}, ensure_ascii=False))
        return 0

    output = {**summary, "result": "applied"}
    try:
        if changes:
            validate(litellm, changes)
            output["backup"] = str(backup(changes, settings.home))
            if not args.apply:
                print(json.dumps({**output, "result": "preview"}, ensure_ascii=False))
                return 0

            # Model writes may persist without the fallback update. Mark the
            # refresh pending before patching so watch cannot skip recovery.
            pending_path.touch()
            apply_changes(litellm, changes)
        set_auto_fallback(litellm, [premium[0], primary[0], runner_up, budget[0]])
    except (ValueError, RuntimeError) as error:
        print(json.dumps({**summary, "result": "rejected", "error": str(error)}))
        return 1

    save_state(state_path, current_fp, hashes)
    pending_path.unlink()
    if args.smoke:
        output["smoke"] = smoke(litellm)
    print(json.dumps(output, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
