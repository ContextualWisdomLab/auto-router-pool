# auto-router-pool

Keeps LiteLLM complexity routers on the best model that is **both reachable right
now and cheap for its measured capability**. There is no scheduler in this
package; it is one idempotent run that belongs in cron.

## Why

A model can be registered, healthy, and still useless: deployments get blocked,
providers rate-limit, and newer versions appear at the same price with a better
score. Picking a pool by hand goes stale within days, and picking by list price
alone ignores that a cheap model at `low` effort can score two thirds lower than
the same model at `max`.

## What one run does

1. **Discover** model ids, optionally narrowed by `--prefix` / `--suffixes`; with
   neither flag the whole catalogue is considered. Candidates come from
   LiteLLM's own `/model/info` by default, so no extra service is required; a
   separate catalogue proxy (CLIProxyAPI) is used only when `CPAPI_API_KEY` is
   set, which keeps `/v1/models` authoritative for setups that prefer it.
2. **Filter to valid models**: a declared input window of at least 272,000
   tokens, at least one unblocked deployment, and a one-token probe answered by
   the model itself. Any error (`400`/`429`/`5xx`/timeout) or an answer produced
   by a fallback means the model cannot serve this run, so it is not a candidate.
   Nothing is written to the catalogue, so the next run re-admits the model as
   soon as it answers. A model below the 272k floor is never a candidate or a
   `--micro` entry.
3. **Fetch scores** from the Artificial Analysis model page, cached per slug.
4. **Pick roles** under price caps:

   | role | rule | cap (per 1M in/out) |
   |---|---|---|
   | `premium` | best index at `xhigh` | $12 / $45 |
   | `primary` | within 2 points of the best at `high`, prefer models without the `-astra` suffix, then lowest price | $3 / $12 |
   | `budget` | best index per blended dollar at `medium` | $0.30 / $1.50 |

   The complexity classifier is pinned to the `runner_up` model with a 30s
   timeout. It runs on every routed request, so it must be a model the router
   already selects and one with headroom: a stale classifier times out, and the
   cheapest model is usually the first to be rate-limited.
5. **Diff** each router's tiers, efforts, context floor and classifier, validate
   server-side, back up, patch, re-read, and roll back on any mismatch.
6. **Cap the advertised window** at the smallest `max_input_tokens` among the
   models its own tiers can select, so a router never advertises more context
   than the weakest solver it can hand a request to. Only models at or above the 272k input floor can be selected, so the advertised
   window lands on the weakest *eligible* solver. A solver with no declared
   window proves no floor, so that router is left alone instead of guessed.

An unchanged pool writes nothing.

## Effort ladder matters

Scores are per effort, not per model. Measured on the Artificial Analysis
Intelligence Index (2026-11), three variants of the same roster:

| variant | max | xhigh | high | medium | low |
|---|---|---|---|---|---|
| premium | 57.3 | 55.7 | 54.9 | 52.7 | 40.6 |
| primary | 62.1 | 61.6 | 60.1 | 57.0 | 45.7 |
| budget | 27.9 | 23.5 | 21.4 | 18.5 | 9.7 |

The budget variant at `low` scores a third of what it does at `max`, so the light
tiers run it at `medium`/`high`. The frontier tier runs every entry at its own
measured peak, so its effort labels can differ: the primary variant gains under
a point from `xhigh` over `high`, while the budget variant peaks at `max`
(27.9 vs 23.5). Peak effort maximizes each model; it does not equalize them,
and a cheaper fallback still scores lower than the frontier model.

## Flaky deployments

A model can be registered more than once (one deployment per credential or
upstream route). When a probe fails, the run samples each unblocked deployment
three times through `/health?model_id=`. A deployment that both answers and
returns a 429 in that window is disabled with `/model/block`; the model is then
probed again and stays eligible if a sibling can serve it. Blocking is skipped
when the group has only one unblocked deployment, or when no sibling answered
every sample: in those cases the outage is the model, not one bad route, and
disabling the last deployment would remove the model entirely. Preview runs
report `would_block` in the status instead of writing, and applied runs list the
blocked ids under `quarantined`.

## Cadence

`--watch` compares a fingerprint of the catalogue and blocked flags and exits
when nothing changed. A context floor that drifted from its own tiers also
forces a run, so a cap change is never masked by an unchanged catalogue. Before
skipping, it probes only the models the routers currently select, because a
cooldown or a recovery never changes the catalogue fingerprint: if a selected
model cannot answer, the run re-plans against the models that can. That keeps a
15-minute cron cheap: no score traffic unless something is actually out of date.

```cron
*/15 * * * * cd /path/to/auto-router-pool && PYTHONPATH=src /usr/bin/python3 -m auto_router_pool.cli --watch --apply >> "$HOME/.local/state/auto-router-pool/refresh.log" 2>&1
# Safety net: a full re-score and re-plan even when the catalogue is unchanged.
3 4,12,20 * * * cd /path/to/auto-router-pool && PYTHONPATH=src /usr/bin/python3 -m auto_router_pool.cli --refresh-scores --apply >> "$HOME/.local/state/auto-router-pool/refresh.log" 2>&1
```

Scores are cached for 7 days (`AA_TTL`) because a released model's index barely
moves; a newly registered model has no cache entry and is fetched on first sight.
Use `--refresh-scores` to re-read everything on demand.

## Chain hygiene

`auto-router-pool-chains` keeps `fallbacks` usable on its own schedule: it drops
targets that are blocked or below the 272k floor, ends every chain on a live
resort, and never lets the resort keep a chain of its own (which could close a
cycle). The resort is chosen from the catalogue — the cheapest model from a
family the auto tiers do not already use — and an incumbent keeps the role while
it still qualifies, so a transient price change cannot churn the graph. It fails
closed when nothing qualifies.

```cron
11,26,41,56 * * * * /usr/bin/python3 /path/to/ensure_live_fallbacks.py
```

## Failure behaviour

| situation | result |
|---|---|
| page layout changed, model uncached | `{"result": "aa_unparseable"}`, exit 1, **no state write** → retried next run |
| page parsed, model not benchmarked, or its page 404s | `no_aa_entry`, model skipped |
| cached model, page now broken | last known numbers used until the cache expires |
| `LITELLM_MASTER_KEY` missing | `{"error": "missing environment"}`, exit 1 |
| another run holds the lock | `{"skipped": "locked"}`, exit 0 |
| probe errors (`429`/`5xx`/timeout) or a fallback answers | `unavailable_*` / `fallback_*`, model is not a candidate for that run |
| a selected model is unavailable while the catalogue is unchanged | `--watch` re-plans instead of skipping |
| server rejects a tier config | `{"result": "rejected"}`, exit 1, nothing written |
| patch does not persist | all routers restored from the backup |

The "no state write" rule is the important one: saving the fingerprint while a
model failed to parse would hide that model forever.

## Install

```bash
python3 -m pip install -e .
auto-router-pool --help
```

Requires Python 3.10+. No third-party runtime dependencies.

## Configuration

| variable | default | meaning |
|---|---|---|
| `LITELLM_MASTER_KEY` | — | LiteLLM admin key (required) |
| `CPAPI_API_KEY` | — | optional catalogue proxy key, used only with `--catalogue-proxy` |
| `LITELLM_BASE` | `http://127.0.0.1:4000` | LiteLLM base URL |
| `CLIPROXY_BASE` | `http://127.0.0.1:8317` | catalogue proxy base URL, used only with `CPAPI_API_KEY` |
| `AA_BASE` | `https://artificialanalysis.ai/models/` | score source, override for tests |
| `AA_TTL` | `604800` | score cache lifetime, seconds |
| `AUTO_POOL_HOME` | `~/.local/state/auto-router-pool` | state, cache, backups, lock |
| `AUTO_POOL_ENV_FILE` | — | file with `KEY=VALUE` secrets |

Flags: `--apply`, `--watch`, `--refresh-scores`, `--smoke`, `--prefix`,
`--suffixes`, `--micro`, `--catalogue-proxy`, `--home`, `--env-file`.

Candidate ids come from LiteLLM's `/model/info`, which is the routing truth: a
model the proxy lists but LiteLLM does not register cannot serve a request. The
managed routers and `__`-prefixed advisory groups are skipped either way. The
catalogue proxy is an explicit opt-in for setups where its `/v1/models` is
authoritative; the run report names the source in `catalogue` (`litellm` or
`cliproxy`).

Nothing about a particular deployment is hardcoded: model families, the small
models used for light tiers, and every path are supplied by the caller. The
repository contains no keys, no host addresses beyond the upstream defaults, and
no runtime state.

## Tests

```bash
python3 -m pytest -q
# or, if an unrelated installed plugin stalls collection:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
python3 -m unittest discover -s tests   # stdlib only
```

The suite covers the page parser (including the layout-break and
cached-after-break paths), discovery filters, role selection, fingerprint
semantics, and the rule that a refetch with identical numbers is not a score
change.

## Limits

- **Artificial Analysis is scraped.** The parser fails loudly rather than
  silently, but commercial use needs the official API; see the terms.
- **No budget control.** Selection is per tier and separable, so greedy is
  optimal here. A global spend budget would turn this into a small MILP over
  tiers and models, solved per day rather than per request.
- **Single host.** State is a local JSON file and the lock is `flock`; a
  multi-worker deployment needs shared state and a distributed lock.
- **Discovery is name-based.** Ids are taken from the catalogue verbatim and
  ids containing `/` are skipped; use `--prefix`/`--suffixes` to keep a pool in
  one family when the catalogue mixes vendors.
