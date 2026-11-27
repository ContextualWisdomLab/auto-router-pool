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
   With `--sync-catalog` the upstream Experiential Labs catalogue is read first
   and every active text model the account can call is registered if it is
   missing. A model routed by any provider is eligible, not only ones served by
   `experiential_cloud`, so a new model appears without a separate cron. A model
   the catalogue describes as a decision model (`/v1/systemone`) is registered on
   the native SystemOne transport instead of the OpenAI chat one, which would
   send `/v1/decisions` and be refused. A **batch** entry (`-batch` name,
   `:batch` route or a `/v1/batches` endpoint) is registered with
   `model_info.mode: batch`, and the `:batch` price-map suffix is stripped from
   the model id because the provider's batch API takes the normal slug. A model
   that advertises chat as well as batches stays a chat model. An older
   deployment already registered as `chat` is still treated as batch when its
   group, display name or upstream carries a batch marker, so it is never probed
   or routed as chat. The run reports
   how many batch entries the catalogue advertises, how many this account can
   actually call, and how many `:batch` prices exist. Batch variants are not chat
   models, so they are registered from the public catalogue even when
   `/v1/models` omits them; a group already stored as `chat` is updated in place
   to `mode: batch`. This stage never writes block flags:
   `--quarantine` owns those, because a probe can prove a deployment dead but an
   account's exhausted credit is not provable from one deployment.
2. **Filter to valid models**: a declared input window of at least 272,000
   tokens, at least one unblocked deployment, and a one-token probe answered by
   the model itself. Any error (`400`/`429`/`5xx`/timeout) or an answer produced
   by a fallback means the model cannot serve this run, so it is not a candidate.
   Nothing is written to the catalogue, so the next run re-admits the model as
   soon as it answers. A model below the 272k floor is never a candidate or a
   `--micro` entry.
3. **Fetch scores** from the Artificial Analysis model page, cached per slug.
4. **Solve roles** as a linear program over the measured ladders
   (`assign.roles`): minimise the modelled price of the four roles subject to
   each role reaching a floor taken from the roster itself.

   | role | floor (measured index) | effort |
   |---|---|---|
   | `premium` | within 1 point of the best | `xhigh` |
   | `primary` | within 2 points, and not the premium model | `high` |
   | `runner_up` | within 6 points, and neither of the two above | `high` |
   | `budget` | no floor: best index per modelled dollar | `medium` |

   No price is written into the policy. The cheapest model that still clears its
   floor wins, so a price change cannot silently demote a role, and a thin
   catalogue reuses the previous role instead of failing the run. Every cost
   comparison prices **one run from the input and output prices separately**,
   weighted by the tier's token mix (`assign.PROFILES`): extraction is
   prompt-heavy and long-horizon reasoning is answer-heavy, so averaging the two
   prices would rank them the same.

   `micro` is not chosen by a floor either: it is the two cheapest candidates the
   run already proved usable, at `low`, so the light tiers need no supplied list.

   The complexity classifier prefers a **decision model**: any registered model
   that serves `/v1/decisions` or `/v1/systemone` and has an unblocked deployment
   becomes `classifier_type: oss_classifier` with
   `opensource_classifier_config.deployment_name`, because asking "which tier" is
   one cheap forward pass rather than a whole chat completion. When no decision
   model can answer, it falls back to the LLM judge pinned to `runner_up` with a
   30s timeout -- a model the router already selects and one with headroom, since
   a stale classifier times out and the cheapest model is usually the first to be
   rate-limited. `classifier_drift` treats both a dead decision deployment and an
   LLM judge that outlived a usable decision model as drift, so the switch happens
   on the next run and reverses when credits or an upstream disappear.

   Priority is measured, not alphabetical. A model with an [Artificial Analysis
   Decision Index](https://artificialanalysis.ai/decision-models/) score is
   preferred over an unmeasured one and ranked by that score per **original**
   input dollar; a free tier is compared at its paid
   base's price, so zero price cannot manufacture a winner and only breaks a
   tie. When a registration carries no price, the LiteLLM price map and then
   the provider-route price recorded at registration fill it in. A model AA
   does not score falls back to original input price and name.

   Image, video, audio/transcription and OCR groups use the same comparison.
   Image and video feed their own router tiers; audio and OCR are ranked and
   reported in `specialized` but have no complexity-router tier yet. Scores are
   pinned from the Artificial Analysis Image, Video and Speech-to-Text
   leaderboards (audio is `100 - AA-WER`) and can be overridden in
   `<home>/auto-pool-modality-benchmarks.json`. OCR has no public benchmark, so
   it is ordered by original input price. A model with neither a score nor a
   price source is ranked last rather than guessed.

   A model that AA does not score can borrow another model's ladder and price
   by setting `model_info.base_model` (or `aa_base_model` / `benchmark_base`) on
   one of its deployments, or by writing
   `<home>/auto-pool-base-models.json` as `{"gateway-model": "base-model"}`.
   The `openai/` prefix on `litellm_params.model` is LiteLLM's provider
   namespace for the OpenAI-compatible gateway route; it is not the model's
   publisher and is ignored when matching prices or benchmarks.

   The *default* model -- the path a request takes when no tier is chosen -- has
   no `tier_model_configs` entry, so it answers at its own default effort rather
   than at the rung its best score was measured at. Candidates are ranked at
   that effort: at the deployment's declared `reasoning_effort` when one exists,
   and otherwise at the lowest rung they measured, which is the floor the default
   path can promise. Ties keep the pool's preference order.
5. **Diff** each router's tiers, efforts, context floor and classifier, validate
   server-side, back up, patch, re-read, and roll back on any mismatch.
6. **Cap the advertised window** at the smallest `max_input_tokens` among the
   models its own tiers can select, so a router never advertises more context
   than the weakest solver it can hand a request to. Only models at or above the 272k input floor can be selected, so the advertised
   window lands on the weakest *eligible* solver. A solver with no declared
   window proves no floor, so that router is left alone instead of guessed.

An unchanged pool writes nothing.

## Other catalogues and upkeep

The provider catalogues and the upkeep jobs are built into the same program, so
a host schedules one entry point instead of a directory of scripts:

| flag | what it does |
|---|---|
| `--sync-catalog` | register missing Experiential models (inside the pool run) |
| `--sync-batch` | register public batch models as `mode: batch` |
| `--sync-catalogs` | Cursor Cloud snapshot, Cursor chat bases, Anthropic official models, CLIProxyAPI `gemini-4-argon` |
| `--maintenance` | add `prompt_cache_key` to the drop list of providers that refuse it |
| `--repair-cooldowns` | cap pooled bridge cooldowns at 30s and clear one only after two real answers |
| `--health-report` | write a bounded per-deployment health report |

Each mode honours `--apply`; without it they preview. A missing credential, an
unreachable catalogue or a failed discovery is reported in the JSON summary and
skipped, so one provider never hides the others. None of these modes changes a
block flag: `--quarantine` owns those.

`--quarantine` also checks one deployment per static credential through
`/health`. When the answer is LiteLLM's `Budget has been exceeded` error, every
deployment sharing that credential is blocked and marked
`auto_pool_budget_blocked`; a later healthy answer lifts only that marker. Budget
is enforced per key, so a sibling deployment on another key cannot hide it.

## Effort ladder matters

Scores are per effort, not per model. Measured on the Artificial Analysis
Intelligence Index (2026-11), three variants of the same roster:

| variant | max | xhigh | high | medium | low |
|---|---|---|---|---|---|
| premium | 57.3 | 55.7 | 54.9 | 52.7 | 40.6 |
| primary | 62.1 | 61.6 | 60.1 | 57.0 | 45.7 |
| budget | 27.9 | 23.5 | 21.4 | 18.5 | 9.7 |

The budget variant at `low` scores a third of what it does at `max`, so the light
tiers run it at `medium`/`high`.

A tier's entries are fallbacks for the same request, so the frontier tier is
placed for **equal score** instead of at each model's own peak. The placement is
a small integer program (`assign.place`): maximise the chain's assumed score `T`
subject to one rung per model, every entry at or above `T`, and every entry
inside a `band` of it; then take the cheapest rung per model, preferring the
lower rung on a price tie. `T` is the weakest ceiling in the chain, because
nothing above it can be assumed equal.

On the roster above the three peaks are 62.2 / 57.3 / 54.7, so a fallback cost
up to 7.5 index points. Placed together they land at 57.1 / 54.8 / 54.7 (spread
2.4): the router now hands out one score whichever entry answers, and the
strongest model runs two rungs cheaper than its peak.

### How many tiers

The tier *count* falls out of the same measurement (`assign.schedule`). A tier
can only assume a score its own members reach, and members must stay inside the
band of each other, so covering the roster's achievable range with the fewest
tiers is an interval-cover problem, which the greedy sweep solves exactly. Every
run reports the result as `tier_plan`: the number of tiers the measured ladders
support, the score each tier assumes and the (model, effort) entries assigned to
it. A repeated entry set is merged instead of given a second name, so the plan
never publishes two tiers that hand out the same score.

### What the tiers divide

The `auto` tiers partition the *task*, not a model class: text (extraction,
conversation, review, long-horizon reasoning), code by scope (one change versus
system design), and media by **output** modality — an image or video generator
cannot answer a text request at all.

| tier | what the request must produce |
|---|---|
| `명시 정보 추출·단순 분류·고정 규칙 처리` | short, checkable output from explicit rules |
| `대화·요약·번역·창작 텍스트` | text-only answer; the catch-all tier |
| `코드·데이터·도구 사용` | one change, one query, one known procedure |
| `복잡한 코딩·시스템 설계·기업 지식 업무` | several systems, trade-offs, design or root cause |
| `최고난도 추론·심층 연구·복잡한 장기 작업` | no known path, long verification, competing hypotheses |
| `이미지·화면 해석·화면 조작` | image or screen *input*, text answer |
| `이미지 생성·편집` | image *output* |
| `비디오 생성·편집` | video *output* |

Precedence is fixed so nearly every request lands in exactly one tier: output
modality, then media input with a text answer, then code/tools, then text shape,
with `대화·요약·번역·창작 텍스트` as the catch-all. A tier with no model is dropped
instead of published empty, which is the failure mode that makes classification
fall through; the catch-all is also the `fallback_tier`, so a classifier that
errors still lands somewhere that can answer. Definitions, examples and the
classifier prompt are published in the same patch as the tiers, because the
engine rejects a tier map that does not cover every definition.

Only one router may carry operator-defined tiers or an operator-written
classifier prompt, and that budget belongs to `auto`. `auto-strix` (security:
static analysis, exploit reproduction, threat modelling, incident response) and
`auto-advisor` (model behaviour monitoring: drift, regression triage, root cause,
guardrail review) therefore keep the shipped `SIMPLE`/`MEDIUM`/`COMPLEX`/`REASONING`
tiers and differ in which models they hand each tier: security keeps a
code-literate model on every tier, while monitoring starts on the budget model
and escalates only when a case gets hard.

Every router classifies each request (`classification_mode: every_request`) with
`session_affinity` off, so the tier is chosen per request — including the
tool-result continuations of one turn, which is what lets a turn that turns out
harder escalate mid-loop. `user_turn` is the opposite choice: it replays the
turn's first decision and never sees the later tool calls.

Media tiers are filled from LiteLLM's registration metadata (`model_info.mode`),
never from a model name, and their entries carry no `reasoning_effort`. A text
chain never falls back to a generator and a media tier never falls back to a chat
model; non-chat deployments are also ignored when capping the advertised window,
because they never receive the routed chat request.

Each tier is then padded with every same-score model (`assign.grow`): a candidate joins only
when one of its rungs reaches the tier's weakest entry inside the same band, fits
the tier's modality, and keeps the tier's declared input window intact. Width is
never a configured number: the roster decides how many interchangeable fallbacks
a tier gets.

A **free deployment leads its model group only after a probe proves it is free**.
The one-token assignment probe reads the response's `usage` object:
`usage.cost == 0` (and zero `cost_details` when present) is the proof. A positive cost vetoes
the group even when a `:free` marker or tag says otherwise, because SpendLogs
shows marker-based pricing can still bill depending on provider configuration.
A marker alone, or a missing/unknown cost field, is not treated as free. A
single free call only proves one deployment, so it does not make a
multi-deployment group free; the group must already be marked free or have only
that one live route. This is ordering only: membership and rung placement are
settled first. SpendLogs also shows `usage_object.cost` is more reliable than
the `spend` column (the latter can be zero while the former is positive), and
OpenAI paid answers may omit cost entirely.

### Guest models

Some deployments are never discovered by a catalogue sync: a free upstream
router, a decision endpoint, a preview alias. They are declared in
`auto-pool-guests.json` under the pool home and registered by `--register`
(which previews unless `--apply` is also given):

```json
{"models": [{"model_name": "jev",
             "litellm_params": {"model": "openrouter/typesafe/jev-router",
                                "api_base": "https://openrouter.ai/api/v1",
                                "litellm_credential_name": "OpenRouter"},
             "model_info": {"mode": "chat", "max_input_tokens": 1000000,
                            "is_free": true}}]}
```

A guest registers once and is then left alone. One that registers but cannot
answer a one-token probe is deleted again, so a stale declaration never leaves a
dead model group behind. A **decision** guest (`"mode": "decision"`) is probed on
its decision endpoint instead: a chat probe cannot reach it, and a shape error
(400/422) is proof the route exists while an auth or alias error (401/403/404)
is proof the registration is wrong. The name is the operator's choice — what
identifies a decision model is the endpoint it exposes, not what it is called.

```json
{"model_name": "my-decision",
 "litellm_params": {"model": "openai/some-alias", "api_base": "https://…/v1",
                    "litellm_credential_name": "shared"},
 "model_info": {"mode": "decision"}}
```

### Live fallbacks

`--live-fallbacks --apply` runs the fallback sanitizer in this package
(`live.py`): dead targets are stripped from every chain and the graph ends on a
live model, with `--prices` and `--seed-file` as its inputs. It is a separate
run from tier planning because the two write different things (router settings
vs. router tiers), and keeping them apart means one failing never half-applies
the other.

A **decision model** is a different kind of endpoint, not a cheaper chat model: it
takes a state plus a schema of typed questions and returns calibrated
probabilities in one forward pass, so it can never fill a routing tier. Such a
group is recognised from registration metadata — a `mode` of `decision` or
`evaluation`, a `/v1/decisions` or `/v1/systemone` endpoint, or a description that
says so — and is listed separately in the run summary rather than counted as a
broken chat candidate.

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
`--micro` stays as an override, but no list is required: the light tiers are
filled from the cheapest candidates the run already proved usable.

Candidate ids come from LiteLLM's `/model/info`, which is the routing truth: a
model the proxy lists but LiteLLM does not register cannot serve a request. The
managed routers and `__`-prefixed advisory groups are skipped either way. The
catalogue proxy is an explicit opt-in for setups where its `/v1/models` is
authoritative; the run report names the source in `catalogue` (`litellm` or
`cliproxy`).

Nothing about a particular deployment is hardcoded: model families, the
light-tier roster (derived from the pool's own cheapest usable candidates) and
every path come from the catalogue or the caller. The
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
