import unittest

from auto_router_pool import pool

VARIANTS = {
    "max": {"index": 57.3, "in": 2, "out": 10},
    "high": {"index": 54.9, "in": 2, "out": 10},
    "medium": {"index": 52.7, "in": 2, "out": 10},
}

class ScoreTest(unittest.TestCase):
    def test_missing_effort_falls_back_to_max(self):
        index, value, entry = pool.score(VARIANTS, "xhigh")
        self.assertEqual(index, 57.3)
        self.assertAlmostEqual(value, 57.3 / 6)

    def test_unknown_model_scores_none(self):
        self.assertIsNone(pool.score({}, "high"))


class DeploymentFlakinessTest(unittest.TestCase):
    def test_only_mixed_samples_are_flaky(self):
        samples = {
            "mixed": ["ok", "rate_limited", "ok"],
            "down": ["rate_limited", "rate_limited", "rate_limited"],
            "healthy": ["ok", "ok", "ok"],
        }
        self.assertEqual(pool.flaky_deployments(samples), ["mixed"])

    def test_stable_requires_every_sample_to_answer(self):
        samples = {
            "mixed": ["ok", "rate_limited", "ok"],
            "healthy": ["ok", "ok", "ok"],
            "unknown": ["unknown", "unknown", "unknown"],
        }
        self.assertEqual(pool.stable_deployments(samples), {"healthy"})


class FingerprintTest(unittest.TestCase):
    def rows(self, blocked):
        return [
            {"model_name": "model-primary", "model_info": {"blocked": blocked}},
            {"model_name": "model-budget", "model_info": {"blocked": False}},
        ]

    def test_blocked_flip_changes_fingerprint(self):
        self.assertNotEqual(
            pool.fingerprint(self.rows(False), ["model-primary"]),
            pool.fingerprint(self.rows(True), ["model-primary"]),
        )

    def test_catalogue_change_changes_fingerprint(self):
        self.assertNotEqual(
            pool.fingerprint(self.rows(False), ["model-primary"]),
            pool.fingerprint(self.rows(False), ["model-primary", "model-premium"]),
        )


class ScoreHashTest(unittest.TestCase):
    def test_fetch_timestamp_does_not_count_as_a_score_change(self):
        first = {"brand-2-sol": {"fetched": 1, "variants": {"max": {"index": 1, "in": 1, "out": 1}}}}
        second = {"brand-2-sol": {"fetched": 2, "variants": {"max": {"index": 1, "in": 1, "out": 1}}}}
        self.assertEqual(pool.score_hashes(first), pool.score_hashes(second))
        self.assertEqual(
            pool.changed_scores(pool.score_hashes(second), pool.score_hashes(first)), []
        )

    def test_number_change_is_reported(self):
        first = {"brand-2-sol": {"fetched": 1, "variants": {"max": {"index": 1, "in": 1, "out": 1}}}}
        second = {"brand-2-sol": {"fetched": 1, "variants": {"max": {"index": 2, "in": 1, "out": 1}}}}
        self.assertEqual(
            pool.changed_scores(pool.score_hashes(second), pool.score_hashes(first)),
            ["brand-2-sol"],
        )


class GroupStateTest(unittest.TestCase):
    def test_all_blocked(self):
        rows = [{"model_name": "m", "model_info": {"blocked": True}}]
        self.assertEqual(pool.group_state(rows, "m"), (False, "all_blocked"))

    def test_one_unblocked_deployment_is_enough(self):
        rows = [
            {"model_name": "m", "model_info": {"blocked": True, "max_input_tokens": 400000}},
            {"model_name": "m", "model_info": {"blocked": False, "max_input_tokens": 400000}},
        ]
        self.assertTrue(pool.group_state(rows, "m")[0])

    def test_unregistered(self):
        self.assertEqual(pool.group_state([], "m"), (False, "unregistered"))


class ContextFloorTest(unittest.TestCase):
    def rows(self, *windows):
        return [
            {"model_name": "m", "model_info": {"blocked": False, "max_input_tokens": window}}
            for window in windows
        ]

    def test_window_uses_largest_deployment(self):
        self.assertEqual(pool.input_window(self.rows(131072, 400000), "m"), 400000)

    def test_undeclared_window_is_zero(self):
        self.assertEqual(pool.input_window([{"model_name": "m", "model_info": {}}], "m"), 0)

    def test_smaller_than_floor_is_unusable(self):
        self.assertEqual(pool.group_state(self.rows(131072), "m"), (False, "context_below_floor"))

    def test_floor_is_inclusive(self):
        self.assertTrue(pool.group_state(self.rows(272000), "m")[0])

    def test_custom_floor(self):
        self.assertEqual(
            pool.group_state(self.rows(400000), "m", minimum=500000),
            (False, "context_below_floor"),
        )


class FreeModelsTest(unittest.TestCase):
    def row(self, name, upstream, info=None):
        return {
            "model_name": name,
            "litellm_params": {"model": upstream},
            "model_info": info or {},
        }

    def test_a_free_tier_is_recognised_by_its_upstream_id(self):
        rows = [
            self.row("free-model", "openrouter/google/gemma-4-31b-it:free"),
            self.row("paid-model", "openai/gpt-6.2-sol"),
        ]
        self.assertEqual(pool.free_models(rows), {"free-model"})

    def test_a_free_tier_with_configured_costs_is_still_free(self):
        # A free tier may still carry a price, so cost must not decide this.
        rows = [self.row(
            "free-model", "openrouter/x/y:free",
            {"input_cost_per_token": 2e-06, "output_cost_per_token": 1e-05},
        )]
        self.assertEqual(pool.free_models(rows), {"free-model"})

    def test_an_unpriced_group_is_not_free(self):
        # Costs left unset declare zero (or nothing); that is not a free model.
        rows = [
            self.row("zero", "openai/paid-model",
                     {"input_cost_per_token": 0, "output_cost_per_token": 0}),
            self.row("unset", "openai/other-model",
                     {"input_cost_per_token": None, "output_cost_per_token": None}),
            self.row("absent", "openai/third-model"),
        ]
        self.assertEqual(pool.free_models(rows), set())

    def test_one_paid_deployment_makes_the_group_paid(self):
        # The router may reach any deployment of the group, so a mixed group is
        # not free and must not jump the queue.
        rows = [
            self.row("mixed", "openai/gpt-6-luna:free"),
            self.row("mixed", "openai/devin/gpt-6-luna"),
        ]
        self.assertEqual(pool.free_models(rows), set())

    def test_an_operator_tag_can_mark_a_group_free(self):
        rows = [self.row("tagged", "openai/custom", {"tags": ["free"]})]
        self.assertEqual(pool.free_models(rows), {"tagged"})
        self.assertEqual(
            pool.free_models([self.row("flagged", "openai/custom", {"is_free": True})]),
            {"flagged"},
        )


class DecisionModelTest(unittest.TestCase):
    def row(self, name, info):
        return {"model_name": name, "model_info": info}

    def test_a_decision_mode_is_recognised(self):
        rows = [self.row("dec", {"mode": "decision"}), self.row("chatty", {"mode": "chat"})]
        self.assertEqual(pool.decision_models(rows), ["dec"])

    def test_an_evaluation_mode_is_recognised(self):
        # The price catalogue files the jev variants under 'evaluation'.
        self.assertEqual(pool.decision_models([self.row("judge", {"mode": "evaluation"})]),
                         ["judge"])

    def test_a_decision_endpoint_is_recognised(self):
        rows = [self.row("clef", {"mode": "chat",
                                   "supported_endpoints": ["/v1/systemone"]})]
        self.assertEqual(pool.decision_models(rows), ["clef"])

    def test_a_described_decision_model_is_recognised(self):
        rows = [self.row("clef-flash", {"mode": "chat", "description":
                          "a 9B decision model returning probabilities; not a chat model"})]
        self.assertEqual(pool.decision_models(rows), ["clef-flash"])

    def test_a_chat_model_with_a_decisions_endpoint_is_not_a_decision_model(self):
        # gpt-6-luna lists /v1/chat/completions and /v1/decisions: it is a chat
        # model that can also decide, not a decision-only endpoint.
        rows = [self.row("luna", {"mode": "chat", "supported_endpoints": [
            "/v1/chat/completions", "/v1/batch", "/v1/decisions"]})]
        self.assertEqual(pool.decision_models(rows), [])

    def test_a_plain_chat_model_is_not_a_decision_model(self):
        rows = [self.row("gpt", {"mode": "chat", "description": "general chat model"})]
        self.assertEqual(pool.decision_models(rows), [])


class FreeGapsTest(unittest.TestCase):
    def row(self, name, upstream, blocked=False, window=400000, info=None):
        return {
            "model_name": name,
            "litellm_params": {"model": upstream},
            "model_info": dict({"blocked": blocked, "max_input_tokens": window}, **(info or {})),
        }

    def test_a_blocked_free_group_is_reported_with_its_reason(self):
        rows = [self.row("free-blocked", "openrouter/x/y:free", blocked=True)]
        self.assertEqual(
            pool.free_gaps(rows, {}),
            [{"model": "free-blocked", "reason": "all_blocked"}],
        )

    def test_a_usable_free_group_without_a_score_is_unmeasured(self):
        rows = [self.row("free-unrated", "openrouter/x/y:free")]
        self.assertEqual(
            pool.free_gaps(rows, {}),
            [{"model": "free-unrated", "reason": "unmeasured"}],
        )

    def test_a_small_context_free_group_is_reported_below_the_floor(self):
        rows = [self.row("free-small", "openrouter/x/y:free", window=65536)]
        self.assertEqual(
            pool.free_gaps(rows, {}),
            [{"model": "free-small", "reason": "context_below_floor"}],
        )

    def test_a_scored_free_group_is_not_a_gap(self):
        rows = [self.row("free-scored", "openrouter/x/y:free")]
        self.assertEqual(pool.free_gaps(rows, {"free-scored": {}}), [])

    def test_a_paid_group_is_never_reported(self):
        self.assertEqual(pool.free_gaps([self.row("paid", "openai/paid")], {}), [])


class PickMicroTest(unittest.TestCase):
    def test_taken_roles_are_excluded(self):
        candidates = {
            "primary": {"max": {"index": 50.0, "in": 1, "out": 2}},
            "cheap": {"max": {"index": 20.0, "in": 0.1, "out": 0.2}},
        }
        self.assertEqual(pool.pick_micro(candidates, ["primary"]), ["cheap"])

    def test_cheapest_two_win(self):
        candidates = {
            "cheap": {"max": {"index": 20.0, "in": 0.10, "out": 0.20}},
            "cheaper": {"max": {"index": 15.0, "in": 0.05, "out": 0.10}},
            "dear": {"max": {"index": 60.0, "in": 5.0, "out": 20.0}},
            "free": {"max": {"index": 25.0, "in": 0, "out": 0}},
        }
        self.assertEqual(pool.pick_micro(candidates, ["dear"]), ["cheaper", "cheap"])

    def test_price_tie_prefers_higher_index(self):
        candidates = {
            "low": {"max": {"index": 10.0, "in": 1, "out": 1}},
            "high": {"max": {"index": 30.0, "in": 1, "out": 1}},
        }
        self.assertEqual(pool.pick_micro(candidates), ["high", "low"])

    def test_models_without_a_priced_rung_are_skipped(self):
        self.assertEqual(pool.pick_micro({"ghost": {}, "empty": {"low": {}}}), [])


if __name__ == "__main__":
    unittest.main()


class DefaultModelTest(unittest.TestCase):
    """The default path has no tier config, so it runs at the model's own effort."""

    SHARP_AT_MAX = {
        "max": {"index": 60.0, "in": 1, "out": 1},
        "low": {"index": 30.0, "in": 1, "out": 1},
    }
    STRONG_ANYWHERE = {
        "max": {"index": 55.0, "in": 1, "out": 1},
        "low": {"index": 45.0, "in": 1, "out": 1},
    }

    def test_a_declared_default_effort_is_scored_at_that_rung(self):
        rows = [{"model_name": "a", "litellm_params": {"reasoning_effort": "max"}}]
        picked, note = pool.pick_default(
            rows, {"a": self.SHARP_AT_MAX, "b": self.STRONG_ANYWHERE}, ["a", "b"]
        )
        self.assertEqual((picked, note["basis"], note["effort"]), ("a", "declared", "max"))
        self.assertEqual(note["index"], 60.0)

    def test_without_a_declared_default_the_floor_decides(self):
        picked, note = pool.pick_default(
            [], {"a": self.SHARP_AT_MAX, "b": self.STRONG_ANYWHERE}, ["a", "b"]
        )
        self.assertEqual((picked, note["basis"], note["effort"]), ("b", "floor", None))
        self.assertEqual(note["index"], 45.0)

    def test_siblings_that_disagree_declare_nothing(self):
        rows = [
            {"model_name": "a", "litellm_params": {"reasoning_effort": "high"}},
            {"model_name": "a", "litellm_params": {"reasoning_effort": "low"}},
        ]
        self.assertIsNone(pool.declared_effort(rows, "a"))

    def test_a_tie_keeps_the_pool_preference(self):
        same = {"max": {"index": 50.0, "in": 1, "out": 1}}
        picked, _ = pool.pick_default([], {"a": same, "b": same}, ["a", "b"])
        self.assertEqual(picked, "a")

    def test_an_unmeasured_model_never_beats_a_measured_one(self):
        picked, _ = pool.pick_default([], {"a": {}, "b": self.STRONG_ANYWHERE}, ["a", "b"])
        self.assertEqual(picked, "b")


class DecisionClassifierTest(unittest.TestCase):
    def rows(self, blocked):
        return [{"model_name": "clef", "litellm_params": {},
                 "model_info": {"id": "1f0b6c33-9c1e-4f57-8a2c-6d2b9e7c4a55", "mode": "decision", "blocked": blocked}}]

    def test_a_usable_decision_model_becomes_the_classifier(self):
        self.assertEqual(pool.decision_classifier(self.rows(False)), "clef")

    def test_a_blocked_decision_model_is_not_usable(self):
        self.assertIsNone(pool.decision_classifier(self.rows(True)))

    def test_a_chat_model_is_never_a_decision_classifier(self):
        rows = [{"model_name": "luna", "litellm_params": {},
                 "model_info": {"mode": "chat", "blocked": False}}]
        self.assertIsNone(pool.decision_classifier(rows))


class DecisionCapabilityTest(unittest.TestCase):
    """Capability (can it answer a decision?) is not exclusivity (is it only a decider?)."""

    def test_a_chat_model_that_serves_decisions_is_capable(self):
        row = {"model_name": "luna", "litellm_params": {},
               "model_info": {"mode": "chat", "blocked": False,
                              "supported_endpoints": ["/v1/chat/completions", "/v1/decisions"]}}
        self.assertTrue(pool.supports_decisions(row))
        self.assertFalse(pool.is_decision(row))

    def test_capability_is_used_for_the_classifier_even_when_not_exclusive(self):
        rows = [{"model_name": "luna", "litellm_params": {},
                 "model_info": {"mode": "chat", "blocked": False,
                                "supported_endpoints": ["/v1/chat/completions", "/v1/decisions"]}}]
        self.assertEqual(pool.decision_classifier(rows), "luna")

    def test_a_decision_only_model_is_preferred_over_a_capable_chat_model(self):
        rows = [
            {"model_name": "luna", "litellm_params": {},
             "model_info": {"mode": "chat", "blocked": False,
                            "supported_endpoints": ["/v1/chat/completions", "/v1/decisions"]}},
            {"model_name": "clef", "litellm_params": {},
             "model_info": {"id": "c1", "mode": "decisions", "blocked": False}},
        ]
        self.assertEqual(pool.decision_classifier(rows), "clef")

    def test_a_plain_chat_model_stays_ineligible(self):
        row = {"model_name": "sol", "litellm_params": {},
               "model_info": {"mode": "chat", "blocked": False,
                              "supported_endpoints": ["/v1/chat/completions"]}}
        self.assertFalse(pool.supports_decisions(row))


class ClassifierCandidateTest(unittest.TestCase):
    def rows(self):
        return [
            {"model_name": "clef", "litellm_params": {},
             "model_info": {"id": "5c9a1d20-8e7b-4c3a-9f10-3a2b6d4e8c77", "mode": "decisions", "blocked": True}},
            {"model_name": "luna", "litellm_params": {},
             "model_info": {"mode": "chat", "blocked": False,
                            "supported_endpoints": ["/v1/chat/completions", "/v1/decisions"]}},
            {"model_name": "sol", "litellm_params": {},
             "model_info": {"mode": "chat", "blocked": False,
                            "supported_endpoints": ["/v1/chat/completions"]}},
        ]

    def test_candidates_report_capability_and_usability(self):
        self.assertEqual(
            pool.classifier_candidates(self.rows()),
            [{"model": "clef", "exclusive": True, "usable": False},
             {"model": "luna", "exclusive": False, "usable": True}],
        )

    def test_the_unblocked_capable_model_wins(self):
        self.assertEqual(pool.decision_classifier(self.rows()), "luna")

    def test_a_measured_index_is_reported(self):
        entries = pool.classifier_candidates(self.rows(), scores={"luna": 66.0})
        self.assertEqual(entries[1]["decision_index"], 66.0)
        self.assertNotIn("decision_index", entries[0])


class SpecializedTest(unittest.TestCase):
    def row(self, name, mode, cost=None, blocked=False, upstream=None, outputs=None):
        params = {"model": upstream or name}
        if cost is not None:
            params["input_cost_per_token"] = cost
        info = {"mode": mode, "blocked": blocked}
        if outputs:
            info["output_modalities"] = outputs
        return {"model_name": name, "litellm_params": params, "model_info": info}

    def test_score_per_original_price_orders_a_family(self):
        rows = [
            self.row("quality", "image_generation", cost=4e-6),
            self.row("value", "image_generation", cost=1e-6),
        ]
        found = pool.specialized_models(
            rows, scores={"image": {"quality": 100.0, "value": 50.0}}
        )
        self.assertEqual(found["image"], ["value", "quality"])

    def test_a_measured_model_beats_an_unmeasured_one(self):
        rows = [
            self.row("measured", "video_generation", cost=2e-6),
            self.row("unmeasured", "video_generation", cost=1e-7),
        ]
        found = pool.specialized_models(rows, scores={"video": {"measured": 1000.0}})
        self.assertEqual(found["video"], ["measured", "unmeasured"])

    def test_price_map_mode_finds_audio(self):
        rows = [self.row("asr", "chat", cost=2.5e-6,
                         upstream="gpt-4o-transcribe")]
        catalog = {"gpt-4o-transcribe": {
            "mode": "audio_transcription", "input_cost_per_token": 2.5e-6,
        }}
        found = pool.specialized_models(rows, catalog, {"audio": {"asr": 96.0}})
        self.assertEqual(found["audio"], ["asr"])

    def test_a_blocked_family_is_not_listed(self):
        rows = [self.row("img", "image_generation", blocked=True)]
        self.assertEqual(pool.specialized_models(rows)["image"], [])

    def test_all_kinds_are_reported(self):
        self.assertEqual(
            set(pool.specialized_models([])), {"image", "video", "audio", "ocr"}
        )

    def test_a_batch_model_is_not_a_chat_candidate(self):
        rows = [{"model_name": "batch", "litellm_params": {}, "model_info": {
            "mode": "batch", "max_input_tokens": 400000,
        }}]
        self.assertEqual(pool.group_state(rows, "batch"), (False, "batch_model"))
        self.assertEqual(pool.batch_models(rows), ["batch"])

    def test_a_batch_name_wins_over_a_stale_chat_mode(self):
        rows = [{"model_name": "gpt-5.4-batch", "litellm_params": {
            "model": "openai/gpt-5.4",
        }, "model_info": {"mode": "chat", "max_input_tokens": 400000}}]
        self.assertEqual(
            pool.group_state(rows, "gpt-5.4-batch"), (False, "batch_model")
        )
        self.assertEqual(pool.batch_models(rows), ["gpt-5.4-batch"])


class DecisionCatalogTest(unittest.TestCase):
    CATALOG = {
        "typesafe/jev-latest": {"mode": "evaluation", "max_input_tokens": 32000},
        "openrouter/typesafe/jev-router": {"mode": "chat"},
        "cloudflare/clef": {"mode": "decisions", "max_input_tokens": 131072},
        "openai/gpt-6.1-sol": {"mode": "chat", "max_input_tokens": 1000000},
    }

    def test_the_catalogue_lists_what_can_classify(self):
        self.assertEqual(
            [entry["model"] for entry in pool.decision_catalog(self.CATALOG)],
            ["cloudflare/clef", "typesafe/jev-latest"],
        )

    def test_a_catalogue_entry_is_matched_by_upstream_name(self):
        row = {"model_name": "jev", "litellm_params": {"model": "typesafe/jev-latest"},
               "model_info": {"mode": "chat"}}
        self.assertTrue(pool.supports_decisions(row, self.CATALOG))
        chat_row = {"model_name": "jev", "litellm_params": {"model": "openrouter/typesafe/jev-router"},
                    "model_info": {"mode": "chat", "supported_endpoints": ["/v1/decisions"]}}
        self.assertFalse(pool.supports_decisions(chat_row, self.CATALOG))


class DecisionCandidateOrderTest(unittest.TestCase):
    def rows(self):
        return [
            {"model_name": "luna", "litellm_params": {},
             "model_info": {"mode": "chat", "blocked": False,
                            "supported_endpoints": ["/v1/decisions"]}},
            {"model_name": "clef", "litellm_params": {"model": "openai/clef"},
             "model_info": {"id": "834d2f18-cd52-4a4a-9c4e-c6af91ee5a4a", "mode": "decision", "blocked": False,
                            "description": "decision model; not a chat"}},
            {"model_name": "jev", "litellm_params": {"model": "openrouter/typesafe/jev-1.13"},
             "model_info": {"id": "b08f4d11-1869-440d-ae44-72f94902e4f9", "mode": "decision", "blocked": False}},
        ]

    def test_decision_only_models_come_first_then_capable_chat_models(self):
        self.assertEqual(pool.decision_candidates(self.rows()), ["clef", "jev", "luna"])

    def test_a_blocked_group_is_not_a_candidate(self):
        rows = self.rows()
        rows[1]["model_info"]["blocked"] = True
        self.assertEqual(pool.decision_candidates(rows), ["jev", "luna"])


class DecisionPriorityTest(unittest.TestCase):
    def row(self, name, cost=None, free=False, upstream=None):
        params = {"model": upstream or name}
        if cost is not None:
            params["input_cost_per_token"] = cost
            params["output_cost_per_token"] = 0.0
        info = {"id": name, "mode": "decisions", "blocked": False}
        if free:
            info["is_free"] = True
        return {"model_name": name, "litellm_params": params, "model_info": info}

    def test_a_measured_score_beats_a_cheaper_unmeasured_model(self):
        rows = [self.row("cheap", cost=1e-7), self.row("strong", cost=2e-6)]
        self.assertEqual(
            pool.decision_candidates(rows, scores={"strong": 66.0}),
            ["strong", "cheap"],
        )

    def test_price_breaks_ties_between_unmeasured_models(self):
        rows = [self.row("expensive", cost=2e-7), self.row("cheap", cost=1e-7)]
        self.assertEqual(pool.decision_candidates(rows), ["cheap", "expensive"])

    def test_a_free_tier_is_compared_at_its_base_price_and_wins_a_tie(self):
        rows = [self.row("m", cost=1e-7), self.row("m:free", cost=0.0, free=True)]
        self.assertEqual(
            pool.decision_candidates(rows, scores={"m": 66.0, "m:free": 66.0}),
            ["m:free", "m"],
        )

    def test_a_free_tier_without_a_base_price_is_not_preferred(self):
        rows = [
            self.row("paid", cost=1e-7),
            self.row("ghost:free", cost=0.0, free=True),
        ]
        self.assertEqual(pool.decision_candidates(rows), ["paid", "ghost:free"])

    def test_the_litellm_price_map_prices_an_unpriced_model(self):
        rows = [
            self.row("unpriced", upstream="typesafe/jev-1.13"),
            self.row("priced", cost=2e-7),
        ]
        catalog = {"typesafe/jev-1.13": {"input_cost_per_token": 1e-7}}
        self.assertEqual(
            pool.decision_candidates(rows, catalog), ["unpriced", "priced"]
        )
