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
