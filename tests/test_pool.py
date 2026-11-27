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
    def rows(self, *specs):
        return [{"model_name": name, "model_info": info} for name, info in specs]

    def test_only_zero_cost_deployments_are_free(self):
        rows = self.rows(
            ("free-model", {"input_cost_per_token": 0, "output_cost_per_token": 0}),
            ("paid-model", {"input_cost_per_token": 2e-06, "output_cost_per_token": 1e-05}),
        )
        self.assertEqual(pool.free_models(rows), {"free-model"})

    def test_one_paid_deployment_makes_the_group_paid(self):
        # The router may reach any deployment of the group, so a mixed group is
        # not free and must not jump the queue.
        rows = self.rows(
            ("mixed", {"input_cost_per_token": 0, "output_cost_per_token": 0}),
            ("mixed", {"input_cost_per_token": 3e-06, "output_cost_per_token": 1e-05}),
        )
        self.assertEqual(pool.free_models(rows), set())

    def test_an_unpriced_group_is_not_free(self):
        self.assertEqual(pool.free_models(self.rows(("unknown", {}))), set())


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
