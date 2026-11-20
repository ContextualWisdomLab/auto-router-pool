import unittest

from auto_router_pool import pool

VARIANTS = {
    "max": {"index": 57.3, "in": 2, "out": 10},
    "high": {"index": 54.9, "in": 2, "out": 10},
    "medium": {"index": 52.7, "in": 2, "out": 10},
}

CANDIDATES = {
    "model-primary": VARIANTS,
    "model-premium": {
        "xhigh": {"index": 61.6, "in": 8, "out": 40},
        "high": {"index": 60.1, "in": 8, "out": 40},
    },
    "model-budget": {"medium": {"index": 18.5, "in": 0.1, "out": 0.5}},
}


class ScoreTest(unittest.TestCase):
    def test_missing_effort_falls_back_to_max(self):
        index, value, entry = pool.score(VARIANTS, "xhigh")
        self.assertEqual(index, 57.3)
        self.assertAlmostEqual(value, 57.3 / 6)

    def test_unknown_model_scores_none(self):
        self.assertIsNone(pool.score({}, "high"))


class PickTest(unittest.TestCase):
    def test_quality_prefers_index_but_breaks_ties_on_price(self):
        chosen = pool.pick(CANDIDATES, "high", 12, 45)
        self.assertEqual(chosen[0], "model-premium")

    def test_quality_honours_price_cap(self):
        chosen = pool.pick(CANDIDATES, "high", 3, 12)
        self.assertEqual(chosen[0], "model-primary")

    def test_value_prefers_index_per_dollar(self):
        chosen = pool.pick(CANDIDATES, "medium", 0.3, 1.5, mode="value")
        self.assertEqual(chosen[0], "model-budget")

    def test_no_candidate_under_cap(self):
        self.assertIsNone(pool.pick(CANDIDATES, "high", 1, 1))


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


if __name__ == "__main__":
    unittest.main()
