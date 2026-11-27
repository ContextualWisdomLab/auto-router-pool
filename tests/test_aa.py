import json
import unittest
import urllib.error

from auto_router_pool import aa


def page(models):
    payload = json.dumps(models).replace('"', '\\"')
    return f'<html><script>self.x={{"initialModels":{payload}}}</script></html>'


MODELS = [
    {"slug": "brand-2-sol", "effort": {"slug": "max"},
     "intelligenceIndex": 57.3, "price1mInputTokens": 2, "price1mOutputTokens": 10},
    {"slug": "brand-2-sol-high", "effort": {"slug": "high"},
     "intelligenceIndex": 54.9, "price1mInputTokens": 2, "price1mOutputTokens": 10},
    {"slug": "brand-2-sol-low", "effort": {"slug": "low"},
     "intelligenceIndex": 40.6, "price1mInputTokens": 2, "price1mOutputTokens": 10},
    {"slug": "other-model", "intelligenceIndex": 99,
     "price1mInputTokens": 1, "price1mOutputTokens": 1},
]


class ParseTest(unittest.TestCase):
    def test_splits_effort_variants(self):
        variants = aa.parse_variants(page(MODELS), "brand-2-sol")
        self.assertEqual(sorted(variants), ["high", "low", "max"])
        self.assertEqual(variants["max"]["index"], 57.3)
        self.assertEqual(variants["high"]["in"], 2)

    def test_ignores_sibling_models(self):
        variants = aa.parse_variants(page(MODELS), "brand-2-so")
        self.assertEqual(variants, {})

    def test_missing_array_raises(self):
        with self.assertRaises(aa.LayoutError):
            aa.parse_variants("<html>client side only</html>", "brand-2-sol")


DECISION_HTML = (
    '<p>Compared with other decision models, Clef is among the lowest scoring.</p>'
    '<p>Clef scores 60.2 on the Artificial Analysis Decision Index, placing it low.</p>'
    '<p>At 354 ms per task, Clef is notably slow.</p>'
    '<p>On average, it costs $863 per 1M tasks to run Clef.</p>'
    '<p>Pricing for Clef is $0.24 per 1M input tokens.</p>'
)


class DecisionPageTest(unittest.TestCase):
    def test_the_decision_index_and_its_cost_are_parsed(self):
        result = aa.parse_decision_page(DECISION_HTML)
        self.assertEqual(result["score"], 60.2)
        self.assertEqual(result["cost_per_million_tasks"], 863.0)
        self.assertEqual(result["time_per_task_ms"], 354.0)
        self.assertEqual(result["input_price_per_million"], 0.24)

    def test_a_missing_score_is_a_layout_break(self):
        with self.assertRaises(aa.LayoutError):
            aa.parse_decision_page("<html>client side only</html>")

    def test_gateway_names_map_to_aa_slugs(self):
        self.assertEqual(aa.decision_slug("gpt-6-luna-decisions:free"),
                         "gpt-6-luna-decisions")
        self.assertEqual(aa.decision_slug("gpt-6-luna"), "gpt-6-luna-decisions")
        self.assertEqual(aa.decision_slug("jev-latest"), "jev-1-14")


class DecisionFetchTest(unittest.TestCase):
    def test_scores_are_cached_by_slug(self):
        calls = []

        def fetch(url):
            calls.append(url)
            return DECISION_HTML

        cache = {}
        first, errors = aa.get_decision_scores(["clef"], cache, fetch=fetch)
        second, _ = aa.get_decision_scores(["clef"], cache, fetch=fetch)
        self.assertEqual(errors, {})
        self.assertEqual(first["clef"]["score"], 60.2)
        self.assertEqual(second["clef"]["score"], 60.2)
        self.assertEqual(len(calls), 1)

    def test_a_404_is_an_unscored_model_not_a_crash(self):
        def fetch(url):
            raise urllib.error.HTTPError(url, 404, "x", {}, None)
        scores, errors = aa.get_decision_scores(["clef-flash"], {}, fetch=fetch)
        self.assertEqual(scores, {})
        self.assertEqual(errors, {"clef-flash": "no_aa_entry"})

    def test_a_negative_score_is_retried_after_a_day(self):
        cache = {"clef-flash": {"fetched": 0, "score": None,
                                "error": "no_aa_entry"}}
        scores, errors = aa.get_decision_scores(
            ["clef-flash"], cache, fetch=lambda url: DECISION_HTML
        )
        self.assertEqual(errors, {})
        self.assertEqual(scores["clef-flash"]["score"], 60.2)


class NotFoundTest(unittest.TestCase):
    def fetch(self, code):
        def call(url):
            raise urllib.error.HTTPError(url, code, "x", {}, None)
        return call

    def test_404_is_a_skip_not_a_layout_break(self):
        variants, error = aa.get_variants("brand-9-sol", {}, fetch=self.fetch(404))
        self.assertEqual((variants, error), ({}, "no_aa_entry"))

    def test_other_http_errors_stay_structural(self):
        variants, error = aa.get_variants("brand-2-sol", {}, fetch=self.fetch(503))
        self.assertEqual((variants, error), ({}, "fetch_error"))


class CacheTest(unittest.TestCase):
    def test_cached_page_is_not_refetched(self):
        cache = {}
        calls = []

        def fetch(url):
            calls.append(url)
            return page(MODELS)

        first, error = aa.get_variants("brand-2-sol", cache, fetch=fetch)
        second, _ = aa.get_variants("brand-2-sol", cache, fetch=fetch)
        self.assertIsNone(error)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first, second)

    def test_force_refetches(self):
        cache = {}
        calls = []

        def fetch(url):
            calls.append(url)
            return page(MODELS)

        aa.get_variants("brand-2-sol", cache, fetch=fetch)
        aa.get_variants("brand-2-sol", cache, force=True, fetch=fetch)
        self.assertEqual(len(calls), 2)

    def test_layout_break_is_reported_not_hidden(self):
        cache = {}
        variants, error = aa.get_variants(
            "brand-2-sol", cache, fetch=lambda url: "<html></html>"
        )
        self.assertEqual(variants, {})
        self.assertEqual(error, "missing_array")

    def test_stale_cache_survives_a_break(self):
        cache = {"brand-2-sol": {"fetched": 0, "variants": {"max": {"index": 1, "in": 1, "out": 1}}}}
        variants, error = aa.get_variants(
            "brand-2-sol", cache, fetch=lambda url: "<html></html>"
        )
        self.assertEqual(error, "missing_array")
        self.assertIn("max", variants)

    def test_model_absent_from_page(self):
        variants, error = aa.get_variants(
            "brand-9-sol", {}, fetch=lambda url: page(MODELS)
        )
        self.assertEqual(variants, {})
        self.assertEqual(error, "no_aa_entry")


if __name__ == "__main__":
    unittest.main()
