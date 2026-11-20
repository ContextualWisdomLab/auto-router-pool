import json
import unittest

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
