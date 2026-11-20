import os
import unittest
from pathlib import Path
from unittest import mock

from auto_router_pool import cli


class FakeClient:
    def __init__(self, ids):
        self.ids = ids

    def get(self, path):
        return {"data": [{"id": name} for name in self.ids]}


IDS = [
    "brand-2-sol", "brand-2-astra", "brand-1.5-terra", "brand-luna",
    "vendor/brand-2-sol", "other-model",
]


class DiscoverTest(unittest.TestCase):
    def test_no_filter_returns_every_direct_id(self):
        found = cli.discover(FakeClient(IDS))
        self.assertNotIn("vendor/brand-2-sol", found)
        self.assertEqual(len(found), 5)

    def test_prefix(self):
        self.assertEqual(
            cli.discover(FakeClient(IDS), "brand-2"),
            ["brand-2-astra", "brand-2-sol"],
        )

    def test_suffixes(self):
        self.assertEqual(
            cli.discover(FakeClient(IDS), suffixes=("sol", "astra")),
            ["brand-2-astra", "brand-2-sol"],
        )

    def test_prefix_and_suffixes_intersect(self):
        self.assertEqual(
            cli.discover(FakeClient(IDS), "brand-2", ("astra",)),
            ["brand-2-astra"],
        )


def info_row(name, blocked=False):
    return {"model_name": name, "model_info": {"blocked": blocked}}


class CatalogueTest(unittest.TestCase):
    ROWS = [
        info_row("brand-2-sol"), info_row("brand-2-astra"),
        info_row("auto"), info_row("auto-strix"), info_row("__advisor_fable"),
        info_row("vendor/brand-2-sol"),
    ]

    def test_rows_are_a_catalogue_without_a_proxy(self):
        self.assertEqual(
            cli.catalogue(self.ROWS), ["brand-2-astra", "brand-2-sol"]
        )

    def test_routers_and_internal_groups_are_excluded(self):
        found = cli.catalogue([info_row("auto"), info_row("auto-classify"),
                               info_row("__advisor_fable"), info_row("model-a")])
        self.assertEqual(found, ["model-a"])

    def test_router_rows_are_not_candidates(self):
        advisor = {"model_name": "auto-advisor", "model_info": {},
                   "litellm_params": {"complexity_router_config": {"tiers": {}}}}
        self.assertEqual(cli.catalogue([advisor, info_row("model-a")]), ["model-a"])

    def test_filters_apply_to_both_sources(self):
        self.assertEqual(cli.catalogue(self.ROWS, "brand-2", ("astra",)),
                         ["brand-2-astra"])
        self.assertEqual(cli.discover(FakeClient(["auto", "brand-2-sol"]), suffixes=("sol",)),
                         ["brand-2-sol"])


class SettingsTest(unittest.TestCase):
    def test_home_honours_environment(self):
        with mock.patch.dict(os.environ, {"AUTO_POOL_HOME": "/tmp/arp"}):
            self.assertEqual(cli.default_home(), Path("/tmp/arp"))

    def test_csv_ignores_blank_entries(self):
        self.assertEqual(cli.csv(" a, ,b "), ("a", "b"))
        self.assertEqual(cli.csv(""), ())
        self.assertEqual(cli.csv(None), ())

    def test_missing_credentials_exit_nonzero(self):
        args = cli.parse_args(["--home", "/tmp/arp-test", "--env-file", "/dev/null"])
        with mock.patch.dict(os.environ, {"LITELLM_MASTER_KEY": "", "CPAPI_API_KEY": ""}):
            self.assertEqual(cli.run(args), 1)

    def test_catalogue_proxy_key_is_optional(self):
        settings = cli.Settings(key="sk-litellm")
        self.assertEqual(cli.required_env(settings), [])
        self.assertEqual(cli.required_env(cli.Settings(key="")), ["LITELLM_MASTER_KEY"])


if __name__ == "__main__":
    unittest.main()
