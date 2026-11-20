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


if __name__ == "__main__":
    unittest.main()
