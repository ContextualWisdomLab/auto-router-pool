import os
import unittest
from pathlib import Path
from unittest import mock

from auto_router_pool import cli, client


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


class SelectedModelsTest(unittest.TestCase):
    def test_collects_models_from_every_managed_router(self):
        rows = [
            {"model_name": "auto", "model_info": {},
             "litellm_params": {"complexity_router_config": {
                 "tiers": {"SIMPLE": ["budget"], "COMPLEX": ["primary", "premium"]}}}},
            {"model_name": "auto-strix", "model_info": {},
             "litellm_params": {"complexity_router_config": {
                 "tiers": {"MEDIUM": ["primary"]}}}},
            info_row("plain-model"),
        ]
        self.assertEqual(cli.selected_models(rows), ["budget", "premium", "primary"])


class StubClient:
    def __init__(self, result):
        self.result = result

    def post(self, *args, **kwargs):
        return self.result


class ProbeTest(unittest.TestCase):
    def test_cooldown_is_unavailable(self):
        self.assertEqual(client.probe(StubClient({"error_status": 429}), "gpt-6.2-sol"),
                         (False, "unavailable_429"))

    def test_rejected_model_is_unavailable(self):
        self.assertEqual(client.probe(StubClient({"error_status": 400}), "m"),
                         (False, "unavailable_400"))

    def test_fallback_answer_is_not_availability(self):
        self.assertEqual(
            client.probe(StubClient({"model": "gemini-3.8-flash"}), "gpt-6.1-sol"),
            (False, "fallback_gemini-3.8-flash"),
        )

    def test_model_answering_itself_is_ok(self):
        self.assertEqual(client.probe(StubClient({"model": "grok-4.7"}), "grok-4.7"),
                         (True, "ok"))

    def test_timeout_is_a_status_not_a_crash(self):
        with mock.patch("urllib.request.urlopen", side_effect=TimeoutError()):
            result = client.Client("http://127.0.0.1:4000", "k").get("/model/info")
        self.assertEqual(result["error_status"], "unreachable")


class DeploymentClient:
    def __init__(self, health_states, block_result=None):
        self.health_states = {key: list(value) for key, value in health_states.items()}
        self.block_result = block_result or {"blocked": True}
        self.blocks = []
        self.calls = {}

    def get(self, path, timeout=None):
        model_id = path.split("model_id=", 1)[1]
        index = self.calls.get(model_id, 0)
        self.calls[model_id] = index + 1
        state = self.health_states[model_id][index]
        if state == "rate_limited":
            # LiteLLM scopes by ?model_id= and leaves the field out of endpoints.
            return {"error_status": 429, "error_body": {"unhealthy_endpoints": [
                {"error": "429 rate limit exceeded"}
            ]}}
        if state == "ok":
            return {"healthy_endpoints": [{"model_id": model_id}]}
        if state == "unhealthy":
            return {"healthy_endpoints": [], "unhealthy_endpoints": [
                {"error": "503 high demand"}
            ]}
        return {"error_status": 500}

    def post(self, path, body, timeout=None):
        self.blocks.append((path, body))
        return self.block_result


def deployment_row(name, model_id, blocked=False):
    return {"model_name": name, "model_info": {"id": model_id, "blocked": blocked}}


class DeploymentHealthTest(unittest.TestCase):
    def test_health_report_in_a_429_body_is_read(self):
        client_ = DeploymentClient({"m": ["rate_limited"]})
        self.assertEqual(client.deployment_health(client_, "m"), "rate_limited")

    def test_healthy_endpoint_is_ok(self):
        client_ = DeploymentClient({"m": ["ok"]})
        self.assertEqual(client.deployment_health(client_, "m"), "ok")

    def test_plain_429_without_a_report_is_rate_limited(self):
        class Plain:
            def get(self, path, timeout=None):
                return {"error_status": 429}
        self.assertEqual(client.deployment_health(Plain(), "m"), "rate_limited")

    def test_non_429_endpoint_error_is_not_flakiness(self):
        client_ = DeploymentClient({"m": ["unhealthy"]})
        self.assertEqual(client.deployment_health(client_, "m"), "unhealthy")


class TriageTest(unittest.TestCase):
    def test_blocks_only_the_intermittent_deployment(self):
        rows = [deployment_row("m", "a"), deployment_row("m", "b")]
        client_ = DeploymentClient({"a": ["ok", "rate_limited", "ok"],
                                    "b": ["ok", "ok", "ok"]})
        blocked, note = cli.triage_deployments(client_, rows, "m", apply=True)
        self.assertEqual(blocked, ["a"])
        self.assertEqual(note, "blocked")
        self.assertTrue(rows[0]["model_info"]["blocked"])
        self.assertFalse(rows[1]["model_info"]["blocked"])
        self.assertEqual(client_.blocks, [("/model/block", {"model_id": "a"})])

    def test_preview_reports_without_writing(self):
        rows = [deployment_row("m", "a"), deployment_row("m", "b")]
        client_ = DeploymentClient({"a": ["ok", "rate_limited", "ok"],
                                    "b": ["ok", "ok", "ok"]})
        blocked, note = cli.triage_deployments(client_, rows, "m", apply=False)
        self.assertEqual((blocked, note), (["a"], "would_block"))
        self.assertEqual(client_.blocks, [])
        self.assertFalse(rows[0]["model_info"]["blocked"])

    def test_single_deployment_is_never_blocked(self):
        rows = [deployment_row("m", "a")]
        client_ = DeploymentClient({"a": ["ok", "rate_limited", "ok"]})
        self.assertEqual(cli.triage_deployments(client_, rows, "m", apply=True),
                         ([], "single_deployment"))
        self.assertEqual(client_.blocks, [])

    def test_no_stable_sibling_means_a_group_outage_not_a_bad_deployment(self):
        rows = [deployment_row("m", "a"), deployment_row("m", "b")]
        client_ = DeploymentClient({"a": ["ok", "rate_limited", "ok"],
                                    "b": ["ok", "rate_limited", "ok"]})
        self.assertEqual(
            cli.triage_deployments(client_, rows, "m", apply=True),
            ([], "no_stable_sibling"),
        )
        self.assertEqual(client_.blocks, [])


if __name__ == "__main__":
    unittest.main()
