import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from auto_router_pool import catalog

GUEST = {"model": {"slug": "m", "status": "active", "output_modalities": ["text"],
                   "maker": "Google", "description": "chat model"},
         "providers": [{"provider": "gemini", "provider_model_id": "m", "routable": True}]}


def entry(slug, *, maker="Google", description="Chat model", status="active",
          outputs=("text",), routes=(("gemini", True),), **model):
    return {
        "model": {
            "slug": slug, "status": status, "output_modalities": list(outputs),
            "description": description, "maker": maker, **model,
        },
        "providers": [
            {"provider": provider, "provider_model_id": slug, "routable": routable}
            for provider, routable in routes
        ],
    }


def by_name(planned, name):
    for action, body in planned:
        if body["model_name"] == name:
            return action, body
    raise AssertionError(f"{name} was not planned")


class FetchTest(unittest.TestCase):
    def test_pages_are_joined_and_authenticated_models_are_separate(self):
        pages = {
            0: {"models": [{"model": {"slug": "a"}}], "total": 2},
            1: {"models": [{"model": {"slug": "b"}}], "total": 2},
        }

        def fake(url, key=None, timeout=60):
            if url.endswith("/v1/models"):
                return {"data": [{"id": "a"}, {"id": "b"}]}
            offset = int(url.rsplit("offset=", 1)[1])
            return pages[offset]

        with mock.patch.object(catalog, "_get_json", side_effect=fake):
            ids, public = catalog.fetch()
        self.assertEqual(ids, {"a", "b"})
        self.assertEqual(sorted(public), ["a", "b"])


class PlanTest(unittest.TestCase):
    def test_a_model_on_another_provider_is_still_registered(self):
        # gemini-4-neon is routed by gemini/vertex, not experiential_cloud; the
        # retired sync required that provider, so new models never appeared.
        planned = catalog.plan(
            {"gemini-4-neon"}, {"gemini-4-neon": entry("gemini-4-neon")}, []
        )
        action, body = by_name(planned, "gemini-4-neon")
        self.assertEqual(action, "create")
        self.assertEqual(body["litellm_params"]["model"], "openai/gemini-4-neon")
        self.assertEqual(body["litellm_params"]["litellm_credential_name"], catalog.CREDENTIAL)
        self.assertEqual(body["model_info"]["mode"], "chat")

    def test_an_anthropic_model_uses_the_anthropic_transport(self):
        planned = catalog.plan(
            {"claude-sonnet-5"}, {"claude-sonnet-5": entry("claude-sonnet-5", maker="Anthropic")}, []
        )
        _, body = by_name(planned, "claude-sonnet-5")
        self.assertEqual(body["litellm_params"]["model"], "anthropic/claude-sonnet-5")
        self.assertEqual(body["litellm_params"]["api_base"], catalog.API)

    def test_a_decision_model_uses_the_native_systemone_transport(self):
        clef = entry(
            "clef", maker="Cloudflare", routes=(("experiential_cloud", True),),
            description="Cloudflare's 27B decision model; served through /v1/systemone",
        )
        planned = catalog.plan({"clef"}, {"clef": clef}, [])
        _, body = by_name(planned, "clef")
        self.assertEqual(body["litellm_params"]["model"], "hosted_vllm/clef")
        self.assertEqual(body["litellm_params"]["custom_llm_provider"], "hosted_vllm")
        self.assertEqual(body["model_info"]["mode"], "decisions")
        self.assertEqual(body["model_info"]["supported_endpoints"], ["/v1/systemone"])

    def test_unreachable_inactive_or_non_text_models_are_not_registered(self):
        public = {
            "unreachable": entry("unreachable"),
            "inactive": entry("inactive", status="inactive"),
            "images": entry("images", outputs=("image",)),
            "no-route": {"model": {"slug": "no-route", "status": "active",
                                   "output_modalities": ["text"]},
                         "providers": [{"provider": "x", "routable": False}]},
        }
        self.assertEqual(catalog.plan({"inactive", "images", "no-route"}, public, []), [])

    def test_a_retired_name_is_not_registered(self):
        self.assertEqual(
            catalog.plan({"m"}, {"m": entry("m")}, [], retired=["m"]), []
        )

    def test_an_adopted_legacy_deployment_keeps_its_id_and_order(self):
        rows = [{
            "model_name": "clef",
            "litellm_params": {"model": "openai/clef", "custom_llm_provider": "openai",
                               "api_base": catalog.API + "/v1",
                               "litellm_credential_name": catalog.CREDENTIAL, "order": 1},
            "model_info": {"id": "834d2f18", "blocked": False,
                           "experientiallabs_sync": catalog.LEGACY_OWNER, "mode": "chat"},
        }]
        clef = entry("clef", maker="Cloudflare", description="a decision model on /v1/systemone")
        action, body = by_name(catalog.plan({"clef"}, {"clef": clef}, rows), "clef")
        self.assertEqual(action, "update")
        self.assertEqual(body["model_info"]["id"], "834d2f18")
        self.assertEqual(body["litellm_params"]["model"], "hosted_vllm/clef")
        self.assertEqual(body["litellm_params"]["order"], 1)
        self.assertNotIn("blocked", body)

    def test_a_new_deployment_follows_its_siblings(self):
        rows = [{"model_name": "m", "litellm_params": {"order": 4}, "model_info": {}}]
        _, body = by_name(catalog.plan({"m"}, {"m": entry("m")}, rows), "m")
        self.assertEqual(body["litellm_params"]["order"], 5)

    def test_an_unchanged_deployment_is_not_rewritten(self):
        first = catalog.plan({"m"}, {"m": entry("m")}, [])
        _, body = by_name(first, "m")
        rows = [{
            "model_name": "m",
            "litellm_params": body["litellm_params"],
            "model_info": {**body["model_info"], "id": "d1"},
        }]
        action, _ = by_name(catalog.plan({"m"}, {"m": entry("m")}, rows), "m")
        self.assertEqual(action, "unchanged")

    def test_two_owned_routes_are_an_error(self):
        rows = [{"model_name": "m", "litellm_params": {},
                 "model_info": {catalog.OWNER: catalog.OWNER}}] * 2
        with self.assertRaises(ValueError):
            catalog.plan({"m"}, {"m": entry("m")}, rows)


class DropTest(unittest.TestCase):
    def test_gemini_gets_the_drop_param(self):
        params = catalog.prompt_cache_drop({"model": "openai/gemini-4-neon"})
        self.assertEqual(params["additional_drop_params"], [catalog.DROP_PARAM])

    def test_openai_is_untouched(self):
        params = {"model": "openai/gpt-6-luna"}
        self.assertIs(catalog.prompt_cache_drop(params), params)

    def test_an_existing_drop_is_not_duplicated(self):
        params = {"model": "anthropic/claude-sonnet-5",
                  "additional_drop_params": [catalog.DROP_PARAM]}
        self.assertIs(catalog.prompt_cache_drop(params), params)


class FakeClient:
    def __init__(self):
        self.calls = []

    def post(self, path, body, timeout=None):
        self.calls.append(("POST", path, body))
        return {"model_id": "new-id"}

    def patch(self, path, body):
        self.calls.append(("PATCH", path, body))
        return {}


class ApplyTest(unittest.TestCase):
    def test_create_and_update_hit_the_right_routes(self):
        client = FakeClient()
        planned = [
            ("create", {"model_name": "a", "litellm_params": {}, "model_info": {}}),
            ("update", {"model_name": "b", "litellm_params": {},
                        "model_info": {"id": "id-b"}}),
            ("unchanged", {"model_name": "c", "litellm_params": {}, "model_info": {}}),
        ]
        self.assertEqual(
            catalog.apply_plan(client, planned),
            [{"model": "a", "action": "created"}, {"model": "b", "action": "updated"}],
        )
        self.assertEqual([call[1] for call in client.calls],
                         ["/model/new", "/model/id-b/update"])

    def test_a_failed_write_is_reported_not_raised(self):
        class Failing(FakeClient):
            def post(self, path, body, timeout=None):
                return {"error_status": 400}

        self.assertEqual(
            catalog.apply_plan(Failing(), [("create", {"model_name": "a",
                                                       "litellm_params": {}, "model_info": {}})]),
            [{"model": "a", "action": "failed", "detail": "error_400"}],
        )


class SyncTest(unittest.TestCase):
    def test_a_missing_key_is_reported(self):
        with tempfile.TemporaryDirectory() as home:
            self.assertEqual(catalog.sync(None, [], Path(home)),
                             {"source": "experientiallabs", "skipped": "no_credential"})

    def test_preview_plans_without_writing(self):
        planned = [("create", {"model_name": "a", "litellm_params": {}, "model_info": {}})]
        with tempfile.TemporaryDirectory() as home, \
                mock.patch.object(catalog, "fetch", return_value=({"a"}, {"a": {}})), \
                mock.patch.object(catalog, "plan", return_value=planned):
            result = catalog.sync(None, [], Path(home), key="k")
        self.assertEqual(result["create"], 1)
        self.assertEqual(result["actions"], [{"model": "a", "action": "create"}])
        self.assertNotIn("error", result)


if __name__ == "__main__":
    unittest.main()
