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

    def test_an_openai_decisions_model_keeps_the_openai_transport(self):
        luna = entry(
            "gpt-6-luna-decisions", maker="OpenAI",
            description=(
                "GPT-6 Luna on the OpenAI Decisions API; served through POST "
                "/v1/decisions in OpenAI's request shape; not a chat or streaming model."
            ),
        )
        _, body = by_name(catalog.plan({"gpt-6-luna-decisions"}, {"gpt-6-luna-decisions": luna}, []), "gpt-6-luna-decisions")
        self.assertEqual(body["litellm_params"]["model"], "openai/gpt-6-luna-decisions")
        self.assertEqual(body["litellm_params"]["custom_llm_provider"], "openai")
        self.assertEqual(body["model_info"]["mode"], "decisions")
        self.assertEqual(body["model_info"]["supported_endpoints"], ["/v1/decisions"])

    def test_a_chat_model_that_also_serves_decisions_stays_chat(self):
        luna = entry("gpt-6-luna", maker="OpenAI", description="general chat model",
                     supported_endpoints=["/v1/chat/completions", "/v1/decisions"])
        _, body = by_name(catalog.plan({"gpt-6-luna"}, {"gpt-6-luna": luna}, []), "gpt-6-luna")
        self.assertEqual(body["model_info"]["mode"], "chat")
        self.assertNotIn("supported_endpoints", body["model_info"])

    def test_a_described_decision_only_model_uses_the_openai_route(self):
        gemini = entry(
            "gemini-decision-1", maker="Google",
            description=(
                "A decision-only multimodal model; called on POST /v1/decisions; "
                "zero output tokens."
            ),
        )
        _, body = by_name(catalog.plan({"gemini-decision-1"}, {"gemini-decision-1": gemini}, []), "gemini-decision-1")
        self.assertEqual(body["litellm_params"]["model"], "openai/gemini-decision-1")
        self.assertEqual(body["model_info"]["mode"], "decisions")
        self.assertEqual(body["model_info"]["supported_endpoints"], ["/v1/decisions"])

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

    def test_a_free_variant_is_registered_from_its_base_entry(self):
        luna = entry(
            "gpt-6-luna-decisions", maker="OpenAI",
            description="served through POST /v1/decisions; not a chat or streaming model",
        )
        luna["providers"][0].update({
            "input_nano_usd_per_million": 100_000_000,
            "output_nano_usd_per_million": 0,
        })
        planned = catalog.plan(
            {"gpt-6-luna-decisions:free"}, {"gpt-6-luna-decisions": luna}, []
        )
        action, body = by_name(planned, "gpt-6-luna-decisions:free")
        self.assertEqual(action, "create")
        self.assertEqual(body["litellm_params"]["model"], "openai/gpt-6-luna-decisions:free")
        self.assertEqual(body["model_info"]["mode"], "decisions")
        self.assertTrue(body["model_info"]["is_free"])
        self.assertEqual(body["litellm_params"]["input_cost_per_token"], 0.0)
        self.assertEqual(
            body["model_info"]["original_input_cost_per_token"], 1e-07
        )
        self.assertEqual(
            body["model_info"]["original_output_cost_per_token"], 0.0
        )

    def test_an_authenticated_model_without_public_metadata_is_still_skipped(self):
        self.assertEqual(catalog.plan({"ghost"}, {}, []), [])

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


class BatchTest(unittest.TestCase):
    def test_a_batch_named_model_is_registered_as_batch(self):
        batch = entry(
            "gpt-5.4-batch", maker="OpenAI",
            routes=(("openai", True),),
        )
        batch["providers"][0]["provider_model_id"] = "gpt-5.4:batch"
        _, body = by_name(
            catalog.plan({"gpt-5.4-batch"}, {"gpt-5.4-batch": batch}, []),
            "gpt-5.4-batch",
        )
        self.assertEqual(body["litellm_params"]["model"], "openai/gpt-5.4")
        self.assertEqual(body["model_info"]["mode"], "batch")
        self.assertEqual(body["model_info"]["supported_endpoints"], ["/v1/batches"])

    def test_a_chat_model_with_a_batch_endpoint_stays_chat(self):
        chat = entry("m", supported_endpoints=["/v1/chat/completions", "/v1/batch"])
        _, body = by_name(catalog.plan({"m"}, {"m": chat}, []), "m")
        self.assertEqual(body["model_info"]["mode"], "chat")
        self.assertNotIn("supported_endpoints", body["model_info"])

    def test_a_provider_prefixed_batch_route_is_not_doubled(self):
        batch = entry("claude-opus-5-batch", maker="Anthropic",
                      routes=(("anthropic", None),))
        batch["providers"][0]["provider_model_id"] = "anthropic/claude-opus-5"
        _, body = by_name(
            catalog.plan({"claude-opus-5-batch"}, {"claude-opus-5-batch": batch}, []),
            "claude-opus-5-batch",
        )
        self.assertEqual(body["litellm_params"]["model"], "anthropic/claude-opus-5")

    def test_a_public_batch_model_is_registered_without_chat_authentication(self):
        batch = entry("gpt-5.4-batch", routes=(("openai", None),))
        planned = catalog.plan({"gpt-5.4-batch"}, {"gpt-5.4-batch": batch}, [])
        action, body = by_name(planned, "gpt-5.4-batch")
        self.assertEqual(action, "create")
        self.assertEqual(body["model_info"]["mode"], "batch")

    def test_a_stale_chat_batch_registration_is_corrected_in_place(self):
        batch = entry("gpt-5.4-batch", routes=(("openai", None),))
        rows = [{
            "model_name": "gpt-5.4-batch",
            "litellm_params": {"model": "openai/gpt-5.4-batch", "order": 1},
            "model_info": {"id": "old-id", "mode": "chat"},
        }]
        action, body = by_name(
            catalog.plan({"gpt-5.4-batch"}, {"gpt-5.4-batch": batch}, rows),
            "gpt-5.4-batch",
        )
        self.assertEqual(action, "update")
        self.assertEqual(body["model_info"]["id"], "old-id")
        self.assertEqual(body["model_info"]["mode"], "batch")

    def test_sync_batch_plans_public_entries_without_auth(self):
        batch = entry("gpt-5.4-batch", routes=(("openai", None),))
        with tempfile.TemporaryDirectory() as home, \
                mock.patch.object(catalog, "fetch", return_value=(set(), {"gpt-5.4-batch": batch})):
            result = catalog.sync_batch(None, [], Path(home), key="k")
        self.assertEqual(result["batch"], 1)
        self.assertEqual(result["create"], 1)
        self.assertEqual(result["actions"], [{"model": "gpt-5.4-batch", "action": "create"}])

    def test_batch_catalog_separates_authenticated_from_advertised(self):
        public = {
            "reachable-batch": entry("reachable-batch"),
            "advertised-batch": entry("advertised-batch"),
            "chat": entry("chat"),
        }
        summary = catalog.batch_catalog(
            {"reachable-batch"}, public,
            {"openrouter/x:batch": {"mode": "chat"}, "openrouter/x": {}},
        )
        self.assertEqual(summary["found"], 2)
        self.assertEqual(summary["authenticated"], 1)
        self.assertEqual(summary["registrable"], 2)
        self.assertEqual(summary["models"], ["reachable-batch"])
        self.assertEqual(summary["price_map"], 1)


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
