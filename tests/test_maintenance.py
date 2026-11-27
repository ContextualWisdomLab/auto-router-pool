import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from auto_router_pool import maintenance


class FakeClient:
    def __init__(self, info=None, health=None):
        self.calls = []
        self.info = info or []
        self.health = health or {}

    def get(self, path, timeout=None):
        self.calls.append(("GET", path))
        if path == "/model/info":
            return {"data": self.info}
        if path.startswith("/health?"):
            return self.health.get(path.split("model_id=", 1)[1], {})
        raise AssertionError(path)

    def post(self, path, body, timeout=None):
        self.calls.append(("POST", path, body))
        return {"choices": [{"message": {"content": "607"}}]}

    def patch(self, path, body):
        self.calls.append(("PATCH", path, body))
        return {}


def row(name, identifier, **params):
    return {
        "model_name": name,
        "litellm_params": params,
        "model_info": {"id": identifier},
    }


class PromptCacheDropTest(unittest.TestCase):
    def test_only_providers_that_refuse_the_parameter_are_patched(self):
        rows = [
            row("gemini-4-neon", "a", model="openai/gemini-4-neon"),
            row("gpt-6-luna", "b", model="openai/gpt-6-luna"),
            row("done", "c", model="anthropic/claude-x",
                additional_drop_params=[maintenance.catalog.DROP_PARAM]),
        ]
        client = FakeClient()
        changes = maintenance.prompt_cache_drop_pass(client, rows, apply=True)
        self.assertEqual(changes, [{"model": "gemini-4-neon", "deployment": "a",
                                    "action": "updated"}])
        self.assertEqual(client.calls, [(
            "PATCH", "/model/a/update",
            {"litellm_params": {"additional_drop_params": ["prompt_cache_key"]}},
        )])

    def test_a_preview_writes_nothing(self):
        rows = [row("gemini-4-neon", "a", model="openai/gemini-4-neon")]
        client = FakeClient()
        changes = maintenance.prompt_cache_drop_pass(client, rows)
        self.assertEqual(changes[0]["action"], "would_update")
        self.assertEqual(client.calls, [])


class PooledTest(unittest.TestCase):
    def test_only_the_shared_bridge_credentials_are_owned(self):
        self.assertTrue(maintenance.pooled({
            "api_base": "http://192.168.68.3:8319/v1",
            "litellm_credential_name": "cursor-chat-bridge",
        }))
        self.assertTrue(maintenance.pooled({
            "api_base": "http://127.0.0.1:8317/v1", "model": "openai/gpt-5.5",
            "litellm_credential_name": "cliproxyapi-shared",
        }))
        self.assertFalse(maintenance.pooled({
            "api_base": "http://127.0.0.1:8317/v1", "model": "gemini/x",
            "litellm_credential_name": "cliproxyapi-shared",
        }))
        self.assertFalse(maintenance.pooled({
            "api_base": "https://api.openai.com/v1",
            "litellm_credential_name": "cursor-chat-bridge",
        }))


class CooldownTest(unittest.TestCase):
    def rows(self):
        return [
            row("gpt-6-luna", "d1", api_base="http://127.0.0.1:8319/v1",
                litellm_credential_name="cursor-chat-bridge", order=1),
            row("gpt-6-luna", "d2", api_base="http://127.0.0.1:8319/v1",
                litellm_credential_name="cursor-chat-bridge", order=2),
        ]

    def test_a_preview_caps_nothing(self):
        client = FakeClient()
        with tempfile.TemporaryDirectory() as home:
            result = maintenance.repair_cooldowns(client, self.rows(), Path(home))
        self.assertEqual(result["changed_this_run"], 2)
        self.assertEqual(client.calls, [])

    def test_apply_caps_and_reports_missing_redis(self):
        rows = self.rows()
        for item in rows:
            item["model_info"]["cooldown_time"] = 5
        info = json.loads(json.dumps(rows))
        for item in info:
            item["model_info"]["cooldown_time"] = 30
        client = FakeClient(info=info)
        with tempfile.TemporaryDirectory() as home, \
                mock.patch.object(maintenance, "REDIS_CONF", Path(home) / "missing.conf"):
            result = maintenance.repair_cooldowns(client, rows, Path(home), apply=True)
            backups = list(Path(home).glob("pooled-cooldown-before-*.json"))
        self.assertEqual(result["changed_this_run"], 2)
        self.assertEqual(result["error"], "redis_config_unavailable")
        self.assertEqual(
            [call[1] for call in client.calls if call[0] == "PATCH"],
            ["/model/d1/update", "/model/d2/update"],
        )
        self.assertTrue(backups)


class HealthTest(unittest.TestCase):
    def test_summary_separates_restrictions_and_skips_media(self):
        rows = [
            row("gpt-a", "a", model="openai/a"),
            row("gpt-b", "b", model="openai/b"),
            row("image", "c", model="openai/c"),
        ]
        rows[0]["model_info"]["mode"] = "chat"
        rows[1]["model_info"]["mode"] = "chat"
        rows[2]["model_info"]["mode"] = "image_generation"
        client = FakeClient(health={
            "a": {"healthy_endpoints": [{"model_id": "a"}]},
            "b": {"unhealthy_endpoints": [{
                "model_id": "b", "exception_status": 429,
                "error": "rate limit exceeded",
            }]},
        })
        with tempfile.TemporaryDirectory() as home:
            summary = maintenance.health_report(client, rows, Path(home), workers=1)
            report = json.loads((Path(home) / "deployment-health-results.json").read_text())
        self.assertEqual(summary["healthy"], 1)
        self.assertEqual(summary["unhealthy"], 1)
        self.assertEqual(summary["not_tested"], 1)
        self.assertEqual(summary["account_restrictions"], 1)
        self.assertEqual(report["summary"], summary)


class ChatOnlyTransportTest(unittest.TestCase):
    def test_rewrites_nvidia_openai_deployments_to_hosted_vllm(self):
        nvidia_row = row(
            "minimax-m3.1-flash", "n1",
            model="openai/minimaxai/minimax-m3.1-flash",
            api_base="https://integrate.api.nvidia.com/v1",
        )
        glm_row = row(
            "glm-5.3-flash", "n2",
            model="openai/z-ai/glm-5.3-flash",
            custom_llm_provider="openai",
            api_base="https://integrate.api.nvidia.com/v1",
        )
        glm_row["model_info"]["litellm_provider"] = "openai"
        already_fixed = row(
            "mistral-large-4", "n3",
            model="hosted_vllm/mistralai/mistral-large-4-1t-a52b",
            custom_llm_provider="hosted_vllm",
            api_base="https://integrate.api.nvidia.com/v1",
        )
        openai_row = row(
            "gpt-5.5", "o1",
            model="openai/gpt-5.5",
            custom_llm_provider="openai",
            api_base="https://api.openai.com/v1",
        )
        client = FakeClient()
        changes = maintenance.chat_only_transport_pass(
            client, [nvidia_row, glm_row, already_fixed, openai_row], apply=True
        )
        self.assertEqual(changes, [
            {
                "model": "minimax-m3.1-flash", "deployment": "n1",
                "action": "updated",
                "from": "openai/minimaxai/minimax-m3.1-flash",
                "to": "hosted_vllm/minimaxai/minimax-m3.1-flash",
            },
            {
                "model": "glm-5.3-flash", "deployment": "n2",
                "action": "updated",
                "from": "openai/z-ai/glm-5.3-flash",
                "to": "hosted_vllm/z-ai/glm-5.3-flash",
            },
        ])
        self.assertEqual(client.calls, [
            (
                "PATCH", "/model/n1/update",
                {"litellm_params": {
                    "model": "hosted_vllm/minimaxai/minimax-m3.1-flash",
                    "custom_llm_provider": "hosted_vllm",
                }},
            ),
            (
                "PATCH", "/model/n2/update",
                {
                    "litellm_params": {
                        "model": "hosted_vllm/z-ai/glm-5.3-flash",
                        "custom_llm_provider": "hosted_vllm",
                    },
                    "model_info": {"litellm_provider": "hosted_vllm"},
                },
            ),
        ])

    def test_preview_writes_nothing(self):
        rows = [row(
            "minimax-m3.1-flash", "n1",
            model="openai/minimaxai/minimax-m3.1-flash",
            api_base="https://integrate.api.nvidia.com/v1",
        )]
        client = FakeClient()
        changes = maintenance.chat_only_transport_pass(client, rows)
        self.assertEqual(changes[0]["action"], "would_update")
        self.assertEqual(changes[0]["to"], "hosted_vllm/minimaxai/minimax-m3.1-flash")
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
