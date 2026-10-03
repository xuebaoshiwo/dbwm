"""Offline regressions for evaluation semantics, leakage and backend checking."""
from copy import deepcopy
import json
import os
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .checker import check_isolated
from .client import ChatClient, parse_observation
from .common import read_json, write_json
from .dataset import make_jobs
from .generate_examples import generate
from .prompts import build_messages
from .run import run_job, summarize


def fixture():
    calls = [{"step_id": i, "tool": "read", "args": {"key": str(i)}} for i in (1, 3, 5)]
    case = {"query": "Read keys 1, 3 and 5 in order.", "tool_chain": calls,
            "trace": [{**call, "result": {"value": call["step_id"]}} for call in calls],
            "initial_state": {"secret": "HIDDEN_INITIAL_STATE"}}
    symbolic = {"planning": {"target_chains": {
        "a": {"anchor_step": 1, "read_step": 3, "write_steps": [], "read_phase": "before"},
        "b": {"anchor_step": 1, "read_step": 5, "write_steps": [], "read_phase": "before"},
    }}}
    examples = {"read": [{"tool_call": {"tool": "read", "args": {}}, "observation": {"value": i},
                           "provenance": {"secret": "HIDDEN_EXAMPLE_STATE"}} for i in range(3)]}
    return case, symbolic, examples


class FakeClient:
    def __init__(self):
        self.messages = []

    def predict(self, messages):
        self.messages.append(deepcopy(messages))
        return {"status": "ok", "observation": {"value": 100 + len(self.messages)}}


class PipelineTests(unittest.TestCase):
    def test_nested_gateway_billing_reveals_hidden_reasoning(self):
        client = ChatClient(model="fake", thinking="disabled")
        response = {"choices": [{"message": {"content": "{}"}}],
                    "provider_response": {"usage": {"billing_usage": {"openai_usage": {
                        "completion_tokens_details": {"reasoning_tokens": 17}}}}}}
        with patch.object(client, "_request", return_value=response):
            result = client.predict([])
        self.assertEqual(result["status"], "thinking_not_disabled")
        self.assertEqual(result["reasoning_tokens_observed"], 17)

    def test_provider_options_cannot_replace_model_or_prompt(self):
        for key in ("model", "messages", "thinking", "max_tokens"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ChatClient(model="fake", request_options={key: "override"})

    def test_disabled_thinking_rejects_provider_reasoning(self):
        client = ChatClient(model="fake", thinking="disabled")
        response = {"choices": [{"message": {"content": "{}", "reasoning_content": "still thinking"}}]}
        with patch.object(client, "_request", return_value=response):
            self.assertEqual(client.predict([])["status"], "thinking_not_disabled")
        response["choices"][0]["message"].pop("reasoning_content")
        response["usage"] = {"completion_tokens_details": {"reasoning_tokens": 4}}
        with patch.object(client, "_request", return_value=response):
            self.assertEqual(client.predict([])["status"], "thinking_not_disabled")

    def test_native_messages_disables_thinking_and_preserves_raw_response(self):
        client = ChatClient(model="fake", thinking="disabled", api_format="anthropic",
                            base_url="https://example.invalid/v1")
        raw = {"model": "fake", "content": [{"type": "text", "text": '{"value": 7}'}],
               "stop_reason": "end_turn", "usage": {"input_tokens": 10, "cache_read_input_tokens": 20, "output_tokens": 5}}
        with patch("bfcl_eval.consistency.test.client.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())) as request:
            result = client.predict([{"role": "system", "content": "simulate"}, {"role": "user", "content": "predict"}])
        sent = request.call_args.args[0]
        self.assertEqual(sent.full_url, "https://example.invalid/v1/messages")
        payload = json.loads(sent.data)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["system"], "simulate")
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["thinking_observed"])
        self.assertEqual(result["observation"], {"value": 7})
        self.assertEqual(result["response"]["provider_response"], raw)
        self.assertEqual(result["response"]["usage"]["total_tokens"], 35)

    def test_atomic_write_retries_transient_file_lock(self):
        replace = os.replace
        attempts = []
        def transient(source, destination):
            attempts.append(True)
            if len(attempts) == 1:
                raise PermissionError("temporary Windows sharing violation")
            return replace(source, destination)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json(path, {"old": True})
            with patch("bfcl_eval.consistency.test.common.os.replace", side_effect=transient), patch(
                    "bfcl_eval.consistency.test.common.time.sleep"):
                write_json(path, {"new": True})
            self.assertEqual(read_json(path), {"new": True})
            self.assertEqual(len(attempts), 2)

    def test_atomic_write_preserves_old_file_on_permanent_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json(path, {"old": True})
            with patch("bfcl_eval.consistency.test.common.os.replace", side_effect=PermissionError("denied")), patch(
                    "bfcl_eval.consistency.test.common.time.sleep"):
                with self.assertRaises(PermissionError):
                    write_json(path, {"new": True})
            self.assertEqual(read_json(path), {"old": True})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_one_endpoint_job_per_field_uses_stable_ids(self):
        case, symbolic, _ = fixture()
        jobs = make_jobs(case, symbolic, "source.json", "endpoint")
        self.assertEqual([j["step_ids"] for j in jobs], [[1, 3], [1, 3, 5]])
        self.assertEqual([j["target"] for j in jobs], ["a", "b"])
        self.assertTrue(all(j["agent_query"] == case["query"] for j in jobs))

    def test_missing_or_invalid_query_is_rejected(self):
        case, symbolic, _ = fixture()
        for mode in ("endpoint", "rollout"):
            for query in (None, "", " \n\t", {}, ["task"]):
                with self.subTest(mode=mode, query=query):
                    case["query"] = query
                    with self.assertRaisesRegex(ValueError, "query must be a non-empty string"):
                        make_jobs(case, symbolic, "source.json", mode)
            del case["query"]
            with self.assertRaisesRegex(ValueError, "query must be a non-empty string"):
                make_jobs(case, symbolic, "source.json", mode)

    def test_shared_read_is_still_two_independent_tests(self):
        case, symbolic, _ = fixture()
        symbolic["planning"]["target_chains"]["b"]["read_step"] = 3
        self.assertEqual(len(make_jobs(case, symbolic, "source.json", "endpoint")), 2)

    def test_deleted_endpoint_is_not_silently_replaced(self):
        case, symbolic, _ = fixture()
        symbolic["planning"]["target_chains"]["a"]["read_step"] = 4
        with self.assertRaisesRegex(ValueError, "deleted"):
            make_jobs(case, symbolic, "source.json", "endpoint")

    def test_post_call_endpoint_can_also_be_writer(self):
        case, symbolic, _ = fixture()
        monitor = symbolic["planning"]["target_chains"]["a"]
        monitor.update(write_steps=[3], read_phase="after")
        self.assertEqual(len(make_jobs(case, symbolic, "source.json", "endpoint")), 2)
        monitor["read_phase"] = "before"
        with self.assertRaisesRegex(ValueError, "interval"):
            make_jobs(case, symbolic, "source.json", "endpoint")

    def test_prompt_allowlist_hides_reference_and_provenance(self):
        case, _, examples = fixture()
        current = {"tool": "read", "args": {}, "observation": "HIDDEN_REFERENCE", "branch": "HIDDEN_BRANCH"}
        messages = build_messages("simulate", [{"name": "read"}], examples, [], current,
                                  agent_query=case["query"])
        self.assertNotIn("HIDDEN_", json.dumps(messages))
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["agent_query"], case["query"])
        self.assertEqual(len(payload["tool_call_examples"]), 3)
        self.assertEqual(payload["tool_schema"], [{"name": "read"}])

    def run_mode(self, mode):
        case, symbolic, examples = fixture()
        job = make_jobs(case, symbolic, "source.json", mode)[0]
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory, patch(
                "bfcl_eval.consistency.test.run.check_isolated", return_value={"status": "sat"}) as checker:
            result = run_job(job, mode=mode, client=client, task="simulate", tools=[], examples=examples,
                             checker="unused:unused", checker_options={}, checker_timeout=2, directory=directory)
            # A finished checkpoint never requests the model again.
            resumed = run_job(job, mode=mode, client=client, task="simulate", tools=[], examples=examples,
                              checker="unused:unused", checker_options={}, checker_timeout=2,
                              directory=directory, resume=True)
            self.assertEqual(result, resumed)
            self.assertEqual(checker.call_count, 2)
            self.assertEqual(result["agent_query"], case["query"])
            for messages in client.messages:
                self.assertEqual(json.loads(messages[1]["content"])["agent_query"], case["query"])
                self.assertNotIn("HIDDEN_", json.dumps(messages))
            for check in checker.call_args_list:
                self.assertTrue(all(set(step) == {"tool", "args", "observation"}
                                    for step in check.args[0]))
        return client, result

    def test_endpoint_teacher_forcing(self):
        client, result = self.run_mode("endpoint")
        self.assertEqual(len(client.messages), 1)
        payload = json.loads(client.messages[0][1]["content"])
        self.assertEqual(payload["history"][0]["observation"], {"value": 1})
        self.assertEqual(result["evaluated_steps"][-1]["observation"], {"value": 101})
        self.assertEqual(len(result["evaluated_steps"]), 2)

    def test_rollout_uses_only_previous_model_observations(self):
        client, result = self.run_mode("rollout")
        self.assertEqual(len(client.messages), 3)
        last = json.loads(client.messages[-1][1]["content"])
        self.assertEqual([s["observation"]["value"] for s in last["history"]], [101, 102])
        self.assertEqual(result["evaluated_steps"][-1]["observation"], {"value": 103})

    def test_invalid_predictions_are_not_resampled(self):
        client = ChatClient(model="fake", retries=2)
        with patch.object(client, "_request", return_value={"choices": [{"message": {"content": "not json"}}]}) as request:
            self.assertEqual(client.predict([])["status"], "invalid_json")
            self.assertEqual(request.call_count, 1)

    def test_json_arrays_supported_and_invalid_json_rejected(self):
        self.assertEqual(parse_observation('["error"]'), ["error"])
        self.assertEqual(parse_observation('```json\n{"value": 1}\n```'), {"value": 1})
        for text in ('{"x": NaN}', '{"x": 1e400}', '{"x": 1, "x": 2}', 'Explanation: {}', ''):
            with self.assertRaises(ValueError):
                parse_observation(text)

    def test_unknown_and_format_errors_are_not_unsat(self):
        records = [{"status": s} for s in ("sat", "unsat", "unknown", "invalid_format", "api_error")]
        summary = summarize(records, 5)
        self.assertEqual(summary["consistency_rate"], 0.5)
        self.assertEqual(summary["decision_coverage"], 0.4)

    def test_examples_are_real_backend_outputs(self):
        root = Path(__file__).parent / "domains"
        config = read_json(root / "trading_bot_hard.json")
        generated = generate(config, root)
        self.assertEqual(json.loads(json.dumps(generated)), read_json(root / config["examples"]))
        self.assertEqual(sum(map(len, generated["tools"].values())), 72)

    def test_isolated_checker_sat_and_unsat(self):
        steps = [{"tool": "get_account_info", "args": {},
                  "observation": {"account_id": 7, "balance": 100.0, "binding_card": 99}},
                 {"tool": "fund_account", "args": {"amount": 1.0},
                  "observation": {"status": "Account funded successfully"}},
                 {"tool": "get_account_info", "args": {},
                  "observation": {"account_id": 7, "balance": 101.0, "binding_card": 99}}]
        with tempfile.TemporaryDirectory() as directory:
            plugin = "bfcl_eval.consistency.trading_solver_hard:check_trace"
            self.assertEqual(check_isolated(steps, plugin, {}, Path(directory) / "sat", 30)["status"], "sat")
            steps[-1]["observation"]["balance"] = 101.001
            self.assertEqual(check_isolated(steps, plugin, {}, Path(directory) / "unsat", 30)["status"], "unsat")


if __name__ == "__main__":
    unittest.main()
