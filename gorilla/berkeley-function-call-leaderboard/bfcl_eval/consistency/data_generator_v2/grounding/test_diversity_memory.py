"""Offline tests for memory persistence, cost bounds and pipeline integration."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from . import ground_and_replay as pipeline
from .diversity_memory import (
    DiversityMemory, GUIDE_CHARS, INPUT_CHARS, MODEL_TOKENS,
    _bounded_features, _bounded_text, extract_features, memory_scope,
)


def candidate(book="book-17", quantity=3):
    return {
        "initial_state": {"books": {book: {"copies": quantity, "category": "science"}},
                          "members": [{"name": "Mira", "loans": []}]},
        "placeholder_values": {"book_id_1": book},
        "steps": [{"step_id": 1, "tool": "lend_book", "branch": "success_lent",
                   "args": {"book_id": book, "days": 14}}],
    }


def symbolic():
    return {"id": "library-case", "steps": [
        {"index": 1, "tool": "lend_book", "branch": "success_lent", "branch_condition": "copies > 0",
         "placeholder_bindings": {"args.book_id": "book_id_1"}, "role": "write"}
    ]}


TOOLS = {"lend_book": {"name": "lend_book", "parameters": {"properties": {
    "book_id": {"type": "string"}, "days": {"type": "integer"}}, "required": ["book_id", "days"]}}}


class DiversityMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "pool.json"
        self.memory = DiversityMemory(self.path)
        self.compact = pipeline.compact_symbolic_trajectory(symbolic(), TOOLS)

    def remember(self, value=None, summary="已有样例集中使用单实体库存和两周借期", scope="scope"):
        ask = Mock(return_value={"summary": summary})
        info = self.memory.remember({"status": "accepted", "candidate": value or candidate()},
                                    scope, "case", ask, "test")
        return info, ask

    def test_general_features_capture_parameters_state_and_numeric_scale(self):
        features = extract_features(candidate())
        self.assertEqual(features["tools"], ["lend_book"])
        params = features["arguments"]["lend_book"]
        self.assertEqual(params['$[*]["days"]']["numeric"]["min"], 14)
        self.assertEqual(features["initial_state"]['$["books"][*]["copies"]']["numeric"]["max"], 3)
        changed = extract_features(candidate("book-88"))
        self.assertEqual(set(changed["initial_state"]), set(features["initial_state"]))

    def test_small_feature_slice_preserves_values_and_state_shapes(self):
        features = _bounded_features(extract_features(candidate()), 900)
        self.assertTrue(any("value_counts" in stats for stats in features["arguments"].values()))
        self.assertTrue(any("size" in stats for stats in features["initial_state"].values()))
        self.assertTrue(any("value_counts" in stats for stats in features["initial_state"].values()))
        self.assertNotIn("$", features["bindings"])

    def test_one_summary_persists_and_replaces_without_per_case_prose(self):
        self.remember(summary="单实体库存")
        self.remember(candidate("book-99", 25), summary="库存数量有差异，借期仍集中为两周")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["summary"], "库存数量有差异，借期仍集中为两周")
        self.assertEqual(len(data["records"]), 2)
        self.assertTrue(all("summary" not in r and "description" not in r for r in data["records"]))
        retrieved = DiversityMemory(self.path).retrieve(self.compact, "scope")
        self.assertEqual(retrieved["accepted_in_scope"], 2)
        self.assertEqual(retrieved["frequencies_from_stored_top_values"]['lend_book:$[*]["days"]'][0]["count"], 2)

    def test_rejected_duplicate_and_cold_start_make_no_paid_calls(self):
        ask = Mock()
        self.assertEqual(self.memory.guide(self.compact, "scope", ask, "test")["guide"], "")
        self.memory.remember({"status": "rejected"}, "scope", "case", ask, "test")
        ask.assert_not_called()
        self.assertFalse(self.path.exists())
        self.remember()
        duplicate = self.memory.remember({"status": "accepted", "candidate": candidate()}, "scope", "case", ask, "test")
        self.assertTrue(duplicate["duplicate"])
        ask.assert_not_called()

    def test_scope_isolation_and_retrieval_sampling(self):
        for i in range(8):
            self.remember(candidate(f"book-{i}", i + 1))
        self.remember(candidate("foreign-book"), scope="foreign")
        first = self.memory.retrieve(self.compact, "scope")
        self.assertEqual(first, self.memory.retrieve(self.compact, "scope"))
        self.assertEqual(len(first["retrieved"]), 4)
        self.assertEqual(first["accepted_in_scope"], 8)
        self.assertEqual(self.memory.retrieve(self.compact, "unseen")["global_summary"], "")
        self.assertNotEqual(memory_scope("code-v1", TOOLS, None), memory_scope("code-v2", TOOLS, None))

    def test_hard_limits_and_avoidance_only_without_retries(self):
        self.remember()
        ask = Mock(return_value={"guide": "避免反复使用相同编号，优先选择新的编号；避免空历史"})
        result = self.memory.guide(self.compact, "scope", ask, "test")
        self.assertEqual(result["guide"], "避免反复使用相同编号；避免空历史")
        self.assertLessEqual(len(result["guide"]), GUIDE_CHARS)
        self.assertEqual(ask.call_count, 1)
        self.assertEqual(ask.call_args.kwargs["max_tokens"], MODEL_TOKENS)
        self.assertEqual(_bounded_text("避免" + "同" * 80, 50, avoidance=True), "")
        old = json.loads(self.path.read_text(encoding="utf-8"))["summary"]
        info, ask = self.remember(candidate("book-other"), summary="很" * 101)
        self.assertIn("warning", info)
        self.assertEqual(ask.call_count, 1)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["summary"], old)

    def test_prompt_size_is_bounded_for_large_chain_and_pool(self):
        for i in range(6):
            item = candidate(f"book-{i}")
            item["initial_state"]["extras"] = {f"field-{n}": "X" * 200 for n in range(200)}
            self.remember(item)
        chain = copy.deepcopy(self.compact)
        chain["steps"] *= 500
        ask = Mock(return_value={"guide": "避免空历史"})
        self.memory.guide(chain, "scope", ask, "test")
        payload = ask.call_args.args[0][1]["content"]
        self.assertLessEqual(len(payload), INPUT_CHARS)
        self.assertGreater(json.loads(payload)["omitted_steps"], 0)
        self.assertNotIn("BACKEND SOURCE", payload)

    def test_summary_failure_retains_structured_record_and_previous_summary(self):
        self.remember()
        ask = Mock(side_effect=RuntimeError("unavailable"))
        info = self.memory.remember({"status": "accepted", "candidate": candidate("second")},
                                    "scope", "case", ask, "test")
        self.assertTrue(info["stored"])
        self.assertIn("warning", info)
        self.assertEqual(len(json.loads(self.path.read_text(encoding="utf-8"))["records"]), 2)
        self.assertFalse(self.path.with_suffix(".json.lock").exists())

    def test_configured_budget_applies_to_both_short_calls(self):
        memory = DiversityMemory(self.path, summary_chars=50, max_tokens=512, model="small-model")
        ask = Mock(side_effect=[{"summary": "单实体库存"}, {"guide": "避免重复单实体库存"}])
        memory.remember({"status": "accepted", "candidate": candidate()}, "scope", "case", ask, "test")
        memory.guide(self.compact, "scope", ask, "test")
        self.assertEqual([call.kwargs["max_tokens"] for call in ask.call_args_list], [512, 512])
        self.assertEqual([call.kwargs["model"] for call in ask.call_args_list], ["small-model", "small-model"])
        with self.assertRaises(ValueError):
            DiversityMemory(self.path, max_tokens=0)

    def test_locked_pool_is_not_overwritten(self):
        self.remember()
        before = self.path.read_bytes()
        self.path.with_suffix(".json.lock").touch()
        with self.assertRaises(FileExistsError):
            self.remember(candidate("second"))
        self.assertEqual(self.path.read_bytes(), before)


class PipelineMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.backend = Path(self.temp.name) / "backend.py"
        self.backend.write_text("# generic library backend", encoding="utf-8")
        self.memory = DiversityMemory(Path(self.temp.name) / "pool.json")
        self.scope = memory_scope(self.backend.read_text(), list(TOOLS.values()), None)
        self.memory.remember({"status": "accepted", "candidate": candidate()}, self.scope, "prior",
                             Mock(return_value={"summary": "借期集中为两周"}), "test")

    def run_case(self, memory, responses, replay_results=None, max_debug=0):
        with patch.object(pipeline, "_model_json", side_effect=responses) as ask, patch.object(
            pipeline, "replay", side_effect=replay_results or [pipeline.ReplayResult(True, [], [])]
        ):
            result = pipeline.ground_one(symbolic(), tool_doc=list(TOOLS.values()), tool_map=TOOLS,
                                        backend_path=self.backend, backend_class=None, model="test",
                                        max_debug=max_debug, max_tokens=1000, memory=memory)
        return result, ask

    def test_empty_model_output_reports_budget_without_exposing_reasoning(self):
        response = {"choices": [{"finish_reason": "length", "message": {
            "content": "", "reasoning_content": "private reasoning"}}]}
        with patch.object(pipeline, "chat", return_value=response):
            with self.assertRaisesRegex(pipeline.GroundingError, "finish_reason=length, max_tokens=512") as error:
                pipeline._model_json([], model="test", max_tokens=512)
        self.assertNotIn("private reasoning", str(error.exception))

    def test_same_guide_reaches_both_phases_and_repair_only_accepted_enters_pool(self):
        responses = [
            {"guide": "避免重复两周借期和单实体库存"},
            {"placeholder_values": {"book_id_1": "book-9"}},
            {"initial_state": {"copies": 0}, "steps": [{"step_id": 1, "args": {"days": 21}}]},
            {"initial_state": {"copies": 4}, "steps": [{"step_id": 1, "args": {"days": 21}}]},
            {"valid": True, "steps": [{"step_id": 1, "branch_match": True}]},
            {"summary": "借期和库存数量已有差异"},
        ]
        result, ask = self.run_case(self.memory, responses, [pipeline.ReplayResult(False, [], ["empty"]),
                                                            pipeline.ReplayResult(True, [], [])], max_debug=1)
        self.assertEqual(result["status"], "accepted")
        self.assertTrue(all(call.kwargs["timeout"] == 300 for call in ask.call_args_list))
        for index in (1, 2, 3):
            self.assertIn(responses[0]["guide"], ask.call_args_list[index].args[0][0]["content"])
        self.assertEqual(len(json.loads(self.memory.path.read_text(encoding="utf-8"))["records"]), 2)
        self.assertTrue(result["diversity"]["update"]["stored"])

    def test_disabled_means_no_memory_calls_or_writes(self):
        before = self.memory.path.read_bytes()
        result, ask = self.run_case(None, [
            {"placeholder_values": {"book_id_1": "book-9"}},
            {"initial_state": {}, "steps": [{"step_id": 1, "args": {"days": 21}}]},
            {"valid": True},
        ])
        self.assertFalse(result["diversity"]["enabled"])
        self.assertEqual(ask.call_count, 3)
        self.assertEqual(self.memory.path.read_bytes(), before)

    def test_failed_case_does_not_update_summary(self):
        before = self.memory.path.read_bytes()
        result, ask = self.run_case(self.memory, [
            {"guide": "避免重复两周借期"}, {"placeholder_values": {"book_id_1": "book-9"}},
            {"initial_state": {}, "steps": [{"step_id": 1, "args": {"days": 21}}]},
        ], [pipeline.ReplayResult(False, [], ["failed"])])
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(ask.call_count, 3)
        self.assertEqual(self.memory.path.read_bytes(), before)

    def test_guide_failure_does_not_retry_or_discard_executable_case(self):
        result, ask = self.run_case(self.memory, [
            RuntimeError("short-call budget exhausted"),
            {"placeholder_values": {"book_id_1": "book-9"}},
            {"initial_state": {}, "steps": [{"step_id": 1, "args": {"days": 21}}]},
            {"valid": True}, {"summary": "借期不再集中为两周"},
        ])
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["diversity"]["guide"], "")
        self.assertIn("budget exhausted", result["diversity"]["warning"])
        self.assertTrue(result["diversity"]["update"]["stored"])
        self.assertEqual(ask.call_count, 5)

    def test_cli_default_enabled_and_explicit_off(self):
        root = Path(self.temp.name)
        tools_path = root / "tools.json"
        tools_path.write_text(json.dumps(list(TOOLS.values())), encoding="utf-8")
        source = root / "chain.json"
        source.write_text(json.dumps(symbolic()), encoding="utf-8")
        for enabled in (True, False):
            output = root / str(enabled)
            args = ["--trajectory", str(source), "--tool-doc", str(tools_path),
                    "--backend", str(self.backend), "--output-dir", str(output),
                    "--memory-path", str(root / "unused.json")]
            if not enabled:
                args.append("--no-memory")
            with patch.object(pipeline, "ground_one", return_value={"status": "accepted"}) as ground:
                self.assertEqual(pipeline.main(args), 0)
            passed_memory = ground.call_args.kwargs["memory"]
            self.assertEqual(passed_memory is not None, enabled)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["diversity_memory"]["enabled"], enabled)
        self.assertFalse((root / "unused.json").exists())


if __name__ == "__main__":
    unittest.main()
