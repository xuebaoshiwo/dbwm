import json
import unittest
from copy import deepcopy

from bfcl_eval.consistency.data_generator_v2.generate_tool_state_specs import (
    BFCL_ROOT,
    parse_json_object,
    build_prompt,
    SYSTEM_PROMPT,
    validate_spec,
)
from bfcl_eval.consistency.data_generator_v2.migrate_trading_bot_hard_sources import annotate


VALID_SPEC = {
    "schema_version": "1.0",
    "tool": "read_value",
    "branches": [
        {
            "id": "success_read",
            "if": "otherwise",
            "uses": [],
            "description": "The value is available.",
        }
    ],
    "mutations": [],
    "returns": [
        {
            "branch": "success_read",
            "fields": [
                {
                    "path": "$.result.value",
                    "source_kind": "state_direct",
                    "source": "$.state_before.record.value",
                    "logic": "Return the stored value unchanged.",
                }
            ],
        }
    ],
    "warnings": [],
}


class ToolStateSpecValidationTests(unittest.TestCase):
    def test_hard_specs_are_complete_and_migration_is_idempotent(self):
        directory = BFCL_ROOT / "bfcl_eval/consistency/data_v2/trading_bot_hard"
        specs = json.loads((directory / "all_tools.json").read_text(encoding="utf-8"))
        self.assertEqual(len(specs), 24)
        for spec in specs:
            with self.subTest(tool=spec["tool"]):
                self.assertEqual(spec["schema_version"], "1.3")
                validate_spec(spec, spec["tool"])
                individual = json.loads((directory / f"{spec['tool']}.json").read_text(encoding="utf-8"))
                self.assertEqual(spec, individual)
                self.assertEqual(spec, annotate(deepcopy(spec)))

        by_tool = {spec["tool"]: spec for spec in specs}
        def field(tool, branch, path):
            result = next(r for r in by_tool[tool]["returns"] if r["branch"] == branch)
            return next(f for f in result["fields"] if f["path"] == path)

        self.assertEqual(field("get_account_info", "success_account_info", "$.result.balance")
                         ["state_source_relations"][0]["recoverability"], "exact")
        self.assertEqual(field("execute_order", "success_buy", "$.result.filled_price")
                         ["state_source_relations"][0]["recoverability"], "conditional")
        self.assertEqual(field("filter_stocks_by_price", "success_filtered", "$.result.filtered_stocks")
                         ["state_source_relations"][0]["recoverability"], "partial")
        self.assertEqual(field("get_available_stocks", "success_extended_context", "$.result.stock_list")
                         ["state_source_relations"], [])
        self.assertIn("$.state_before.transaction_history", next(m for m in by_tool["fund_account"]["mutations"]
                       if m["target"] == "$.state_after.transaction_history")["value_from"])
        status = field("place_order", "success_placed_pending_order", "$.result.status")
        self.assertEqual(status["source_kind"], "constant")
        self.assertEqual(status["state_source_relations"], [])
        self.assertEqual(status["state_observation_relations"][0]["path"],
                         "$.state_after.orders['{result.order_id}'].status")
        self.assertEqual(field("fund_account", "success_funded", "$.result.status")
                         ["state_observation_relations"], [])

    def test_v12_observations_are_separate_from_computational_sources(self):
        spec = deepcopy(VALID_SPEC)
        spec["schema_version"] = "1.2"
        field = spec["returns"][0]["fields"][0]
        field.update(source_kind="constant", value="Ready", state_source_relations=[],
                     state_observation_relations=[{
                         "path": "$.state_after.records['{result.record_id}'].status",
                         "transform": "copy", "recoverability": "exact", "given": [],
                         "reason": "The returned status equals the stored status.",
                     }])
        field.pop("source")
        validate_spec(spec, "read_value")
        field["state_observation_relations"][0].update(
            transform="arithmetic", recoverability="conditional", given=["$.state_before.fee"])
        validate_spec(spec, "read_value")
        field["state_observation_relations"][0]["path"] = "$.state_before.records.status"
        with self.assertRaisesRegex(ValueError, "invalid path root"):
            validate_spec(spec, "read_value")

    def test_v12_requires_observation_lists_and_rejects_duplicate_paths(self):
        spec = deepcopy(VALID_SPEC)
        spec["schema_version"] = "1.2"
        field = spec["returns"][0]["fields"][0]
        field["state_source_relations"] = [{"path": field["source"], "transform": "copy",
            "recoverability": "exact", "given": [], "reason": "Returned unchanged."}]
        with self.assertRaisesRegex(ValueError, "state_observation_relations must be an array"):
            validate_spec(spec, "read_value")
        field["state_observation_relations"] = [{"path": "$.state_after.record.value", "transform": "copy",
            "recoverability": "exact", "given": [], "reason": "Unchanged stored value."}]
        validate_spec(spec, "read_value")
        field["state_observation_relations"].append(deepcopy(field["state_observation_relations"][0]))
        with self.assertRaisesRegex(ValueError, "uniquely match"):
            validate_spec(spec, "read_value")

    def test_generation_prompt_requests_post_state_for_an_arbitrary_domain(self):
        prompt = build_prompt("def create_record(): pass", [{"name": "create_record"}], {"name": "create_record"})
        self.assertIn("create_record", prompt)
        self.assertIn("state_observation_relations", SYSTEM_PROMPT)
        self.assertIn('"schema_version": "1.3"', SYSTEM_PROMPT)
        self.assertIn("target_identity_sources", SYSTEM_PROMPT)

    def test_order_post_state_correspondences_match_backend_execution(self):
        from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.trading_bot_hard import TradingBot

        for terminal in ("execute_order", "cancel_order"):
            backend = TradingBot()
            backend._load_scenario({"authenticated": True, "market_status": "Open",
                                    "orders": {}, "order_counter": 100}, long_context=False)
            created = backend.place_order("Buy", "AAPL", 300.0, 1)
            order_id = created["order_id"]
            self.assertEqual(backend.order_counter, order_id + 1)
            self.assertEqual(backend.orders[order_id]["id"], order_id)
            for field in ("status", "price", "amount", "order_type"):
                self.assertEqual(created[field], backend.orders[order_id][field])
            activated = backend.activate_order(order_id)
            self.assertEqual(activated["status"], backend.orders[order_id]["status"])
            result = getattr(backend, terminal)(order_id)
            self.assertNotIn("error", result)
            self.assertEqual(result["status"], backend.orders[order_id]["status"])
            if terminal == "execute_order":
                self.assertEqual(result["filled_price"], backend.orders[order_id]["filled_price"])

    def test_v11_direct_source_classification(self):
        spec = deepcopy(VALID_SPEC)
        spec["schema_version"] = "1.1"
        spec["returns"][0]["fields"][0]["state_source_relations"] = [{
            "path": "$.state_before.record.value", "transform": "copy",
            "recoverability": "exact", "given": [], "reason": "Returned unchanged.",
        }]
        validate_spec(spec, "read_value")
        spec["returns"][0]["fields"][0]["state_source_relations"] = []
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            validate_spec(spec, "read_value")

    def test_v11_rejects_incorrect_inverse_and_missing_co_source(self):
        spec = deepcopy(VALID_SPEC)
        spec["schema_version"] = "1.1"
        field = spec["returns"][0]["fields"][0]
        field.update(source_kind="derived", value_from=["$.state_before.record.value", "$.state_before.fee"])
        field.pop("source")
        field["state_source_relations"] = [{"path": "$.state_before.record.value",
            "transform": "comparison", "recoverability": "exact", "given": ["$.state_before.fee"],
            "reason": "A threshold comparison."}]
        with self.assertRaisesRegex(ValueError, "exact recoverability requires copy"):
            validate_spec(spec, "read_value")
        field["state_source_relations"][0]["recoverability"] = "partial"
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            validate_spec(spec, "read_value")

    def test_valid_spec(self):
        validate_spec(VALID_SPEC, "read_value")

    def test_rejects_invalid_state_direct_source(self):
        spec = {
            **VALID_SPEC,
            "returns": [
                {
                    "branch": "success_read",
                    "fields": [
                        {
                            "path": "$.result.value",
                            "source_kind": "state_direct",
                            "source": "self.record.value",
                            "logic": "Return the stored value unchanged.",
                        }
                    ],
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "invalid path root"):
            validate_spec(spec, "read_value")

    def test_rejects_branch_without_return(self):
        spec = {
            **VALID_SPEC,
            "branches": VALID_SPEC["branches"]
            + [
                {
                    "id": "error_invalid_input",
                    "if": "$.args.value < 0",
                    "uses": ["$.args.value"],
                    "description": "The input is invalid.",
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "Every branch"):
            validate_spec(spec, "read_value")

    def test_rejects_non_english_description(self):
        spec = {
            **VALID_SPEC,
            "branches": [
                {
                    **VALID_SPEC["branches"][0],
                    "description": "Value available: non-English text follows.",
                }
            ],
        }
        spec["branches"][0]["description"] += chr(20540)
        with self.assertRaisesRegex(ValueError, "English-only"):
            validate_spec(spec, "read_value")

    def test_rejects_branch_without_outcome_prefix(self):
        spec = {
            **VALID_SPEC,
            "branches": [
                {
                    **VALID_SPEC["branches"][0],
                    "id": "ordinary_read",
                }
            ],
            "returns": [
                {
                    **VALID_SPEC["returns"][0],
                    "branch": "ordinary_read",
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "success_, fallback_, or error_"):
            validate_spec(spec, "read_value")

    def test_parses_json_code_fence(self):
        fence = chr(96) * 3
        raw = fence + "json\n" + json.dumps(VALID_SPEC) + "\n" + fence
        self.assertEqual(parse_json_object(raw), VALID_SPEC)


if __name__ == "__main__":
    unittest.main()
