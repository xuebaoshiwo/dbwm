"""Entity isolation, global lifecycle coverage, and executable trading examples."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from bfcl_eval.consistency.data_generator_v2.backward_state_knowledge import StateKnowledge, covers
from bfcl_eval.consistency.data_generator_v2.entity_bindings import EntityRegistry, monitor_targets, parse_entity_types
from bfcl_eval.consistency.data_generator_v2.entity_lifecycle import LifecyclePlanner
from bfcl_eval.consistency.data_generator_v2.generate_backward_trajectories import (
    BackwardSampler, DEFAULT_CATALOG, load_inputs, load_lifecycle_rules, main,
)
from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.trading_bot_hard import TradingBot
from bfcl_eval.consistency.data_generator_v2.lifecycle_rules import parse_lifecycle_rules
from tests.test_backward_trajectory_sampling import assert_forward_sources, mutation, returned, spec


class EntityTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs, cls.targets, cls.refinements, _ = load_inputs(DEFAULT_CATALOG)
        cls.types = json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8"))["entity_types"]
        cls.rules, _ = load_lifecycle_rules(DEFAULT_CATALOG, cls.specs, cls.targets)

    def sampler(self, seed=42):
        return BackwardSampler(self.specs, self.targets, reader_refinements=self.refinements,
                               lifecycle_rules=self.rules, entity_types=self.types, seed=seed)

    def assert_lifecycles(self, result):
        for entity, lifecycle in result["planning"]["entity_lifecycles"].items():
            states = set(lifecycle["initial_states"])
            terminal_count = 0
            for transition in lifecycle["transitions"]:
                self.assertIn(transition["from"], states)
                states = {transition["to"]}
                terminal_count += transition["to"] in {"Completed", "Cancelled"}
                step = result["steps"][transition["step"] - 1]
                self.assertEqual(step["entity_bindings"]["order"], entity)
            self.assertLessEqual(terminal_count, 1)

    def test_state_knowledge_does_not_cross_entity_keys(self):
        a = "$.state_before.orders['@order:A'].status"
        b = "$.state_before.orders['@order:B'].status"
        self.assertFalse(covers(a, b))
        knowledge = StateKnowledge()
        knowledge.observe({a: {"kind": "return", "source": a}}, 1)
        self.assertIsNone(knowledge.get(b))
        self.assertEqual(knowledge.get(a).anchor_id, 1)

    def test_two_monitored_orders_have_independent_intervals(self):
        monitors = [{"id": entity, "type": "order", "fields": ["status"]} for entity in ("A", "B")]
        for seed in range(4):
            engine = self.sampler(seed)
            result = engine.build(monitors=monitors, min_length=1, max_length=35)
            chains = result["planning"]["target_chains"]
            for entity in ("A", "B"):
                chain = chains[f"{entity}.status"]
                self.assertEqual(chain["write_count"], 2)
                self.assertEqual(chain["write_shortfall"], 1)
                for index in {chain["anchor_step"], chain["read_step"], *chain["write_steps"]}:
                    self.assertEqual(result["steps"][index - 1]["entity_bindings"]["order"], entity)
            self.assertFalse(set(chains["A.status"]["write_steps"]) & set(chains["B.status"]["write_steps"]))
            self.assert_lifecycles(result)
            assert_forward_sources(self, engine, result)

    def test_multiple_fields_of_each_order_share_identity(self):
        engine = self.sampler()
        result = engine.build(monitors=[{"id": entity, "type": "order", "fields": ["status", "amount"]}
                                        for entity in ("A", "B")], min_length=1, max_length=40)
        for entity in ("A", "B"):
            fields = result["planning"]["entities"][entity]["monitored_fields"]
            self.assertEqual(set(fields), {"status", "amount"})
            amount = result["planning"]["target_chains"][fields["amount"]]
            self.assertEqual(amount["write_count"], 0)
            self.assertEqual(amount["write_shortfall"], 3)
            for index in (amount["anchor_step"], amount["read_step"]):
                self.assertEqual(result["steps"][index - 1]["entity_bindings"]["order"], entity)
        self.assert_lifecycles(result)
        assert_forward_sources(self, engine, result)

    def test_holdings_only_uses_distinct_legal_orders(self):
        for seed in range(4):
            engine = self.sampler(seed)
            result = engine.build(monitors=[{"id": "position", "type": "stock", "fields": ["holding"]}],
                                  min_length=1, max_length=40)
            chain = result["planning"]["target_chains"]["position.holding"]
            self.assertEqual(chain["write_count"], 3)
            executions = [step for step in result["steps"] if step["tool"] == "execute_order"]
            entities = {step["entity_bindings"]["order"] for step in executions}
            self.assertEqual(len(entities), 3)
            for entity in entities:
                description = result["planning"]["entities"][entity]
                self.assertEqual(description["monitored_fields"], {})
                self.assertEqual(description["relations"]["symbol"], "position")
                trace = result["planning"]["entity_lifecycles"][entity]["transitions"]
                self.assertEqual([step["tool"] for step in trace], ["activate_order", "execute_order"])
            self.assert_lifecycles(result)
            assert_forward_sources(self, engine, result)

    def test_monitored_order_does_not_block_holdings_writers(self):
        engine = self.sampler()
        result = engine.build(monitors=[{"id": "A", "type": "order", "fields": ["status"]},
                                       {"id": "position", "type": "stock", "fields": ["holding"]}],
                              min_length=1, max_length=40)
        self.assertGreaterEqual(result["planning"]["target_chains"]["position.holding"]["write_count"], 3)
        self.assert_lifecycles(result)
        assert_forward_sources(self, engine, result)

    def test_registry_retains_order_symbol_when_fixing_another_field(self):
        engine = self.sampler()
        engine.build(monitors=[{"id": "position", "type": "stock", "fields": ["holding"]}],
                     min_length=1, max_length=40)
        node = next(node for node in engine.candidate_nodes("$.state_before.orders['@order:order_1'].status")
                    if node.tool == "execute_order")
        self.assertEqual(node.entity_binding.entities["stock"], "position")

    def test_terminal_order_cannot_execute_again_or_be_cancelled(self):
        engine = self.sampler()
        engine.build(monitors=[{"id": "position", "type": "stock", "fields": ["holding"]}],
                     min_length=1, max_length=40)
        registry = engine._registry
        planner = LifecyclePlanner(self.rules, {}, registry, 40)
        path = "$.state_before.orders['@order:order_1'].status"
        execute = next(node for node in engine.candidate_nodes(path) if node.tool == "execute_order")
        cancel = next(node for node in engine.candidate_nodes(path) if node.tool == "cancel_order")
        planner.apply(execute)
        self.assertFalse(planner.eligible(execute))
        self.assertFalse(planner.eligible(cancel))

    def test_unknown_status_writer_is_rejected_for_unmonitored_entity(self):
        engine = self.sampler()
        engine.build(monitors=[{"id": "position", "type": "stock", "fields": ["holding"]}],
                     min_length=1, max_length=40)
        planner = LifecyclePlanner(self.rules, {}, engine._registry, 40)
        node = next(node for node in engine.candidate_nodes("$.state_before.orders['@order:order_1'].status")
                    if node.tool == "activate_order")
        node.branch = {**node.branch, "id": "success_unconfigured"}
        self.assertFalse(planner.eligible(node))

    def test_singletons_and_duplicate_monitor_ids_are_validated(self):
        types = parse_entity_types(self.types)
        for monitors in (
            [{"id": "A", "type": "account", "fields": ["balance"]},
             {"id": "B", "type": "account", "fields": ["balance"]}],
            [{"id": "A", "type": "order", "fields": ["status"]}] * 2,
        ):
            with self.assertRaises(ValueError):
                monitor_targets(monitors, types)

    def test_entity_rules_cannot_silently_run_without_entity_types(self):
        with self.assertRaisesRegex(ValueError, "entity_types"):
            BackwardSampler(self.specs, self.targets, lifecycle_rules=self.rules)

    def test_multiple_argument_ids_of_one_type_are_not_collapsed(self):
        registry = EntityRegistry(parse_entity_types(self.types), {})
        document = {"tool": "transfer", "sources": ["$.state_before.orders['{args.source_id}']",
                                                     "$.state_before.orders['{args.destination_id}']"]}
        with self.assertRaisesRegex(ValueError, "role bindings"):
            registry.bindings_for(document)

    def test_a_read_before_entity_creation_is_rejected(self):
        engine = self.sampler()
        monitors = monitor_targets([{"id": "A", "type": "order", "fields": ["status"]}],
                                   engine.entity_types)
        registry = EntityRegistry(engine.entity_types, monitors)
        planner = LifecyclePlanner(self.rules, monitors, registry, 20)
        planner.created.add("A")
        steps = [{"index": 1, "symbolic_slots": {"args.order_id": "@order:A"}},
                 {"index": 2, "tool": "place_order", "branch": "success_placed_pending_order",
                  "lifecycle_transitions": {"A": {"from": "Absent", "to": "Pending"}}}]
        with self.assertRaisesRegex(ValueError, "before creation"):
            planner.audit(steps)

    def test_non_trading_entities_use_the_same_planner(self):
        path = "$.state_before.records['{args.record_id}'].status"
        entries = [spec("read_record", [returned("status", path)]),
                   spec("open_record", mutations=[mutation(path.replace("state_before", "state_after"))]),
                   spec("close_record", mutations=[mutation(path.replace("state_before", "state_after"))])]
        specs = {entry["tool"]: entry for entry in entries}
        raw_types = {"record": {"paths": ["$.state_before.records['{id}']"],
                                "fields": {"status": "$.state_before.records['{id}'].status"}}}
        types = parse_entity_types(raw_types)
        rules = parse_lifecycle_rules({"version": 2, "entity_types": {"record": {
            "state_field": "status", "initial_states": ["Ready"], "transitions": [
                {"tool": "open_record", "branch": "success_main", "from": "Ready", "to": "Active"},
                {"tool": "close_record", "branch": "success_main", "from": "Active", "to": "Done"}]}}}, specs, {}, types)
        engine = BackwardSampler(specs, {}, entity_types=raw_types, lifecycle_rules=rules)
        result = engine.build(monitors=[{"id": entity, "type": "record", "fields": ["status"]}
                                       for entity in ("first", "second")], min_length=1, max_length=20)
        for entity in ("first", "second"):
            trace = result["planning"]["entity_lifecycles"][entity]["transitions"]
            self.assertEqual([step["tool"] for step in trace], ["open_record", "close_record"])
        assert_forward_sources(self, engine, result)

    def test_cli_can_monitor_two_entities(self):
        with TemporaryDirectory() as directory:
            main(["--monitor", "A=order:status", "--monitor", "B=order:status",
                  "--count", "1", "--min-length", "1", "--output-dir", directory])
            output = json.loads((Path(directory) / "backward_42_0000.json").read_text())
            self.assertEqual(set(output["planning"]["target_chains"]), {"A.status", "B.status"})
            self.assert_lifecycles(output)

    def test_holdings_plan_executes_on_backend_with_bound_order_ids(self):
        engine = self.sampler()
        result = engine.build(monitors=[{"id": "position", "type": "stock", "fields": ["holding"]}],
                              min_length=1, max_length=40)
        executions = [step for step in result["steps"] if step["tool"] == "execute_order"]
        order_ids = {step["entity_bindings"]["order"]: 100 + index for index, step in enumerate(executions)}
        orders = {}
        for step in executions:
            order_id = order_ids[step["entity_bindings"]["order"]]
            orders[order_id] = {"id": order_id, "status": "Pending", "symbol": "AAPL", "amount": 1,
                                "price": 100.0, "order_type": "Buy" if step["branch"] == "success_buy" else "Sell"}
        initial_holding = 10
        backend = TradingBot()
        backend._load_scenario({"orders": orders, "holdings": {"AAPL": initial_holding},
            "account_info": {"balance": 10000.0}, "stocks": {"AAPL": {"price": 100.0}},
            "authenticated": True, "market_status": "Open"}, long_context=False)
        expected = initial_holding
        for step in result["steps"]:
            if step["tool"] == "get_holdings":
                response = backend.get_holdings()
                self.assertEqual(response["holdings"]["AAPL"], expected)
            else:
                binding = step["key_bindings"]["$.args.order_id"]
                response = getattr(backend, step["tool"])(order_ids[binding["entity"]])
                self.assertNotIn("error", response, (step["tool"], response))
                if step["tool"] == "execute_order":
                    self.assertNotEqual(step["branch"], "success_sell_position_closed")
                    expected += 1 if step["branch"] == "success_buy" else -1
        self.assertEqual(backend.holdings["AAPL"], expected)


if __name__ == "__main__":
    unittest.main()
