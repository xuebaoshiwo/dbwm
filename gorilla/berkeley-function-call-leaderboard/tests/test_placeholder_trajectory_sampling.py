"""Identity isolation and lifecycle coverage without an entity registry."""

import unittest

from bfcl_eval.consistency.data_generator_v2.backward_state_knowledge import StateKnowledge
from bfcl_eval.consistency.data_generator_v2.generate_backward_trajectories import (
    BackwardSampler, DEFAULT_CATALOG, SamplingError, covers, load_inputs, load_lifecycle_rules,
)
from bfcl_eval.consistency.data_generator_v2.lifecycle_rules import LifecycleRule, Transition
from bfcl_eval.consistency.data_generator_v2.placeholder_bindings import (
    PLACEHOLDER, PlaceholderPool, strings, substitute,
)
from tests.test_backward_trajectory_sampling import assert_forward_sources, mutation, returned, spec


class PlaceholderTests(unittest.TestCase):
    def test_template_matching_preserves_distinct_instances(self):
        template = "$.state_before.items['{item_id}'].value"
        first, second = (template.replace("item_id", name) for name in ("item_id_1", "item_id_2"))
        self.assertTrue(covers(template, first))
        self.assertFalse(covers(first, second))
        self.assertTrue(covers("$.state_before.items", second))
        knowledge = StateKnowledge()
        knowledge.observe({first: {"kind": "return", "source": first}}, 1)
        self.assertIsNone(knowledge.get(second))
        knowledge.apply_mutations([mutation(second.replace("before", "after"), "$.args.value")], {}, 2)
        self.assertEqual(knowledge.get(first).anchor_id, 1)

    def test_binding_aliases_and_multiple_slots(self):
        pool = PlaceholderPool()
        path = pool.allocate("$.state_before.items['{target_item_id}'].value")
        value = ["$.state_before.items['{args.item_id}'].value",
                 "$.state_after.items['{result.item_id}'].value",
                 "$.state_before.prices['{args.symbol}']",
                 "$.state_before.items['{destination_id}'].value"]
        bindings = pool.bindings(value, value[0], path)
        self.assertEqual(bindings["args.item_id"], "item_id_1")
        self.assertEqual(bindings["result.item_id"], "item_id_1")
        self.assertEqual(bindings["args.symbol"], "symbol_1")
        self.assertEqual(bindings["destination_id"], "destination_id_1")
        self.assertIn("{item_id_1}", substitute(value, bindings)[1])

    def test_two_nontrading_instances_have_separate_sources_and_anchors(self):
        target = "$.state_before.widgets['{widget_key}'].value"
        read = target.replace("widget_key", "args.key")
        entries = [spec("read", [returned("value", read)]),
                   spec("write", mutations=[mutation(read.replace("before", "after"), read)])]
        engine = BackwardSampler({item["tool"]: item for item in entries}, {"widget": {"path": target}})
        result = engine.build(["widget", "widget"], max_writes=2, dependency_max_writes=0,
                              min_length=1, max_length=12)
        chains = result["planning"]["target_chains"]
        self.assertEqual(set(chains), {"widget_1", "widget_2"})
        self.assertEqual(len({chain["path"] for chain in chains.values()}), 2)
        for chain in chains.values():
            self.assertEqual(chain["write_count"], 2)
            identity = PLACEHOLDER.findall(chain["path"])[0]
            for index in [chain["anchor_step"], *chain["write_steps"], chain["read_step"]]:
                step = result["steps"][index - 1]
                self.assertEqual(step["placeholder_bindings"]["args.key"], identity)
                if step["write_source_requirements"]:
                    self.assertEqual(step["write_source_requirements"][0]["path"], chain["path"])
        assert_forward_sources(self, engine, result)

    def test_nested_placeholders_are_bound_independently(self):
        pool = PlaceholderPool()
        source = "$.state_before.tenants['{args.tenant}'].jobs['{args.job}'].status"
        target = pool.allocate("$.state_before.tenants['{tenant}'].jobs['{job}'].status")
        bindings = pool.bindings([source], source, target)
        self.assertEqual(substitute(source, bindings), target)
        self.assertNotEqual(bindings["args.tenant"], bindings["args.job"])

    def test_literal_braces_and_distinct_names_are_preserved(self):
        pool = PlaceholderPool()
        value = ["kind in {'buy', 'sell'}", "{owner.key}", "{owner_key}"]
        bindings = pool.bindings(value)
        self.assertEqual(set(bindings), {"owner.key", "owner_key"})
        self.assertNotEqual(bindings["owner.key"], bindings["owner_key"])
        self.assertEqual(substitute(value, bindings)[0], value[0])

    def test_unmonitored_side_effect_obeys_lifecycle(self):
        entity = "$.state_before.tasks['{task_id}'].status"
        entries = [spec("read_x", [returned("x", "$.state_before.x")]),
                   spec("read_task", [returned("status", entity)]),
                   spec("activate", mutations=[mutation(entity.replace("before", "after"), "$.args.value")]),
                   spec("finish", mutations=[mutation("$.state_after.x", "$.state_before.x"),
                                             mutation(entity.replace("before", "after"), "$.args.value")]),
                   spec("undeclared", mutations=[mutation("$.state_after.x", "$.args.value"),
                                                 mutation(entity.replace("before", "after"), "$.args.value")])]
        rule = LifecycleRule("task", frozenset({"Pending"}), (
            Transition("activate", "success_main", "Pending", "Open", 1),
            Transition("finish", "success_main", "Open", "Done", 1),
        ), entity)
        engine = BackwardSampler({item["tool"]: item for item in entries},
                                 {"x": {"path": "$.state_before.x"}}, lifecycle_rules={"task": rule})
        result = engine.build(["x"], max_writes=2, dependency_max_writes=0, min_length=1, max_length=12)
        instances = result["planning"]["lifecycle_instances"]
        self.assertEqual(len(instances), 2)
        self.assertNotIn("undeclared", {step["tool"] for step in result["steps"]})
        for path, lifecycle in instances.items():
            steps = [result["steps"][index - 1] for index in lifecycle["transition_steps"]]
            self.assertEqual([step["tool"] for step in steps], ["finish"])
            self.assertEqual(lifecycle["initial_states"], ["Open"])
            self.assertEqual(steps[0]["lifecycle_transitions"][path]["from"], "Open")
        self.assertNotIn("activate", {step["tool"] for step in result["steps"]})
        assert_forward_sources(self, engine, result)

    def test_same_instance_cannot_have_two_terminal_calls(self):
        entity = "$.state_before.tasks['{task_id_1}'].status"
        entries = [spec("read_x", [returned("x", "$.state_before.x")])]
        entries.extend(spec(tool, mutations=[mutation("$.state_after.x", "$.args.value"),
                                            mutation(entity.replace("before", "after"), "$.args.value")])
                       for tool in ("execute", "cancel"))
        rule = LifecycleRule("task", frozenset({"Pending"}), (
            Transition("execute", "success_main", "Open", "Completed", 1),
            Transition("cancel", "success_main", "Open", "Cancelled", 1),
        ), "$.state_before.tasks['{task_id}'].status")
        for tool in ("execute", "cancel"):
            specs = {item["tool"]: item for item in entries if item["tool"] in {"read_x", tool}}
            engine = BackwardSampler(specs, {"x": {"path": "$.state_before.x"}},
                                     lifecycle_rules={"task": rule})
            result = engine.build(max_writes=1, min_length=1, max_length=5)
            self.assertEqual([step["tool"] for step in result["steps"] if step["mutations"]], [tool])
            with self.assertRaises(SamplingError):
                engine.build(max_writes=2, min_length=1, max_length=5, attempts=3)
        engine = BackwardSampler({item["tool"]: item for item in entries},
                                 {"x": {"path": "$.state_before.x"}}, lifecycle_rules={"task": rule})
        with self.assertRaises(SamplingError):
            engine.build(max_writes=2, min_length=1, max_length=5, attempts=3)

    def test_monitored_order_can_start_open_without_activation(self):
        path = "$.state_before.orders['{order_id}'].status"
        entries = [spec("read", [returned("status", path)]),
                   spec("activate", mutations=[mutation(path.replace("before", "after"), "$.args.value")]),
                   spec("execute", mutations=[mutation(path.replace("before", "after"), "$.args.value")])]
        rule = LifecycleRule("order", frozenset({"Open"}), (
            Transition("activate", "success_main", "Pending", "Open", 1),
            Transition("execute", "success_main", "Open", "Completed", 1),
        ), path)
        engine = BackwardSampler({item["tool"]: item for item in entries}, {"order": {"path": path}},
                                 lifecycle_rules={"order": rule})
        result = engine.build(max_writes=3, min_length=1, max_length=10)
        self.assertEqual([step["tool"] for step in result["steps"] if step["mutations"]], ["execute"])
        self.assertEqual(result["planning"]["target_chains"]["order"]["write_count"], 1)
        assert_forward_sources(self, engine, result)

    def test_state_preserving_call_limit_is_independent_per_instance(self):
        path = "$.state_before.jobs['{job_id}'].status"
        entries = [spec("read", [returned("status", path)]),
                   spec("touch", mutations=[mutation(path.replace("before", "after"), path)])]
        rule = LifecycleRule("job", frozenset({"Ready"}), (
            Transition("touch", "success_main", "Ready", "Ready", 1),
        ), path)
        for seed in range(8):
            with self.subTest(seed=seed):
                engine = BackwardSampler({item["tool"]: item for item in entries},
                                         {"job": {"path": path}}, lifecycle_rules={"job": rule}, seed=seed)
                result = engine.build(["job", "job"], max_writes=3, dependency_max_writes=0,
                                      min_length=1, max_length=12)
                chains = result["planning"]["target_chains"]
                self.assertEqual(set(chains), {"job_1", "job_2"})
                self.assertEqual(len({chain["path"] for chain in chains.values()}), 2)
                self.assertEqual(sum(step["tool"] == "touch" for step in result["steps"]), 2)
                for chain in chains.values():
                    self.assertEqual(chain["write_count"], 1)
                    self.assertEqual(chain["write_shortfall"], 2)
                    lifecycle = result["planning"]["lifecycle_instances"][chain["path"]]
                    self.assertEqual(lifecycle["initial_states"], ["Ready"])
                    self.assertEqual(lifecycle["transition_steps"], chain["write_steps"])
                assert_forward_sources(self, engine, result)

    def test_support_anchor_reuses_target_instance(self):
        value = "$.state_before.widgets['{widget_id}'].value"
        detail = "$.state_before.widgets['{widget_id}'].detail"
        entries = [spec("read_value", [returned("value", value)]),
                   spec("read_detail", [returned("detail", detail)]),
                   spec("write_value", mutations=[mutation(value.replace("before", "after"), value)])]
        engine = BackwardSampler({item["tool"]: item for item in entries},
                                 {"widget": {"path": value}})
        result = engine.build(max_writes=1, dependency_max_writes=0, min_length=4, max_length=8)
        chain = result["planning"]["target_chains"]["widget"]
        support = next(step for step in result["steps"] if step["role"] == "support_anchor")
        self.assertEqual(support["fixes"][0]["reason"], "target_dict_source")
        self.assertEqual(support["selected_return_source"],
                         detail.replace("{widget_id}", "{widget_id_1}"))
        self.assertEqual(support["placeholder_bindings"]["widget_id"], "widget_id_1")
        self.assertIn("{widget_id_1}", chain["path"])
        assert_forward_sources(self, engine, result)

    def test_same_branch_padding_does_not_repeat_existing_order_observations(self):
        specs, targets, refinements, _ = load_inputs(DEFAULT_CATALOG)
        rules, _ = load_lifecycle_rules(DEFAULT_CATALOG, specs, targets)
        engine = BackwardSampler(specs, targets, reader_refinements=refinements,
                                 lifecycle_rules=rules, seed=20260929)
        result = engine.build(["holdings", "orders", "watch_list"], max_writes=3,
                              min_length=20, max_length=20)
        support = [step for step in result["steps"] if step["role"] == "support_anchor"]
        for step in support:
            self.assertEqual(step["fixes"][0]["reason"], "target_dict_source")
        self.assertGreater(result["planning"]["distractor_count"], 0)
        assert_forward_sources(self, engine, result)

    def test_dependency_fixers_reuse_each_independent_instance(self):
        source = "$.state_before.jobs['{args.job_id}'].cost"
        entries = [spec("read_x", [returned("x", "$.state_before.x")]),
                   spec("read_job", [returned("cost", source)]),
                   spec("write_x", mutations=[mutation("$.state_after.x", source)])]
        engine = BackwardSampler({item["tool"]: item for item in entries},
                                 {"x": {"path": "$.state_before.x"}})
        result = engine.build(max_writes=2, dependency_max_writes=0, min_length=1, max_length=10)
        writers = [step for step in result["steps"] if step["tool"] == "write_x"]
        paths = {step["write_source_requirements"][0]["path"] for step in writers}
        anchors = {fixed["path"] for step in result["steps"] for fixed in step["fixes"]
                   if fixed["kind"] == "dependency_anchor"}
        self.assertEqual(len(paths), 2)
        self.assertEqual(paths, anchors)
        assert_forward_sources(self, engine, result)

    def test_lifecycle_matches_paths_independently_of_monitor_names(self):
        path = "$.state_before.jobs['{job_id}'].status"
        entries = [spec("read", [returned("status", path)]),
                   spec("activate", mutations=[mutation(path.replace("before", "after"), "$.args.value")]),
                   spec("finish", mutations=[mutation(path.replace("before", "after"), "$.args.value")])]
        rule = LifecycleRule("job_rule", frozenset({"Pending"}), (
            Transition("activate", "success_main", "Pending", "Open", 1),
            Transition("finish", "success_main", "Open", "Done", 1),
        ), path)
        engine = BackwardSampler({item["tool"]: item for item in entries},
                                 {"first": {"path": path}, "second": {"path": path}},
                                 lifecycle_rules={"job_rule": rule})
        result = engine.build(max_writes=3, dependency_max_writes=0, min_length=1, max_length=12)
        self.assertEqual(result["planning"]["lifecycle_targets"], ["first", "second"])
        for chain in result["planning"]["target_chains"].values():
            self.assertEqual(chain["write_count"], 2)
            self.assertEqual(chain["write_shortfall"], 1)
        assert_forward_sources(self, engine, result)

    def test_two_orders_and_repeated_holdings_writes(self):
        specs, targets, refinements, _ = load_inputs(DEFAULT_CATALOG)
        rules, _ = load_lifecycle_rules(DEFAULT_CATALOG, specs, targets)
        for names in (["orders", "orders"], ["holdings"], ["holdings", "orders", "orders"]):
            with self.subTest(names=names):
                engine = BackwardSampler(specs, targets, reader_refinements=refinements,
                                         lifecycle_rules=rules, seed=42)
                result = engine.build(names, max_writes=3, dependency_max_writes=0,
                                      min_length=1, max_length=60)
                chains = result["planning"]["target_chains"]
                if "holdings" in chains:
                    self.assertGreaterEqual(chains["holdings"]["write_count"], 3)
                    order_ids = {step["placeholder_bindings"]["args.order_id"]
                                 for step in result["steps"] if step["tool"] == "execute_order"
                                 and "holdings" in step["write_targets"]}
                    self.assertEqual(len(order_ids), chains["holdings"]["write_count"])
                    activated = {step["placeholder_bindings"].get("args.order_id")
                                 for step in result["steps"] if step["tool"] == "activate_order"}
                    self.assertTrue(order_ids.isdisjoint(activated))
                for name, chain in chains.items():
                    if name.startswith("orders"):
                        self.assertEqual(chain["write_count"], 2)
                for path, lifecycle in result["planning"]["lifecycle_instances"].items():
                    state = lifecycle["initial_states"][0]
                    terminals = 0
                    for index in lifecycle["transition_steps"]:
                        transition = result["steps"][index - 1]["lifecycle_transitions"][path]
                        self.assertEqual(transition["from"], state)
                        state = transition["to"]
                        terminals += state in {"Completed", "Cancelled"}
                    self.assertLessEqual(terminals, 1)
                for step in result["steps"]:
                    for text in strings([step["mutations"], step["return_fields"], step["branch_condition"]]):
                        for token in PLACEHOLDER.findall(text):
                            self.assertRegex(token, r"_[1-9][0-9]*$")
                assert_forward_sources(self, engine, result)


if __name__ == "__main__":
    unittest.main()
