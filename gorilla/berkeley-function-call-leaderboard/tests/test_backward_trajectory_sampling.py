"""Structural tests for the independent reverse-time sampler."""

import copy
import json
from io import StringIO
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bfcl_eval.consistency.data_generator_v2.generate_backward_trajectories import (
    BackwardSampler, DEFAULT_CATALOG, LONG_CONTEXT_DISABLED, SamplingError, covers,
    load_inputs, load_lifecycle_rules, main,
)
from bfcl_eval.consistency.data_generator_v2.backward_state_knowledge import (
    StateKnowledge, mutation_writes_path, state_sources,
)
from bfcl_eval.consistency.data_generator_v2.lifecycle_rules import LifecycleRule, Transition, parse_lifecycle_rules


def relation(path, *, reversible=True):
    return {"path": path, "transform": "copy" if reversible else "filter",
            "recoverability": "exact" if reversible else "partial", "given": [], "reason": "Test source."}


def returned(name, *sources, reversible=True):
    if len(sources) == 1 and reversible:
        return {"path": "$.result." + name, "source_kind": "state_direct", "source": sources[0],
                "logic": "Copied.", "state_source_relations": [relation(sources[0])]}
    relations = [relation(source, reversible=reversible) for source in sources]
    if reversible:
        for item in relations:
            item.update(transform="arithmetic", recoverability="conditional",
                        given=[source for source in sources if source != item["path"]])
    return {"path": "$.result." + name, "source_kind": "derived", "value_from": list(sources),
            "logic": "Combined sources.", "state_source_relations": relations}


def mutation(target, *sources, operation="set"):
    return {"branch": "success_main", "target": target, "operation": operation,
            "value_from": list(sources), "state_source_relations": [relation(source) for source in sources],
            "logic": "Test mutation."}


def observed(name, path, *, requires=(), reversible=True):
    item = relation(path, reversible=reversible)
    if requires:
        item.update(transform="arithmetic", recoverability="conditional", given=list(requires))
    return {"path": "$.result." + name, "source_kind": "constant", "value": "Ready",
            "logic": "A returned value also observes persistent post-state.",
            "state_source_relations": [], "state_observation_relations": [item]}


def spec(tool, fields=(), mutations=()):
    fields = list(fields) or [{"path": "$.result.status", "source_kind": "constant", "value": "ok",
                              "state_source_relations": [], "logic": "Constant."}]
    return {"schema_version": "1.1", "tool": tool,
            "branches": [{"id": "success_main", "if": "otherwise", "uses": [], "description": "Test branch."}],
            "returns": [{"branch": "success_main", "fields": fields}],
            "mutations": list(mutations), "warnings": []}


def sampler(entries, targets=None, seed=42):
    return BackwardSampler({entry["tool"]: entry for entry in entries},
                           targets or {"x": {"path": "$.state_before.x"}}, seed=seed)


def basic_entries():
    return [spec("read_x", [returned("x", "$.state_before.x")]),
            spec("write_x", mutations=[mutation("$.state_after.x", "$.state_before.x", "$.args.amount")]),
            spec("unrelated", [returned("other", "$.state_before.other")])]


def provisional_steps(engine, sequence):
    nodes = {node.tool: node for node in engine.nodes}
    steps = []
    for index, (tool, kind, path) in enumerate(sequence, 1):
        node = nodes[tool]
        role = "write" if kind == "write" else "read" if kind == "final_read" else "anchor"
        fixed = [] if kind == "write" else [{"path": path, "kind": kind, "depth": 0,
                    "evidence": engine.proof(engine.closure(node), path)}]
        selected = node.matching_mutation_indices(path)[0] if kind == "write" else None
        steps.append({"_id": index, "tool": tool, "branch": node.branch["id"],
                      "role": role, "roles": [role], "fixes": fixed, "write_targets": [],
                      "selected_mutation_index": selected,
                      "mutations": copy.deepcopy(node.mutations),
                      "write_source_requirements": [{"path": source, "resolution": "earlier_chain",
                                                     "evidence": None}
                          for mutation_item in ([node.mutations[selected]] if selected is not None else [])
                          for source in mutation_item["value_from"]
                          if source.startswith("$.state_before")]})
    return steps


def assert_forward_sources(test, engine, result):
    """Audit structural knowledge independently in chronological order."""
    nodes = {(node.tool, node.branch["id"]): node for node in engine.nodes}
    knowledge = StateKnowledge()
    for step in result["steps"]:
        node = engine.bind_node(nodes[step["tool"], step["branch"]],
                                bindings=step.get("placeholder_bindings", {}))
        referenced = node.referenced_paths() | {fixed["path"] for fixed in step["fixes"]}
        proofs = engine.closure(node, knowledge.known_paths(referenced))
        selected = step["selected_mutation_index"]
        selected_mutation = node.mutations[selected] if selected is not None else {}
        required = set(state_sources(selected_mutation)) if selected is not None else set()
        required.update(entry["path"] for entry in selected_mutation.get("target_identity_sources", []))
        test.assertEqual(set(required), {item["path"] for item in step["write_source_requirements"]})
        for source in required:
            test.assertTrue(engine.proof(proofs, source),
                            f"Unfixed selected source {source} at step {step['index']} ({node.tool})")
        for fixed in step["fixes"]:
            path = fixed["path"]
            if fixed.get("phase") == "after":
                path = path.replace("$.state_before", "$.state_after", 1)
            test.assertTrue(engine.proof(proofs, path),
                            f"Unjustified fixed field {fixed['path']} at step {step['index']}")
        knowledge.observe(proofs, step["index"])
        knowledge.apply_mutations(node.mutations, proofs, step["index"])


class StateKnowledgeTests(unittest.TestCase):
    def test_unknown_child_blocks_parent_but_preserves_unaffected_siblings(self):
        knowledge = StateKnowledge()
        parent, balance, owner = ("$.state_before.account", "$.state_before.account.balance",
                                  "$.state_before.account.owner")
        knowledge.observe({parent: {"kind": "return", "source": parent}}, 1)
        status = knowledge.apply_mutations([mutation("$.state_after.account.balance", "$.state_before.hidden")], {}, 2)
        self.assertFalse(status[0]["sources_fixed"])
        self.assertIsNone(knowledge.get(parent))
        self.assertIsNone(knowledge.get(balance))
        self.assertEqual(knowledge.get(owner).anchor_id, 1)
        self.assertEqual(knowledge.known_paths((parent, balance, owner)), {owner})
        knowledge.observe({balance: {"kind": "return", "source": balance}}, 3)
        self.assertEqual(knowledge.get(balance).anchor_id, 3)
        self.assertEqual(knowledge.get(parent).anchor_id, 1)
        self.assertIn(3, knowledge.get(parent).observation_ids)

    def test_observed_child_does_not_restore_unknown_entire_parent(self):
        knowledge = StateKnowledge()
        knowledge.apply_mutations([mutation("$.state_after.account", "$.state_before.hidden")], {}, 1)
        balance = "$.state_before.account.balance"
        knowledge.observe({balance: {"kind": "return", "source": balance}}, 2)
        self.assertEqual(knowledge.get(balance).anchor_id, 2)
        self.assertIsNone(knowledge.get("$.state_before.account"))
        self.assertIsNone(knowledge.get("$.state_before.account.owner"))

    def test_source_fixed_parent_write_preserves_child_anchor(self):
        knowledge = StateKnowledge()
        balance = "$.state_before.account.balance"
        knowledge.observe({balance: {"kind": "return", "source": balance}}, 1)
        knowledge.apply_mutations([mutation("$.state_after.account", "$.args.account")], {}, 2)
        self.assertEqual(knowledge.get(balance).anchor_id, 1)
        self.assertEqual(knowledge.get(balance).mutation_ids, (2,))

    def test_same_branch_mutations_do_not_confuse_pre_and_post_inputs(self):
        knowledge = StateKnowledge()
        statuses = knowledge.apply_mutations([
            mutation("$.state_after.x", "$.args.value"),
            mutation("$.state_after.y", "$.state_before.x"),
        ], {}, 1)
        self.assertTrue(statuses[0]["post_state_known"])
        self.assertFalse(statuses[1]["sources_fixed"])
        self.assertFalse(statuses[1]["post_state_known"])

    def test_parent_observation_fixes_a_previously_only_computed_child(self):
        knowledge = StateKnowledge()
        knowledge.apply_mutations([mutation("$.state_after.account.balance", "$.args.value")], {}, 1)
        balance = "$.state_before.account.balance"
        self.assertIsNone(knowledge.get(balance).anchor_id)
        knowledge.observe({"$.state_after.account": {"kind": "return", "source": "$.state_after.account"}},
                          2, phase="after")
        self.assertEqual(knowledge.get(balance).anchor_id, 2)
        self.assertEqual(knowledge.get(balance).anchor_phase, "after")


class BackwardSamplerTests(unittest.TestCase):
    def test_creation_return_is_a_post_state_anchor_without_old_collection_fixing(self):
        x = "$.state_before.records['{record_id}'].status"
        created = observed("status", "$.state_after.records['{result.record_id}'].status")
        entries = [spec("read_record", [returned("status", x)]),
                   spec("create_record", [created], [mutation(
                       "$.state_after.records['{result.record_id}']", "$.state_before.records")]),
                   spec("update_record", mutations=[mutation(x.replace("state_before", "state_after"), x)])]
        targets = {"record": {"path": x}}
        lifecycle = LifecycleRule("record", frozenset({"Ready"}), (
            Transition("create_record", "success_main", "Absent", "Ready", 1),
            Transition("update_record", "success_main", "Ready", "Ready", 1),
        ))
        found = False
        for seed in range(10):
            engine = BackwardSampler({item["tool"]: item for item in entries}, targets,
                                     lifecycle_rules={"record": lifecycle}, seed=seed)
            result = engine.build(max_writes=3, min_length=1, max_length=10)
            chain = result["planning"]["target_chains"]["record"]
            self.assertEqual(chain["write_count"], 1)
            anchor = result["steps"][chain["anchor_step"] - 1]
            if anchor["tool"] == "create_record":
                found = True
                self.assertEqual(chain["anchor_phase"], "after")
                self.assertNotIn(chain["anchor_step"], chain["write_steps"])
                self.assertEqual(anchor["write_source_requirements"], [])
                self.assertFalse(anchor["mutation_knowledge"][0]["sources_fixed"])
                self.assertTrue(any(fixed["phase"] == "after" for fixed in anchor["fixes"]))
            assert_forward_sources(self, engine, result)
        self.assertTrue(found)

    def test_observation_co_source_is_fixed_by_an_earlier_chain(self):
        entries = [spec("read_x", [returned("x", "$.state_before.x")]),
                   spec("create_x", [observed("combined", "$.state_after.x", requires=["$.state_before.fee"])],
                        [mutation("$.state_after.x", "$.state_before.unobservable")]),
                   spec("read_fee", [returned("fee", "$.state_before.fee")]),
                   spec("write_x", mutations=[mutation("$.state_after.x", "$.state_before.x")])]
        found = False
        for seed in range(10):
            engine = sampler(entries, seed=seed)
            result = engine.build(max_writes=1, dependency_max_writes=0, min_length=1, max_length=12)
            if any(step["tool"] == "create_x" for step in result["steps"]):
                found = True
                self.assertTrue(any(step["tool"] == "read_fee" for step in result["steps"]))
            assert_forward_sources(self, engine, result)
        self.assertTrue(found)

    def test_post_observation_neither_fixes_old_state_nor_partial_observations(self):
        opaque = mutation("$.state_after.x", "$.state_before.hidden")
        opaque["state_source_relations"][0].update(transform="unknown", recoverability="none")
        entries = [spec("create", [observed("new_x", "$.state_after.x")], [opaque]),
                   spec("partial", [observed("summary", "$.state_after.z", reversible=False)])]
        engine = sampler(entries)
        node = next(node for node in engine.nodes if node.tool == "create")
        proofs = engine.closure(node)
        self.assertIsNotNone(engine.proof(proofs, "$.state_after.x"))
        self.assertIsNone(engine.proof(proofs, "$.state_before.x"))
        self.assertIsNone(engine.proof(proofs, "$.state_before.hidden"))
        self.assertTrue(engine.fix_options("$.state_before.x", 2))
        self.assertFalse(engine.fix_options("$.state_before.z", 0))

    def test_place_order_observations_are_independent_and_do_not_recover_old_mapping(self):
        specs, targets, refinements, _ = load_inputs(DEFAULT_CATALOG)
        engine = BackwardSampler(specs, targets, reader_refinements=refinements)
        node = next(node for node in engine.nodes if node.tool == "place_order")
        proofs = engine.closure(node)
        self.assertIsNotNone(engine.proof(proofs, "$.state_after.order_counter"))
        self.assertIsNotNone(engine.proof(proofs, "$.state_after.orders['{result.order_id}'].id"))
        self.assertIsNone(engine.proof(proofs, "$.state_before.order_counter"))
        self.assertIsNone(engine.proof(proofs, "$.state_before.orders"))
        self.assertIsNone(engine.proof(proofs, "$.state_after.orders"))
        self.assertTrue(any(option[0].tool == "place_order"
                            for option in engine.fix_options(targets["orders"]["path"], 0)))

    def test_lifecycle_filters_order_and_accepts_unavoidable_shortfall(self):
        entries = [spec("read_x", [returned("x", "$.state_before.x")])]
        entries.extend(spec(name, mutations=[mutation("$.state_after.x", "$.state_before.x")])
                       for name in ("activate", "execute", "cancel"))
        rules = {"x": LifecycleRule("x", frozenset({"Pending"}), (
            Transition("activate", "success_main", "Pending", "Open"),
            Transition("execute", "success_main", "Open", "Completed"),
            Transition("cancel", "success_main", "Open", "Cancelled"),
        ))}
        for seed in range(10):
            engine = BackwardSampler({item["tool"]: item for item in entries},
                                     {"x": {"path": "$.state_before.x"}},
                                     lifecycle_rules=rules, seed=seed)
            result = engine.build(max_writes=3, min_length=1, max_length=10)
            writes = [step["tool"] for step in result["steps"] if step["role"] == "write"]
            self.assertEqual(writes[0], "activate")
            self.assertIn(writes[1], {"execute", "cancel"})
            self.assertEqual(len(writes), 2)
            chain = result["planning"]["target_chains"]["x"]
            self.assertEqual((chain["min_write_count"], chain["write_count"], chain["write_shortfall"]), (3, 2, 1))
            assert_forward_sources(self, engine, result)

    def test_lifecycle_rules_validate_branch_writes(self):
        entries = {item["tool"]: item for item in basic_entries()}
        data = {"version": 1, "targets": {"x": {"initial_states": ["Before"], "transitions": [
            {"tool": "write_x", "branch": "success_main", "from": "Before", "to": "After"}]}}}
        self.assertIn("x", parse_lifecycle_rules(data, entries, {"x": {"path": "$.state_before.x"}}))
        data["targets"]["x"]["transitions"][0]["tool"] = "unrelated"
        with self.assertRaises(ValueError):
            parse_lifecycle_rules(data, entries, {"x": {"path": "$.state_before.x"}})

    def test_lifecycle_limits_a_state_preserving_writer(self):
        entries = [spec("read_x", [returned("x", "$.state_before.x")]),
                   spec("write_x", mutations=[mutation("$.state_after.x", "$.state_before.x")])]
        specs = {item["tool"]: item for item in entries}
        targets = {"x": {"path": "$.state_before.x"}}
        for max_calls, expected in ((1, 1), (None, 3)):
            rule = LifecycleRule("x", frozenset({"Ready"}), (
                Transition("write_x", "success_main", "Ready", "Ready", max_calls),
            ))
            result = BackwardSampler(specs, targets, lifecycle_rules={"x": rule}).build(
                max_writes=3, min_length=1, max_length=10)
            self.assertEqual(result["planning"]["target_chains"]["x"]["write_count"], expected)

    def test_real_order_lifecycle_rules_keep_one_terminal_action(self):
        specs, targets, refinements, _ = load_inputs(DEFAULT_CATALOG)
        rules, _ = load_lifecycle_rules(DEFAULT_CATALOG, specs, targets)
        self.assertIn("orders", rules)
        found_creation_anchor = False
        for seed in range(10):
            engine = BackwardSampler(specs, targets, reader_refinements=refinements,
                                     lifecycle_rules=rules, seed=seed)
            result = engine.build(max_writes=3, min_length=1, max_length=60)
            chain = result["planning"]["target_chains"]["orders"]
            tools = [result["steps"][index - 1]["tool"] for index in chain["write_steps"]]
            self.assertEqual(tools[0], "activate_order")
            self.assertIn(tools[1], {"execute_order", "cancel_order"})
            self.assertEqual((chain["write_count"], chain["write_shortfall"]), (2, 1))
            anchor = result["steps"][chain["anchor_step"] - 1]
            if anchor["tool"] == "place_order":
                found_creation_anchor = True
                self.assertEqual(chain["anchor_phase"], "after")
                self.assertEqual(anchor["write_source_requirements"], [])
            assert_forward_sources(self, engine, result)
        self.assertTrue(found_creation_anchor)

    def test_unfixable_side_effect_does_not_expand_selected_writer_dependencies(self):
        x, hidden = "$.state_before.x", "$.state_before.hidden_fee"
        side_effect = mutation("$.state_after.y", hidden)
        entries = [basic_entries()[0], spec("write_x", mutations=[
            mutation("$.state_after.x", x, "$.args.amount"), side_effect])]
        engine = sampler(entries)
        result = engine.build(max_writes=2, dependency_max_writes=0, min_length=1, max_length=4)
        self.assertEqual([step["tool"] for step in result["steps"]], ["read_x", "write_x", "write_x", "read_x"])
        for step in result["steps"]:
            if step["tool"] == "write_x":
                self.assertEqual(step["selected_mutation_index"], 0)
                self.assertEqual({item["path"] for item in step["write_source_requirements"]}, {x})
                self.assertEqual(len(step["mutations"]), 2)
                self.assertFalse(step["mutation_knowledge"][1]["sources_fixed"])
                self.assertFalse(step["mutation_knowledge"][1]["post_state_known"])
        self.assertEqual(len(result["planning"]["unfixed_side_effects"]), 2)
        self.assertEqual(result["planning"]["unresolved_structural_sources"], [])
        candidate = result["planning"]["writer_candidates_by_target"]["x"][0]
        self.assertEqual(candidate["unfixable_write_sources"], [])
        assert_forward_sources(self, engine, result)

    def test_real_holdings_writer_does_not_add_balance_anchor(self):
        specs, _, refinements, _ = load_inputs(DEFAULT_CATALOG)
        target = "$.state_before.holdings['{order.symbol}']"
        engine = BackwardSampler(specs, {"holdings": {"path": target}}, reader_refinements=refinements)
        result = engine.build(max_writes=1, dependency_max_writes=0, min_length=1, max_length=5)
        writer = next(step for step in result["steps"] if step["tool"] == "execute_order")
        identity_reader = next(step for step in result["steps"] if step["tool"] == "get_order_details")
        self.assertLess(identity_reader["index"], writer["index"])
        self.assertEqual(result["steps"][-1]["tool"], "get_holdings")
        self.assertNotIn("get_account_info", {step["tool"] for step in result["steps"]})
        selected = writer["mutations"][writer["selected_mutation_index"]]
        self.assertEqual(selected["target"], "$.state_after.holdings['{order.symbol_1}']")
        self.assertNotIn("$.state_before.account_info.balance",
                         {item["path"] for item in writer["write_source_requirements"]})
        identity = next(item for item in writer["write_source_requirements"] if item["path"].endswith(".symbol"))
        self.assertEqual(identity["target_identity_relations"][0]["placeholder"], "order.symbol_1")
        balance = next(status for status in writer["mutation_knowledge"]
                       if status["target"] == "$.state_after.account_info.balance")
        self.assertFalse(balance["sources_fixed"])
        self.assertFalse(balance["post_state_known"])
        assert_forward_sources(self, engine, result)

    def test_support_anchors_fill_before_unrelated_distractors(self):
        balance, account_id = "$.state_before.account.balance", "$.state_before.account.account_id"
        entries = [spec("read_balance", [returned("balance", balance)]),
                   spec("write_balance", mutations=[mutation("$.state_after.account.balance", balance)]),
                   spec("read_account_id", [returned("account_id", account_id)])]
        engine = sampler(entries, {"balance": {"path": balance}})
        result = engine.build(max_writes=1, min_length=4, max_length=8)
        self.assertEqual(result["planning"]["core_length"], 3)
        self.assertEqual(result["planning"]["support_anchor_count"], 1)
        self.assertEqual(result["planning"]["distractor_count"], 0)
        support = next(step for step in result["steps"] if step["role"] == "support_anchor")
        self.assertEqual(support["selected_return_field"], "$.result.account_id")
        self.assertEqual(support["selected_return_source"], account_id)
        self.assertEqual(support["mutations"], [])
        assert_forward_sources(self, engine, result)

    def test_support_anchor_pool_is_finite_then_distractors_are_used(self):
        balance, account_id = "$.state_before.account.balance", "$.state_before.account.account_id"
        entries = [spec("read_balance", [returned("balance", balance)]),
                   spec("write_balance", mutations=[mutation("$.state_after.account.balance", balance)]),
                   spec("read_account_id", [returned("account_id", account_id)]),
                   spec("unrelated", [returned("other", "$.state_before.other")])]
        result = sampler(entries, {"balance": {"path": balance}}).build(max_writes=1, min_length=6, max_length=8)
        self.assertEqual(result["planning"]["support_anchor_count"], 1)
        self.assertEqual(result["planning"]["distractor_count"], 2)
        self.assertEqual(sum(step["role"] == "support_anchor" for step in result["steps"]), 1)
        self.assertEqual(sum(step["role"] == "distractor" for step in result["steps"]), 2)

    def test_unrelated_mutation_can_be_a_distractor(self):
        entries = [*basic_entries()[:2],
                   spec("write_y", mutations=[mutation("$.state_after.y", "$.args.value")])]
        engine = sampler(entries)
        result = engine.build(max_writes=1, min_length=4, max_length=4)
        distractor, = (step for step in result["steps"] if step["role"] == "distractor")
        self.assertEqual(distractor["tool"], "write_y")
        self.assertEqual(distractor["mutations"][0]["target"], "$.state_after.y")
        self.assertTrue(distractor["mutation_knowledge"][0]["sources_fixed"])
        self.assertEqual(distractor["write_targets"], [])
        assert_forward_sources(self, engine, result)

    def test_distractor_writer_uses_a_fresh_entity(self):
        path = "$.state_before.orders['{order_id_1}'].status"
        entries = [spec("read_order", [returned("status", path)]),
                   spec("write_order", mutations=[mutation(path.replace("state_before", "state_after"),
                                                            "$.args.status")]),
                   spec("write_other_order", mutations=[mutation(
                       "$.state_after.orders['{args.order_id}'].price", "$.args.price")])]
        engine = sampler(entries, {"order": {"path": path}})
        result = engine.build(max_writes=1, min_length=4, max_length=4)
        distractor, = (step for step in result["steps"] if step["role"] == "distractor")
        self.assertEqual(distractor["tool"], "write_other_order")
        self.assertEqual(distractor["placeholder_bindings"]["args.order_id"], "order_id_2")
        self.assertIn("{order_id_2}", distractor["mutations"][0]["target"])
        assert_forward_sources(self, engine, result)

    def test_distractor_cannot_write_an_existing_entity_or_parent_field(self):
        path = "$.state_before.orders['{order_id_1}'].status"
        entries = [spec("read_order", [returned("status", path)]),
                   spec("write_order", mutations=[mutation(path.replace("state_before", "state_after"),
                                                            "$.args.status")]),
                   spec("write_same_order_price", mutations=[mutation(
                       "$.state_after.orders['{order_id_1}'].price", "$.args.price")]),
                   spec("write_all_orders", mutations=[mutation("$.state_after.orders", "$.args.orders")])]
        with self.assertRaisesRegex(SamplingError, "No unrelated branch"):
            sampler(entries, {"order": {"path": path}}).build(
                max_writes=1, min_length=4, max_length=4, attempts=2)

    def test_distractor_cannot_change_a_core_branch_condition(self):
        entries = [*basic_entries()[:2],
                   spec("change_auth", mutations=[mutation("$.state_after.authenticated", "$.args.value")])]
        entries[0]["branches"][0]["uses"] = ["$.state_before.authenticated"]
        with self.assertRaisesRegex(SamplingError, "No unrelated branch"):
            sampler(entries).build(max_writes=1, min_length=4, max_length=4, attempts=2)

    def test_derived_address_cannot_write_inside_a_protected_mapping(self):
        monitored = "$.state_before.bins['{key_1}'].quantity"
        derived = mutation("$.state_after.bins['{job.symbol}'].price", "$.args.price")
        derived["target_identity_sources"] = [{
            "placeholder": "job.symbol",
            "path": "$.state_before.jobs['{args.job_id}'].symbol",
            "logic": "The bin key comes from the job symbol.",
        }]
        entries = [spec("read_bin", [returned("quantity", monitored)]),
                   spec("write_bin", mutations=[mutation(
                       monitored.replace("state_before", "state_after"), "$.args.quantity")]),
                   spec("write_derived_bin", mutations=[derived]),
                   spec("read_other", [returned("other", "$.state_before.other")])]
        result = sampler(entries, {"bin": {"path": monitored}}).build(
            max_writes=1, min_length=4, max_length=4, linked_distractors=True)
        self.assertEqual([step["tool"] for step in result["steps"]
                          if step["role"] == "distractor"], ["read_other"])

    def test_linked_distractors_are_opt_in_and_preserve_the_core(self):
        entries = [*basic_entries()[:2],
                   spec("read_z", [returned("z", "$.state_before.z")]),
                   spec("write_z", mutations=[mutation("$.state_after.z", "$.state_before.q")]),
                   spec("write_q", mutations=[mutation("$.state_after.q", "$.args.value")])]
        options = {"max_writes": 1, "min_length": 6, "max_length": 6}
        default = sampler(entries).build(**options)
        explicit_off = sampler(entries).build(**options, linked_distractors=False)
        linked = sampler(entries).build(**options, linked_distractors=True)
        self.assertEqual(default, explicit_off)
        self.assertEqual(linked["planning"]["distractor_chain_lengths"], [3])
        chain = [step for step in linked["steps"] if step.get("distractor_chain_id") == 1]
        self.assertEqual([step["tool"] for step in chain], ["write_q", "write_z", "read_z"])
        self.assertEqual([step.get("distractor_link_source") for step in chain],
                         [None, "$.state_before.q", "$.state_before.z"])
        core = lambda result: [(step["tool"], step["branch"], step["placeholder_bindings"])
                               for step in result["steps"] if step["role"] != "distractor"]
        self.assertEqual(core(default), core(linked))
        assert_forward_sources(self, sampler(entries), linked)

    def test_linked_distractors_respect_a_fresh_entity_lifecycle(self):
        status = "$.state_before.tasks['{args.task_id}'].status"
        entries = [*basic_entries()[:2], spec("read_task", [returned("status", status)]),
                   spec("start_task", mutations=[mutation(status.replace("state_before", "state_after"),
                                                            "$.args.next_status")]),
                   spec("create_task", mutations=[mutation(
                       "$.state_after.tasks['{args.task_id}']", operation="insert")])]
        entries[3]["branches"][0]["uses"] = [status]
        rule = LifecycleRule("task", frozenset({"Absent"}), (
            Transition("create_task", "success_main", "Absent", "Pending", 1),
            Transition("start_task", "success_main", "Pending", "Open", 1),
        ), path="$.state_before.tasks['{task_id}'].status")
        engine = BackwardSampler({entry["tool"]: entry for entry in entries},
                                 {"x": {"path": "$.state_before.x"}},
                                 lifecycle_rules={"task": rule})
        result = engine.build(max_writes=1, min_length=6, max_length=6,
                              linked_distractors=True)
        chain = [step for step in result["steps"] if step.get("distractor_chain_id") == 1]
        self.assertEqual([step["tool"] for step in chain],
                         ["create_task", "start_task", "read_task"])
        self.assertEqual({step["placeholder_bindings"]["args.task_id"] for step in chain},
                         {"task_id_1"})
        lifecycle = result["planning"]["lifecycle_instances"][
            "$.state_before.tasks['{task_id_1}'].status"]
        self.assertEqual(lifecycle["initial_states"], ["Absent"])
        self.assertEqual([result["steps"][index - 1]["tool"]
                          for index in lifecycle["transition_steps"]],
                         ["create_task", "start_task"])
        assert_forward_sources(self, engine, result)

    def test_lifecycle_link_uses_the_same_entity_not_its_parent_mapping(self):
        specs, targets, refinements, _ = load_inputs(DEFAULT_CATALOG)
        rules, _ = load_lifecycle_rules(DEFAULT_CATALOG, specs, targets)
        engine = BackwardSampler(specs, targets, reader_refinements=refinements,
                                 lifecycle_rules=rules, seed=2712018334673078314)
        cancel = next(node for node in engine.nodes
                      if (node.tool, node.branch["id"]) == ("cancel_order", "success_cancelled"))
        bound = engine.bind_node(cancel, bindings={"args.order_id": "order_id_1"})
        sources = engine._distractor_link_sources(bound)
        self.assertIn("$.state_before.orders['{order_id_1}'].status", sources)
        self.assertNotIn("$.state_before.orders", sources)

        result = engine.build(["watch_list"], max_writes=2, min_length=7, max_length=60,
                              linked_distractors=True)
        chain = [step for step in result["steps"] if step.get("distractor_chain_id") == 1]
        self.assertEqual([step["tool"] for step in chain], ["activate_order", "cancel_order"])
        self.assertEqual({step["placeholder_bindings"]["args.order_id"] for step in chain},
                         {"order_id_1"})
        self.assertEqual(chain[1]["distractor_link_source"],
                         "$.state_before.orders['{order_id_1}'].status")

    def test_linked_distractors_use_a_new_entity_in_the_same_mapping(self):
        monitored = "$.state_before.jobs['{job_id_1}'].status"
        other = "$.state_before.jobs['{args.job_id}'].price"
        entries = [spec("read_job", [returned("status", monitored)]),
                   spec("write_job", mutations=[mutation(
                       monitored.replace("state_before", "state_after"), "$.args.status")]),
                   spec("read_price", [returned("price", other)]),
                   spec("write_price", mutations=[mutation(
                       other.replace("state_before", "state_after"), "$.args.price")])]
        result = sampler(entries, {"job": {"path": monitored}}).build(
            max_writes=1, min_length=6, max_length=6, linked_distractors=True)
        chain = [step for step in result["steps"] if step.get("distractor_chain_id") == 1]
        self.assertEqual([step["tool"] for step in chain], ["write_price", "read_price"])
        self.assertEqual({step["placeholder_bindings"]["args.job_id"] for step in chain},
                         {"job_id_2"})

    def test_random_padding_does_not_write_a_linked_chain_field(self):
        entries = [*basic_entries()[:2],
                   spec("read_z", [returned("z", "$.state_before.z")]),
                   spec("write_z", mutations=[mutation("$.state_after.z", "$.state_before.q")]),
                   spec("write_q", mutations=[mutation("$.state_after.q", "$.args.value")]),
                   spec("read_other", [returned("other", "$.state_before.other")])]
        result = sampler(entries).build(max_writes=1, min_length=7, max_length=7,
                                        linked_distractors=True)
        self.assertEqual(result["planning"]["distractor_chain_lengths"], [3])
        self.assertEqual([step["tool"] for step in result["steps"]
                          if step["role"] == "distractor" and "distractor_chain_id" not in step],
                         ["read_other"])

    def test_existing_branch_observes_all_return_fields_without_duplicate_support_call(self):
        balance, account_id = "$.state_before.account.balance", "$.state_before.account.account_id"
        entries = [spec("read_account", [returned("balance", balance), returned("account_id", account_id)]),
                   spec("write_balance", mutations=[mutation("$.state_after.account.balance", balance)]),
                   spec("unrelated", [returned("other", "$.state_before.other")])]
        result = sampler(entries, {"balance": {"path": balance}}).build(max_writes=1, min_length=4, max_length=8)
        self.assertEqual(result["planning"]["support_anchor_count"], 0)
        self.assertFalse(any(step["role"] == "support_anchor" for step in result["steps"]))
        self.assertEqual(sum(step["tool"] == "read_account" for step in result["steps"]), 2)

    def test_reader_only_requires_sources_of_its_selected_return_field(self):
        x, z = "$.state_before.x", "$.state_before.z"
        entries = [spec("read_sum", [returned("sum", x, z),
                                    returned("unrelated_sum", "$.state_before.y", "$.state_before.hidden")],
                        [mutation("$.state_after.side_effect", "$.state_before.unreadable")]),
                   spec("read_z", [returned("z", z)]), basic_entries()[1]]
        engine = sampler(entries)
        result = engine.build(max_writes=1, dependency_max_writes=0, min_length=1, max_length=10)
        for step in result["steps"]:
            if step["tool"] == "read_sum":
                self.assertEqual(step["selected_return_field"], "$.result.sum")
                self.assertEqual(step["write_source_requirements"], [])
                self.assertIsNone(step["selected_mutation_index"])
            self.assertNotIn(step.get("field"), {"$.state_before.y", "$.state_before.hidden", "$.state_before.unreadable"})
        self.assertTrue(any(step["tool"] == "read_z" for step in result["steps"]))
        assert_forward_sources(self, engine, result)

    def test_writer_selects_one_matching_mutation_not_all_matching_mutations(self):
        balance = "$.state_before.account.balance"
        entries = [spec("read_balance", [returned("balance", balance)]),
                   spec("write_account", mutations=[
                       mutation("$.state_after.account", "$.state_before.unreadable_account"),
                       mutation("$.state_after.account.balance", "$.args.amount")])]
        engine = sampler(entries, {"balance": {"path": balance}})
        result = engine.build(max_writes=1, min_length=1, max_length=3)
        writer = next(step for step in result["steps"] if step["tool"] == "write_account")
        self.assertEqual(writer["selected_mutation_index"], 1)
        self.assertEqual(writer["write_source_requirements"], [])
        self.assertFalse(writer["mutation_knowledge"][0]["sources_fixed"])
        self.assertTrue(writer["mutation_knowledge"][1]["sources_fixed"])
        assert_forward_sources(self, engine, result)

    def test_unknown_side_effect_prevents_false_anchor_reuse(self):
        x, y = "$.state_before.x", "$.state_before.y"
        entries = [basic_entries()[0], spec("read_y", [returned("y", y)]),
                   spec("write_x", mutations=[mutation("$.state_after.x", x),
                                               mutation("$.state_after.y", "$.state_before.hidden")])]
        engine = sampler(entries)
        sequence = [("read_y", "dependency_anchor", y), ("read_x", "initial_anchor", x),
                    ("write_x", "write", x), ("read_y", "dependency_anchor", y),
                    ("read_x", "final_read", x)]
        chain = {"x": {"initial_id": 2, "final_id": 5, "write_ids": [3]}}
        steps, removed = engine._reuse_anchors(provisional_steps(engine, sequence), chain, {"x": x})
        self.assertFalse(removed)
        self.assertIn(4, [step["_id"] for step in steps])
        self.assertFalse(steps[2]["mutation_knowledge"][1]["post_state_known"])

    def test_post_write_observation_reuses_anchor_without_counting_its_own_write(self):
        x, y = "$.state_before.x", "$.state_before.y"
        opaque = mutation("$.state_after.y", "$.state_before.hidden")
        opaque["state_source_relations"][0]["recoverability"] = "none"
        entries = [basic_entries()[0], spec("read_y", [returned("y", y)]),
                   spec("write_x", [returned("new_y", "$.state_after.y")],
                        [mutation("$.state_after.x", x), opaque]),
                   spec("write_y", mutations=[mutation("$.state_after.y", y)])]
        engine = sampler(entries)
        sequence = [("read_x", "dependency_anchor", x), ("write_x", "write", x),
                    ("read_y", "initial_anchor", y), ("write_y", "write", y),
                    ("read_y", "final_read", y)]
        chain = {"y": {"initial_id": 3, "final_id": 5, "write_ids": [4]}}
        steps, removed = engine._reuse_anchors(provisional_steps(engine, sequence), chain, {"y": y})
        self.assertEqual([step["original_step_id"] for step in removed], [3])
        self.assertEqual(chain["y"]["initial_id"], 2)
        self.assertEqual(chain["y"]["initial_phase"], "after")
        self.assertEqual(chain["y"]["write_ids"], [4])
        self.assertFalse(steps[1]["mutation_knowledge"][1]["sources_fixed"])
        self.assertTrue(steps[1]["mutation_knowledge"][1]["post_state_known"])
        for index, step in enumerate(steps, 1):
            step["index"] = index
        assert_forward_sources(self, engine, {"steps": steps})

    def test_prior_anchors_survive_writes_and_replace_later_initial_anchors(self):
        b, h = "$.state_before.balance", "$.state_before.history"
        entries = [spec("read_b", [returned("balance", b)]), spec("read_h", [returned("history", h)]),
                   spec("write_b", mutations=[mutation("$.state_after.balance", b, "$.args.amount")]),
                   spec("write_bh", mutations=[mutation("$.state_after.balance", b, "$.args.amount"),
                                               mutation("$.state_after.history", h, "$.args.amount")])]
        engine = sampler(entries, {"b": {"path": b}, "h": {"path": h}})
        sequence = [("read_b", "dependency_anchor", b), ("write_b", "write", b),
                    ("read_b", "dependency_anchor", b), ("read_h", "dependency_anchor", h),
                    ("write_bh", "write", b), ("read_h", "initial_anchor", h),
                    ("read_b", "initial_anchor", b), ("write_bh", "write", h),
                    ("write_bh", "write", h), ("read_h", "final_read", h), ("read_b", "final_read", b)]
        chain = {"b": {"initial_id": 7, "final_id": 11, "write_ids": [8, 9]},
                 "h": {"initial_id": 6, "final_id": 10, "write_ids": [8, 9]}}
        steps, removed = engine._reuse_anchors(provisional_steps(engine, sequence), chain, {"b": b, "h": h})
        self.assertEqual({step["original_step_id"] for step in removed}, {3, 6, 7})
        self.assertEqual((chain["b"]["initial_id"], chain["h"]["initial_id"]), (1, 4))
        self.assertEqual(chain["b"]["write_ids"], [2, 5, 8, 9])
        self.assertEqual(chain["h"]["write_ids"], [5, 8, 9])
        self.assertEqual([step["_id"] for step in steps if step["role"] == "read"], [10, 11])
        later_writer = next(step for step in steps if step["_id"] == 8)
        evidence = next(item["evidence"] for item in later_writer["write_source_requirements"] if item["path"] == h)
        self.assertEqual((evidence["anchor_id"], evidence["mutation_ids"]), (4, [5]))
        for index, step in enumerate(steps, 1):
            step["index"] = index
        assert_forward_sources(self, engine, {"steps": steps})

    def test_parent_anchor_remains_known_after_noninvertible_child_write(self):
        parent, child = "$.state_before.account", "$.state_before.account.balance"
        update = mutation("$.state_after.account.balance", child, "$.args.amount")
        update["state_source_relations"][0].update(recoverability="partial")
        entries = [spec("read_account", [returned("account", parent)]), spec("write_balance", mutations=[update])]
        engine = sampler(entries, {"balance": {"path": child}})
        sequence = [("read_account", "dependency_anchor", parent), ("write_balance", "write", child),
                    ("read_account", "initial_anchor", child), ("write_balance", "write", child),
                    ("read_account", "final_read", child)]
        chain = {"balance": {"initial_id": 3, "final_id": 5, "write_ids": [4]}}
        steps, removed = engine._reuse_anchors(provisional_steps(engine, sequence), chain, {"balance": child})
        self.assertEqual([step["original_step_id"] for step in removed], [3])
        self.assertEqual(chain["balance"]["write_ids"], [2, 4])
        self.assertEqual(len(steps), 4)

    def test_computed_but_never_observed_target_keeps_its_first_anchor(self):
        x = "$.state_before.x"
        entries = [*basic_entries()[:2], spec("set_x", mutations=[mutation("$.state_after.x", "$.args.value")])]
        engine = sampler(entries)
        sequence = [("set_x", "write", x), ("read_x", "initial_anchor", x),
                    ("write_x", "write", x), ("read_x", "final_read", x)]
        chain = {"x": {"initial_id": 2, "final_id": 4, "write_ids": [3]}}
        steps, removed = engine._reuse_anchors(provisional_steps(engine, sequence), chain, {"x": x})
        self.assertFalse(removed)
        self.assertEqual(chain["x"]["initial_id"], 2)
        self.assertEqual(len(steps), 4)

    def test_unfixed_writer_source_cannot_be_hidden_by_anchor_reuse(self):
        x = "$.state_before.x"
        entries = [basic_entries()[0], spec("write_x", mutations=[mutation("$.state_after.x", "$.state_before.z")])]
        engine = sampler(entries)
        sequence = [("read_x", "dependency_anchor", x), ("write_x", "write", x),
                    ("read_x", "initial_anchor", x)]
        with self.assertRaisesRegex(SamplingError, "Unfixed source after anchor reuse"):
            engine._reuse_anchors(provisional_steps(engine, sequence), {}, {})

    def test_max_length_is_checked_after_redundant_anchors_are_removed(self):
        x, y = "$.state_before.x", "$.state_before.y"
        entries = [spec("read_both", [returned("x", x), returned("y", y)]),
                   spec("write_both", mutations=[mutation("$.state_after.x", x), mutation("$.state_after.y", y)])]
        engine = sampler(entries, {"x": {"path": x}, "y": {"path": y}}, seed=2)
        result = engine.build(max_writes=1, target_max_writes={"y": 2}, min_length=1, max_length=5)
        self.assertLessEqual(len(result["steps"]), 5)
        self.assertTrue(result["planning"]["redundant_anchors_removed"])
        self.assertEqual(result["planning"]["target_chains"]["x"]["anchor_step"],
                         result["planning"]["target_chains"]["y"]["anchor_step"])
        assert_forward_sources(self, engine, result)

    def test_parent_coverage_is_directional(self):
        self.assertTrue(covers("$.state_before.account", "$.state_after.account.balance"))
        self.assertFalse(covers("$.state_before.account.balance", "$.state_before.account"))
        self.assertTrue(covers("$.state_after.orders['{args.order_id}']", "$.state_before.orders['{target_order_id}'].status"))

    def test_minimum_write_count_and_minimum_padding(self):
        result = sampler(basic_entries()).build(max_writes=4, min_length=20, max_length=20)
        self.assertEqual(len(result["steps"]), 20)
        self.assertEqual(result["planning"]["necessary_length"], 6)
        self.assertEqual(result["planning"]["distractor_count"], 14)
        chain = result["planning"]["target_chains"]["x"]
        self.assertEqual(chain["write_count"], 4)
        self.assertEqual(chain["min_write_count"], 4)
        self.assertEqual(result["planning"]["write_count_mode"], "at_least_minimum")
        self.assertEqual(result["planning"]["target_min_writes"], {"x": 4})
        self.assertTrue(all(chain["anchor_step"] < index < chain["read_step"] for index in chain["write_steps"]))
        self.assertEqual(result["steps"][0]["role"], "anchor")
        self.assertEqual(result["steps"][-1]["role"], "read")
        self.assertTrue(all(step["tool"] == "unrelated" for step in result["steps"] if step["role"] == "distractor"))

    def test_necessary_steps_are_not_truncated_to_minimum(self):
        result = sampler(basic_entries()).build(max_writes=4, min_length=1, max_length=10)
        self.assertEqual(len(result["steps"]), 6)
        self.assertEqual(result["planning"]["distractor_count"], 0)

    def test_maximum_length_rejects_necessary_steps(self):
        with self.assertRaisesRegex(SamplingError, "exceed max-length"):
            sampler(basic_entries()).build(max_writes=2, min_length=1, max_length=3, attempts=2)

    def test_no_unrelated_padding_is_reported(self):
        with self.assertRaisesRegex(SamplingError, "No unrelated"):
            sampler(basic_entries()[:2]).build(max_writes=1, min_length=10, max_length=10, attempts=2)

    def test_same_call_other_field_fixes_co_source(self):
        entries = [spec("read_combination", [returned("sum", "$.state_before.x", "$.state_before.z"),
                                              returned("z", "$.state_before.z")]), basic_entries()[1]]
        result = sampler(entries).build(max_writes=1, min_length=1, max_length=3)
        self.assertEqual([step["tool"] for step in result["steps"]],
                         ["read_combination", "write_x", "read_combination"])
        self.assertFalse(any(step.get("fix_dependencies") for step in result["steps"]))

    def test_same_call_fixing_runs_to_fixed_point(self):
        node_spec = spec("chain", [returned("x_plus_z", "$.state_before.x", "$.state_before.z"),
                                   returned("z_plus_q", "$.state_before.z", "$.state_before.q"),
                                   returned("q", "$.state_before.q")])
        engine = sampler([node_spec, basic_entries()[1]])
        self.assertTrue(engine.proof(engine.closure(engine.nodes[0]), "$.state_before.x"))
        result = engine.build(max_writes=1, min_length=1, max_length=3)
        self.assertEqual(len(result["steps"]), 3)

    def test_an_assumed_co_source_is_not_reported_as_a_same_call_anchor(self):
        combined = returned("sum", "$.state_before.x", "$.state_before.z")
        combined["state_source_relations"][1].update(transform="aggregation", recoverability="partial")
        entries = [spec("read_sum", [combined]),
                   spec("read_z", [returned("z", "$.state_before.z")]), basic_entries()[1]]
        result = sampler(entries).build(max_writes=1, dependency_max_writes=0, min_length=1, max_length=15)
        for step in result["steps"]:
            self.assertFalse(any(fixed["evidence"]["kind"] == "earlier_state" for fixed in step["fixes"]))
            if step["tool"] == "read_sum":
                self.assertFalse(any(fixed["path"] == "$.state_before.z" for fixed in step["fixes"]))

    def test_writer_self_fixes_write_source(self):
        entries = [basic_entries()[0], spec("write_with_z", [returned("z", "$.state_before.z")],
                    [mutation("$.state_after.x", "$.state_before.x", "$.state_before.z")])]
        result = sampler(entries).build(max_writes=1, min_length=1, max_length=3)
        writer = next(step for step in result["steps"] if step["tool"] == "write_with_z")
        z = next(item for item in writer["write_source_requirements"] if item["path"].endswith(".z"))
        self.assertEqual(z["resolution"], "same_call_return")
        self.assertFalse(any(step.get("field") == "$.state_before.z" for step in result["steps"]))

    def test_conditional_writer_source_uses_same_call_co_source(self):
        entries = [basic_entries()[0], spec("write_with_inverse", [
            returned("sum", "$.state_before.x", "$.state_before.z"), returned("z", "$.state_before.z")
        ], [mutation("$.state_after.x", "$.state_before.x", "$.state_before.z")])]
        result = sampler(entries).build(max_writes=1, min_length=1, max_length=3)
        writer = next(step for step in result["steps"] if step["tool"] == "write_with_inverse")
        self.assertIn("anchor", writer["roles"])
        self.assertTrue(all(item["resolution"] == "same_call_return" for item in writer["write_source_requirements"]))
        self.assertEqual(len(result["steps"]), 2)

    def test_changed_post_state_does_not_fix_pre_state_without_inverse(self):
        entries = [basic_entries()[0], spec("read_z", [returned("z", "$.state_before.z")]),
                   spec("write_and_reset_z", [returned("new_z", "$.state_after.z")], [
                       mutation("$.state_after.x", "$.state_before.x", "$.state_before.z"),
                       mutation("$.state_after.z", "$.args.new_z")])]
        result = sampler(entries).build(max_writes=1, dependency_max_writes=0, min_length=1, max_length=10)
        writer = next(step for step in result["steps"] if step["tool"] == "write_and_reset_z"
                      and step["selected_mutation_index"] is not None)
        old_z = next(item for item in writer["write_source_requirements"] if item["path"].endswith(".z"))
        self.assertEqual(old_z["resolution"], "earlier_chain")
        assert_forward_sources(self, sampler(entries), result)

    def test_parent_return_covers_target_but_parent_set_mutation_does_not(self):
        entries = [spec("read_account", [returned("account", "$.state_before.account")]),
                   spec("write_account", mutations=[mutation("$.state_after.account", "$.state_before.account")])]
        with self.assertRaises(SamplingError):
            sampler(entries, {"balance": {"path": "$.state_before.account.balance"}}).build(
                max_writes=1, dependency_max_writes=0, min_length=1, max_length=8,
                attempts=2)

    def test_child_mutation_is_a_parent_writer(self):
        entries = [spec("read_account", [returned("account", "$.state_before.account")]),
                   spec("write_balance", mutations=[mutation("$.state_after.account.balance", "$.args.amount")])]
        result = sampler(entries, {"account": {"path": "$.state_before.account"}}).build(
            max_writes=1, min_length=1, max_length=10)
        self.assertEqual(result["planning"]["target_chains"]["account"]["write_count"], 1)

    def test_insert_parent_mutation_is_a_descendant_writer(self):
        entries = [spec("read_account", [returned("account", "$.state_before.account")]),
                   spec("insert_account", mutations=[mutation("$.state_after.account", operation="insert")])]
        result = sampler(entries, {"balance": {"path": "$.state_before.account.balance"}}).build(
            max_writes=1, dependency_max_writes=0, min_length=1, max_length=8)
        self.assertEqual(result["planning"]["target_chains"]["balance"]["write_count"], 1)

    def test_mutation_write_relationship_is_directional(self):
        monitored = "$.state_before.account.balance"
        self.assertTrue(mutation_writes_path(mutation("$.state_after.account.balance"), monitored))
        self.assertFalse(mutation_writes_path(mutation("$.state_after.account"), monitored))
        self.assertTrue(mutation_writes_path(mutation("$.state_after.account", operation="insert"), monitored))

    def test_depth_two_uses_single_source_reader_immediately(self):
        entries = [basic_entries()[0], spec("read_z", [returned("z", "$.state_before.z")]),
                   spec("read_q", [returned("q", "$.state_before.q")]),
                   spec("write_x", mutations=[mutation("$.state_after.x", "$.state_before.x", "$.state_before.z")]),
                   spec("write_z", mutations=[mutation("$.state_after.z", "$.state_before.z", "$.state_before.q")]),
                   spec("write_q", mutations=[mutation("$.state_after.q", "$.state_before.q")])]
        found = False
        for seed in range(20):
            result = sampler(entries, seed=seed).build(max_writes=1, dependency_max_writes=1,
                                                       min_length=1, max_length=15)
            self.assertFalse(any(step["tool"] == "write_q" for step in result["steps"]))
            for index, step in enumerate(result["steps"]):
                if step["tool"] == "write_z":
                    self.assertEqual(result["steps"][index - 1]["tool"], "read_q")
                    self.assertEqual(result["steps"][index - 1]["dependency_depth"], 2)
                    found = True
        self.assertTrue(found)

    def test_depth_two_rejects_multi_source_even_if_co_source_is_self_fixed(self):
        entries = [spec("read_x", [returned("sum", "$.state_before.x", "$.state_before.z")]),
                   spec("read_z", [returned("sum", "$.state_before.z", "$.state_before.q")]),
                   spec("read_q", [returned("sum", "$.state_before.q", "$.state_before.r"),
                                    returned("r", "$.state_before.r")]), basic_entries()[1]]
        with self.assertRaises(SamplingError):
            sampler(entries).build(max_writes=1, min_length=1, max_length=20, attempts=2)

    def test_cyclic_inverse_without_seed_is_not_fixed(self):
        entries = [spec("read_sum", [returned("sum", "$.state_before.x", "$.state_before.z")]), basic_entries()[1]]
        with self.assertRaises(SamplingError):
            sampler(entries).build(max_writes=1, min_length=1, max_length=10, attempts=2)

    def test_multiple_targets_can_interleave_initial_anchors_and_writes(self):
        entries = [*basic_entries()[:2], spec("read_y", [returned("y", "$.state_before.y")]),
                   spec("write_y", mutations=[mutation("$.state_after.y", "$.state_before.y")])]
        targets = {name: {"path": "$.state_before." + name} for name in ("x", "y")}
        interleaved = False
        for seed in range(20):
            result = sampler(entries, targets, seed=seed).build(max_writes=2, min_length=1, max_length=10)
            chains = result["planning"]["target_chains"]
            for value in chains.values():
                self.assertEqual(value["write_count"], 2)
            last_anchor = max(value["anchor_step"] for value in chains.values())
            if any(step["role"] == "write" and step["index"] < last_anchor for step in result["steps"]):
                interleaved = True
        self.assertTrue(interleaved)

    def test_shared_writer_counts_for_both_targets(self):
        entries = [spec("read_both", [returned("x", "$.state_before.x"), returned("y", "$.state_before.y")]),
                   spec("write_both", mutations=[mutation("$.state_after.x", "$.state_before.x"),
                                                 mutation("$.state_after.y", "$.state_before.y")])]
        targets = {name: {"path": "$.state_before." + name} for name in ("x", "y")}
        result = sampler(entries, targets).build(max_writes=2, min_length=1, max_length=10)
        chains = result["planning"]["target_chains"]
        for name in targets:
            self.assertGreaterEqual(chains[name]["write_count"], 2)
            for index in chains[name]["write_steps"]:
                self.assertIn(name, result["steps"][index - 1]["write_targets"])
        assert_forward_sources(self, sampler(entries, targets), result)

    def test_shared_writes_may_exceed_another_targets_minimum(self):
        entries = [spec("read_x", [returned("x", "$.state_before.x")]),
                   spec("read_y", [returned("y", "$.state_before.y")]),
                   spec("write_both", mutations=[mutation("$.state_after.x", "$.state_before.x"),
                                                 mutation("$.state_after.y", "$.state_before.y")])]
        targets = {name: {"path": "$.state_before." + name} for name in ("x", "y")}
        exceeded = False
        outside_interval = False
        for seed in range(20):
            engine = sampler(entries, targets, seed=seed)
            result = engine.build(max_writes=1, target_max_writes={"y": 3},
                                  min_length=1, max_length=15)
            assert_forward_sources(self, engine, result)
            chains = result["planning"]["target_chains"]
            self.assertGreaterEqual(chains["x"]["write_count"], 1)
            self.assertGreaterEqual(chains["y"]["write_count"], 3)
            exceeded |= chains["x"]["write_count"] > 1
            for name, chain in chains.items():
                expected = [step["index"] for step in result["steps"]
                            if step["tool"] == "write_both"
                            and chain["anchor_step"] <= step["index"] < chain["read_step"]]
                self.assertEqual(chain["write_steps"], expected)
                outside_interval |= any(step["tool"] == "write_both" and step["index"] not in expected
                                        for step in result["steps"])
        self.assertTrue(exceeded)
        self.assertTrue(outside_interval)

    def test_invalid_controls_are_rejected(self):
        engine = sampler(basic_entries())
        for kwargs in ({"max_writes": 0}, {"min_length": 10, "max_length": 5},
                       {"dependency_max_writes": -1}, {"target_max_writes": {"missing": 2}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                engine.build(**kwargs)

    def test_seeded_output_is_reproducible(self):
        one = sampler(basic_entries(), seed=7).build(min_length=10, max_length=20)
        two = sampler(basic_entries(), seed=7).build(min_length=10, max_length=20)
        self.assertEqual(one, two)

    def test_partial_reader_requires_explicit_refinement(self):
        entries = [spec("read_x", [returned("x", "$.state_before.x", reversible=False)]), basic_entries()[1]]
        with self.assertRaises(SamplingError):
            sampler(entries).build(max_writes=1, min_length=1, max_length=10, attempts=2)
        engine = BackwardSampler({entry["tool"]: entry for entry in entries},
                                 {"x": {"path": "$.state_before.x"}}, reader_refinements=[{
                                     "tool": "read_x", "branch": "success_main", "field": "$.result.x",
                                     "source": "$.state_before.x", "recoverability": "conditional",
                                     "condition": "Omit the filter arguments."}])
        result = engine.build(max_writes=1, min_length=1, max_length=10)
        self.assertTrue(all("Omit the filter arguments." in step["additional_conditions"]
                            for step in result["steps"] if step["tool"] == "read_x"))

    def test_enabled_long_context_readers_writers_and_distractors_are_excluded(self):
        enabled_reader = spec("enabled_reader", [returned("x", "$.state_before.x")])
        enabled_writer = spec("enabled_writer", mutations=[mutation("$.state_after.x", "$.args.value")])
        enabled_distractor = spec("enabled_distractor")
        for entry in (enabled_reader, enabled_writer, enabled_distractor):
            entry["branches"][0]["if"] = "$.state_before.long_context == true"
        mode_writer = spec("mode_writer", mutations=[mutation("$.state_after.long_context", "$.args.value")])
        engine = sampler([*basic_entries(), enabled_reader, enabled_writer, enabled_distractor, mode_writer])
        self.assertEqual({node.tool for node in engine.nodes}, {"read_x", "write_x", "unrelated"})
        result = engine.build(min_length=20)
        self.assertEqual(result["state_conditions"], {"$.state_before.long_context": False})
        for step in result["steps"]:
            self.assertIn(LONG_CONTEXT_DISABLED, step["additional_conditions"])

    def test_real_specs_only_disabled_long_context_branches_are_sampled(self):
        specs, targets, refinements, _ = load_inputs(DEFAULT_CATALOG)
        engine = BackwardSampler(specs, targets, reader_refinements=refinements)
        available = {(node.tool, node.branch["id"]) for node in engine.nodes}
        enabled = {(entry["tool"], branch["id"]) for entry in specs.values()
                   for branch in entry["branches"]
                   if branch["id"].startswith("success_") and "long_context == true" in branch["if"]}
        self.assertEqual(len(enabled), 5)
        self.assertFalse(available & enabled)
        self.assertIn(("get_available_stocks", "success_standard_context"), available)
        self.assertIn(("get_order_details", "success_standard"), available)
        self.assertIn(("get_transaction_history", "success_filtered_history"), available)
        for seed in range(20):
            engine = BackwardSampler(specs, targets, reader_refinements=refinements, seed=seed)
            result = engine.build(min_length=20)
            assert_forward_sources(self, engine, result)
            for step in result["steps"]:
                self.assertNotIn((step["tool"], step["branch"]), enabled)
                self.assertIn(LONG_CONTEXT_DISABLED, step["additional_conditions"])
        # Enabled guards remain as conditions to avoid, not selected branch requirements.
        node = next(node for node in engine.nodes if node.tool == "get_order_details")
        self.assertTrue(any("long_context == true" in branch["if"] for branch in node.earlier))

    def test_real_specs_eligible_success_branches_and_execute_order_are_available(self):
        specs, targets, refinements, _ = load_inputs(DEFAULT_CATALOG)
        engine = BackwardSampler(specs, targets, reader_refinements=refinements, seed=42)
        branches = {node.branch["id"] for node in engine.nodes if node.tool == "execute_order"}
        self.assertEqual(branches, {"success_buy", "success_sell_with_remaining_shares", "success_sell_position_closed"})
        self.assertTrue(any(node.tool == "place_order" for node in engine.nodes))
        seen = set()
        for _ in range(15):
            result = engine.build(target_max_writes={"balance": 5}, min_length=30, max_length=60)
            assert_forward_sources(self, engine, result)
            place = next(node for node in result["planning"]["writer_candidates_by_target"]["orders"]
                         if node["tool"] == "place_order")
            self.assertIn("$.state_before.orders", place["unfixable_write_sources"])
            self.assertGreaterEqual(len(result["steps"]), 30)
            for name, count in (("balance", 5), ("transaction_history", 3), ("orders", 3)):
                self.assertGreaterEqual(result["planning"]["target_chains"][name]["write_count"], count)
            for step in result["steps"]:
                self.assertIn("branch_condition", step)
                self.assertNotIn("args", step)
                if step["tool"] == "execute_order":
                    seen.add(step["branch"])
                    selected = step["selected_mutation_index"]
                    if selected is not None and state_sources(step["mutations"][selected]):
                        self.assertTrue(any(source["resolution"] == "same_call_return"
                                            for source in step["write_source_requirements"]))
        self.assertEqual(seen, branches)

    def test_forward_source_audit_across_multiple_controls(self):
        specs, targets, refinements, _ = load_inputs(DEFAULT_CATALOG)
        for seed in range(30):
            engine = BackwardSampler(specs, targets, reader_refinements=refinements, seed=seed)
            for names, overrides in ((["balance", "orders"], {}),
                                     (["transaction_history", "orders"], {}),
                                     (list(targets), {"balance": 4})):
                result = engine.build(names, max_writes=2, target_max_writes=overrides,
                                      min_length=1, max_length=50)
                assert_forward_sources(self, engine, result)

    def test_catalog_writer_templates_and_state_assumptions_are_ignored(self):
        with TemporaryDirectory() as directory:
            config = json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8"))
            config["state_specs"] = str((DEFAULT_CATALOG.parent / config["state_specs"]).resolve())
            config["writer_sequences"] = {"orders": [["nonexistent:success_invalid"]]}
            config["state_conditions"] = {"$.state_before.long_context": "not a usable assumption"}
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            specs, targets, refinements, _ = load_inputs(path)
            result = BackwardSampler(specs, targets, reader_refinements=refinements).build(
                ["balance", "transaction_history", "orders"], min_length=20,
            )
            self.assertEqual(len(result["steps"]), 20)
            self.assertEqual(result["state_conditions"], {"$.state_before.long_context": False})

    def test_cli_writes_complete_manifest_and_symbolic_artifacts(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            main(["--count", "3", "--min-length", "20", "--max-length", "50",
                  "--min-writes", "2", "--target-min-writes", "balance=5", "--output-dir", str(output)])
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((manifest["status"], manifest["accepted"]), ("complete", 3))
            self.assertFalse(manifest["writer_sequences_used"])
            self.assertEqual(manifest["filler_strategy"], ["support_anchor", "distractor"])
            self.assertFalse(manifest["linked_distractors"])
            self.assertEqual(manifest["support_anchor_policy"],
                             "sample_then_forward_anchor_reuse")
            self.assertFalse(manifest["backend_execution"])
            self.assertEqual(manifest["dependency_scope"], "selected_mutation_and_return_field")
            self.assertEqual(manifest["state_conditions"], {"$.state_before.long_context": False})
            self.assertEqual(manifest["target_min_writes"], {"balance": 5, "transaction_history": 2, "orders": 2})
            self.assertEqual(manifest["write_count_mode"], "best_effort_lifecycle")
            for path in output.glob("backward_*.json"):
                result = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(len(result["steps"]), 20)
                self.assertEqual(result["state_conditions"], manifest["state_conditions"])
                self.assertTrue(all(LONG_CONTEXT_DISABLED in step["additional_conditions"] for step in result["steps"]))
                self.assertNotIn("initial_state", result)
                self.assertEqual(result["planning"]["dependency_scope"], "selected_mutation_and_return_field")
                self.assertFalse(any("args" in step or "observation" in step for step in result["steps"]))
            with patch("sys.stderr", new=StringIO()), self.assertRaises(SystemExit):
                main(["--output-dir", str(output)])

    def test_cli_can_enable_linked_distractors(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "linked"
            main(["--targets", "orders", "--min-writes", "1", "--min-length", "6",
                  "--max-length", "12", "--count", "1", "--linked-distractors",
                  "--output-dir", str(output)])
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            result = json.loads(next(output.glob("backward_*.json")).read_text(encoding="utf-8"))
            self.assertTrue(manifest["linked_distractors"])
            self.assertIn("linked_distractor", manifest["filler_strategy"])
            self.assertTrue(result["planning"]["linked_distractors"])


if __name__ == "__main__":
    unittest.main()
