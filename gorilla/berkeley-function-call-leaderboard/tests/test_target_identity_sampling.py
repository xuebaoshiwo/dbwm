"""Target addresses must be observed independently of the values being written."""

import copy
import unittest

from bfcl_eval.consistency.data_generator_v2.backward_state_knowledge import (
    StateKnowledge, mutation_sources, state_sources,
)
from bfcl_eval.consistency.data_generator_v2.generate_backward_trajectories import (
    BackwardSampler, SamplingError, DEFAULT_CATALOG, load_inputs, load_lifecycle_rules,
)
from bfcl_eval.consistency.data_generator_v2.generate_tool_state_specs import validate_spec
from tests.test_backward_trajectory_sampling import mutation, returned, spec


TARGET = "$.state_before.bins['{key}'].quantity"
ADDRESS = "$.state_before.jobs['{args.job_id}'].destination"
WRITE = "$.state_after.bins['{job.destination}'].quantity"


def addressed_mutation(target=WRITE, source=ADDRESS):
    item = mutation(target, "$.args.quantity")
    item["state_source_relations"] = []
    item["target_identity_sources"] = [{
        "placeholder": "job.destination", "path": source,
        "logic": "The target key equals the job destination before the call.",
    }]
    return item


def entries(reader=True):
    result = [spec("read_bins", [returned("bins", "$.state_before.bins")]),
              spec("write", mutations=[addressed_mutation()])]
    if reader:
        result.append(spec("read_job", [returned("destination", ADDRESS)]))
    return result


def engine(items):
    return BackwardSampler({item["tool"]: item for item in items}, {"bins": {"path": TARGET}})


class TargetIdentitySamplingTests(unittest.TestCase):
    def test_target_key_adds_observation_and_bound_equality(self):
        result = engine(entries()).build(max_writes=1, dependency_max_writes=0,
                                         min_length=1, max_length=10)
        writer = next(step for step in result["steps"] if step["tool"] == "write")
        reader = next(step for step in result["steps"] if step["tool"] == "read_job")
        self.assertLess(reader["index"], writer["index"])
        requirement, = writer["write_source_requirements"]
        identity, = requirement["target_identity_relations"]
        self.assertEqual(identity["placeholder"], "key_1")
        self.assertEqual(identity["path"], ADDRESS.replace("args.job_id", "job_id_1"))
        self.assertEqual(reader["placeholder_bindings"]["args.job_id"], "job_id_1")
        self.assertEqual(state_sources(writer["mutations"][0]), ())
        self.assertEqual(mutation_sources(writer["mutations"][0]), (identity["path"],))

    def test_target_without_identity_reader_is_rejected(self):
        with self.assertRaises(SamplingError):
            engine(entries(False)).build(max_writes=1, min_length=1, max_length=10, attempts=2)

    def test_same_call_observation_reuses_identity_without_extra_reader(self):
        items = entries(False)
        items[1] = spec("write", [returned("destination", ADDRESS)], [addressed_mutation()])
        result = engine(items).build(max_writes=1, dependency_max_writes=0, min_length=1, max_length=10)
        writer = next(step for step in result["steps"] if step["tool"] == "write")
        self.assertEqual(writer["write_source_requirements"][0]["resolution"], "same_call_return")
        self.assertNotIn("read_job", {step["tool"] for step in result["steps"]})

    def test_post_state_cannot_replace_changed_pre_state_identity(self):
        items = entries(False)
        items[1] = spec("write", [returned("destination", ADDRESS.replace("before", "after"))],
                        [addressed_mutation(), mutation(ADDRESS.replace("before", "after"), "$.args.new_key")])
        items[1]["mutations"][1]["state_source_relations"] = []
        result = engine(items).build(max_writes=1, min_length=1, max_length=10, attempts=2)
        writer = next(step for step in result["steps"] if step["selected_mutation_index"] == 0)
        requirement, = writer["write_source_requirements"]
        self.assertEqual(requirement["resolution"], "earlier_chain")
        self.assertLess(requirement["evidence"]["anchor_step"], writer["index"])

    def test_unselected_side_effect_does_not_schedule_identity_reader(self):
        items = [spec("read_x", [returned("x", "$.state_before.x")]),
                 spec("write_x", mutations=[mutation("$.state_after.x", "$.state_before.x"),
                                             addressed_mutation()])]
        sampler = BackwardSampler({item["tool"]: item for item in items},
                                  {"x": {"path": "$.state_before.x"}})
        result = sampler.build(max_writes=1, min_length=1, max_length=5)
        writer = next(step for step in result["steps"] if step["tool"] == "write_x")
        self.assertEqual([r["path"] for r in writer["write_source_requirements"]], ["$.state_before.x"])
        self.assertFalse(writer["mutation_knowledge"][1]["sources_fixed"])
        self.assertTrue(writer["mutation_knowledge"][1]["missing_target_identities"])

    def test_unknown_address_invalidates_containing_map(self):
        knowledge = StateKnowledge()
        knowledge.observe({"$.state_before.bins": {"kind": "return"}}, 1)
        knowledge.apply_mutations([addressed_mutation()], {}, 2)
        self.assertIsNone(knowledge.get("$.state_before.bins"))
        self.assertIsNone(knowledge.get("$.state_before.bins['{key_2}'].quantity"))

    def test_source_keys_do_not_add_identity_dependencies(self):
        source = "$.state_before.prices['{job.destination}']"
        item = mutation("$.state_after.x", source)
        item["target_identity_sources"] = []
        items = [spec("read_x", [returned("x", "$.state_before.x")]),
                 spec("write_x", [returned("price", source)], [item])]
        sampler = BackwardSampler({s["tool"]: s for s in items}, {"x": {"path": "$.state_before.x"}})
        result = sampler.build(max_writes=1, min_length=1, max_length=5)
        writer = next(s for s in result["steps"] if s["tool"] == "write_x")
        self.assertEqual(len(writer["write_source_requirements"]), 1)
        self.assertEqual(writer["write_source_requirements"][0]["target_identity_relations"], [])

    def test_real_holdings_writers_bind_distinct_orders_to_monitored_symbol(self):
        specs, targets, refinements, _ = load_inputs(DEFAULT_CATALOG)
        rules, _ = load_lifecycle_rules(DEFAULT_CATALOG, specs, targets)
        sampler = BackwardSampler(specs, targets, reader_refinements=refinements, lifecycle_rules=rules)
        result = sampler.build(["holdings"], max_writes=3, dependency_max_writes=0, min_length=1, max_length=30)
        chain = result["planning"]["target_chains"]["holdings"]
        order_ids = set()
        for index in chain["write_steps"]:
            step = result["steps"][index - 1]
            identity = next(r for r in step["write_source_requirements"] if r["target_identity_relations"])
            self.assertEqual(identity["target_identity_relations"][0]["placeholder"], "symbol_1")
            order_ids.add(step["placeholder_bindings"]["args.order_id"])
            self.assertEqual(identity["path"], ADDRESS.replace("jobs", "orders").replace("destination", "symbol")
                             .replace("args.job_id", step["placeholder_bindings"]["args.order_id"]))
        self.assertEqual(len(order_ids), 3)


class TargetIdentityValidationTests(unittest.TestCase):
    def valid(self):
        item = spec("write", mutations=[addressed_mutation()])
        item["schema_version"] = "1.3"
        item["returns"][0]["fields"][0]["state_observation_relations"] = []
        return item

    def test_valid_target_identity_does_not_require_source_relation(self):
        validate_spec(self.valid(), "write")

    def test_invalid_target_identity_entries(self):
        for change in ("missing", "empty", "unrelated", "duplicate", "post_state", "argument", "self_reference"):
            with self.subTest(change=change):
                item = self.valid()
                mutation_item = item["mutations"][0]
                identities = mutation_item["target_identity_sources"]
                if change == "missing":
                    del mutation_item["target_identity_sources"]
                elif change == "empty":
                    identities.clear()
                elif change == "unrelated":
                    identities[0]["placeholder"] = "unused"
                elif change == "duplicate":
                    identities.append(copy.deepcopy(identities[0]))
                elif change == "post_state":
                    identities[0]["path"] = ADDRESS.replace("before", "after")
                elif change == "argument":
                    identities[0]["path"] = "$.args.key"
                else:
                    identities[0]["path"] = "$.state_before.bins['{job.destination}'].key"
                with self.assertRaises(ValueError):
                    validate_spec(item, "write")

    def test_direct_argument_target_needs_no_identity_anchor(self):
        item = self.valid()
        item["mutations"][0].update(target="$.state_after.bins['{args.key}'].quantity", target_identity_sources=[])
        validate_spec(item, "write")
