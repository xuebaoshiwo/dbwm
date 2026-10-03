"""Offline checks for domain-neutral generation and safe in-place updates."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from bfcl_eval.consistency.data_generator_v2.grounding.generate_queries import (
    audit_problems, blind_problems, build_context, find_cases, fingerprint, generate_query, process_file, validate_query, writer_context,
)


def fixture():
    chain = [
        {"step_id": 2, "tool": "reserve_book", "args": {"title": "Dune"},
         "requirements": [{"kind": "result_binding", "result": "$.result.id", "placeholder": "reservation_id_1"}]},
        {"step_id": 5, "tool": "get_reservation", "args": {"reservation_id": 781},
         "placeholder_bindings": {"args.reservation_id": "reservation_id_1"}},
    ]
    case = {"status": "accepted", "source": "example.json", "initial_state": {"private": 99},
            "placeholder_values": {"reservation_id_1": 781},
            "tool_chain": chain,
            "trace": [{**step, "result": {"id": 781, "status": "reserved"},
                       "state_before": {"private": 99}} for step in chain]}
    schemas = {step["tool"]: {"name": step["tool"], "description": "Library tool",
                              "parameters": {}} for step in chain}
    return case, schemas


QUERY = "Could you reserve Dune for me and check the details of that reservation?"


def audit():
    return {"valid": True, "natural": True, "faithful": True, "no_answer_leakage": True,
            "issues": [], "coverage": [
                {"step_id": 2, "supported": True, "query_quote": "reserve Dune for me"},
                {"step_id": 5, "supported": True, "query_quote": "check the details of that reservation"},
            ]}


def successful_model(messages, **kwargs):
    if "without seeing any reference" in messages[0]["content"]:
        return {"natural": True, "naturalness_reason": "A cohesive library request",
                "tool_call_counts": {"reserve_book": 1, "get_reservation": 1}, "extra_tasks": []}
    if "Independently review" in messages[1]["content"]:
        return audit()
    return {"query": QUERY}


class QueryTests(unittest.TestCase):
    def test_blind_reconstruction_counts_each_call_and_requires_quoted_support(self):
        case, schemas = fixture()
        context = build_context(case, schemas)
        review = {'requested_calls': [
            {'tool': 'reserve_book', 'query_quote': 'reserve Dune for me'},
            {'tool': 'get_reservation', 'query_quote': 'check the details of that reservation'}],
            'extra_tasks': []}
        self.assertEqual(blind_problems(review, context, QUERY), [])
        review['requested_calls'].append(review['requested_calls'][0])
        self.assertTrue(blind_problems(review, context, QUERY))
        review['requested_calls'].pop()
        review['requested_calls'][0]['query_quote'] = 'imaginary instruction'
        self.assertTrue(blind_problems(review, context, QUERY))

    def test_revising_existing_draft_redacts_its_future_id(self):
        case, schemas = fixture()
        case['query'] = 'Reserve Dune and check reservation 781.'
        requests = []
        def model(messages, **kwargs):
            requests.append(messages)
            return successful_model(messages, **kwargs)
        query, metadata = generate_query(case, schemas, ask=model, revise_existing=True)
        self.assertEqual(query, QUERY)
        self.assertTrue(metadata['revised_existing_query'])
        self.assertNotIn('781', requests[0][1]['content'])
        self.assertIn('Make MINIMAL corrections', requests[0][1]['content'])

    def test_writer_never_sees_replay_answers_or_future_id_literals(self):
        case, schemas = fixture()
        context = build_context(case, schemas)
        safe = writer_context(context)
        self.assertNotIn('781', json.dumps(safe))
        self.assertNotIn('retrospective_results_NOT_user_knowledge', safe)
        self.assertEqual(safe['reference_calls'][0]['args'], {'title': 'Dune'})
        self.assertIn('created by reference call 2', safe['reference_calls'][1]['args']['reservation_id'])
        self.assertEqual(case['tool_chain'][1]['args']['reservation_id'], 781)

    def test_intent_first_changes_writer_context_but_not_review_reference(self):
        case, schemas = fixture()
        requests = []
        def model(messages, **kwargs):
            requests.append(messages)
            if len(requests) == 1:
                return {"overall_intent": "Reserve Dune and inspect the new reservation"}
            return successful_model(messages, **kwargs)
        query, metadata = generate_query(case, schemas, ask=model, intent_first=True)
        self.assertEqual(query, QUERY)
        self.assertTrue(metadata["intent_first"])
        self.assertIn('"intent_brief"', requests[1][1]["content"])
        self.assertNotIn('"reference_calls"', requests[1][1]["content"])
        self.assertIn('"reference_calls"', requests[-1][1]["content"])

    def test_future_id_cannot_leak_but_independent_input_can_match_it(self):
        case, schemas = fixture()
        context = build_context(case, schemas)
        with self.assertRaisesRegex(ValueError, "prematurely names"):
            validate_query("Reserve Dune and check reservation 781.", context)
        self.assertEqual(validate_query(QUERY, context), QUERY)
        case["tool_chain"][0]["args"]["edition"] = 781
        case["trace"][0]["args"]["edition"] = 781
        self.assertEqual(validate_query("Reserve edition 781 of Dune for me.", build_context(case, schemas)),
                         "Reserve edition 781 of Dune for me.")

    def test_context_keeps_identity_and_excludes_internal_state(self):
        case, schemas = fixture()
        context = build_context(case, schemas)
        self.assertNotIn("private", json.dumps(context))
        self.assertIn("result_binding", json.dumps(context))
        self.assertEqual(context["reference_calls"][1]["args"], {"reservation_id": 781})
        self.assertEqual(len(context["tool_schemas"]), 2)

    def test_mismatched_replay_rejected(self):
        case, schemas = fixture()
        case["trace"] = copy.deepcopy(case["trace"])
        case["trace"][1]["args"] = {"reservation_id": 999}
        with self.assertRaisesRegex(ValueError, "Replay does not match"):
            build_context(case, schemas)

    def test_all_calls_need_real_quotes(self):
        case, schemas = fixture()
        context = build_context(case, schemas)
        review = audit()
        review["coverage"].pop()
        self.assertTrue(audit_problems(review, QUERY, context))
        review = audit()
        review["coverage"][0]["query_quote"] = "made-up quote"
        self.assertTrue(audit_problems(review, QUERY, context))
        review = audit()
        review["no_answer_leakage"] = False
        self.assertTrue(audit_problems(review, QUERY, context))

    def test_revision_uses_review_feedback(self):
        case, schemas = fixture()
        calls = []
        def model(messages, **kwargs):
            calls.append(messages)
            if len(calls) == 3:
                bad = audit()
                bad["issues"] = ["Please preserve the reservation reference"]
                return bad
            return successful_model(messages, **kwargs)
        query, metadata = generate_query(case, schemas, ask=model)
        self.assertEqual(query, QUERY)
        self.assertEqual(metadata["attempts"], 2)
        self.assertIn("Please preserve", calls[3][1]["content"])

    def test_blind_review_catches_unrequested_repeats(self):
        case, schemas = fixture()
        context = build_context(case, schemas)
        review = {"natural": True, "tool_call_counts": {"reserve_book": 1, "get_reservation": 0}, "extra_tasks": []}
        self.assertTrue(blind_problems(review, context))

    def test_in_place_preserves_all_original_fields_and_resumes(self):
        case, schemas = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            path.write_text(json.dumps(case), encoding="utf-8")
            self.assertEqual(process_file(path, schemas, ask=successful_model), "written")
            updated = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual({k: v for k, v in updated.items() if k in case}, case)
            self.assertEqual(updated["query"], QUERY)
            self.assertEqual(fingerprint(updated), fingerprint(case))
            self.assertEqual(process_file(path, schemas, ask=lambda *a, **k: self.fail("must skip")), "skipped")
            self.assertEqual(list(Path(directory).iterdir()), [path])
            updated["initial_state"]["private"] = 100
            path.write_text(json.dumps(updated), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale"):
                process_file(path, schemas)

    def test_failed_review_does_not_modify_file(self):
        case, schemas = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            original = json.dumps(case).encode()
            path.write_bytes(original)
            with self.assertRaises(ValueError):
                process_file(path, schemas, ask=lambda *a, **k: {"query": "1. reserve_book"}, max_revisions=0)
            self.assertEqual(path.read_bytes(), original)

    def test_concurrent_edit_is_preserved(self):
        case, schemas = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            path.write_text(json.dumps(case), encoding="utf-8")
            def model(messages, **kwargs):
                path.write_text('{"edited": true}', encoding="utf-8")
                return successful_model(messages, **kwargs)
            with self.assertRaisesRegex(ValueError, "changed during generation"):
                process_file(path, schemas, ask=model)
            self.assertEqual(json.loads(path.read_text()), {"edited": True})

    def test_discovery_excludes_history_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("manifest.json", "_run_history/old.json", "group/case.json"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            self.assertEqual(find_cases(root), [root / "group/case.json"])


if __name__ == "__main__":
    unittest.main()
