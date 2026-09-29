"""Reproducibly annotate the existing TradingBotHard state specifications."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from bfcl_eval.consistency.data_generator_v2.generate_tool_state_specs import (
    BFCL_ROOT, CURRENT_SCHEMA_VERSION, _write_json_atomic, validate_spec,
)
from bfcl_eval.consistency.data_generator_v2.backward_state_knowledge import overlaps


DEFAULT_DIR = BFCL_ROOT / "bfcl_eval/consistency/data_v2/trading_bot_hard"


def relation(path, transform, recoverability, reason, given=()):
    return {"path": path, "transform": transform, "recoverability": recoverability,
            "given": list(given), "reason": reason}


def return_rule(tool, branch, field, path):
    result_path = field["path"]
    if field["source_kind"] == "state_direct":
        return "copy", "exact", "The complete source value is returned unchanged when this field is present."
    if tool == "execute_order":
        if branch.startswith("success_"):
            return "arithmetic", "conditional", (
                "The returned float/int is the source value on its exactly representable numeric domain; "
                "conversion can otherwise lose information."
            )
        if branch in {"error_order_not_open", "error_buy_limit_not_met", "error_sell_limit_not_met"}:
            return "format", "partial", "Case normalization or two-decimal formatting does not preserve every possible source value."
        if branch in {"error_stock_not_found", "error_invalid_order_type"}:
            return "format", "conditional", "The source string is embedded verbatim in a fixed message and can be extracted."
    if tool == "activate_order" and branch == "error_order_not_pending":
        return "format", "partial", "The status is lowercased in the message, so its original spelling is lost."
    if tool == "filter_stocks_by_price":
        return "comparison", "partial", "Membership reveals only whether each price falls inside the requested interval."
    if tool == "get_order_details":
        if branch == "error_order_not_found":
            return "aggregation", "partial", "Only order IDs are formatted into the message; order contents are omitted."
        if branch == "success_long_context" and result_path == "$.result.metadata":
            return "format", "conditional", "The known metadata template embeds the symbol verbatim; extract it from a formatted entry."
    if tool == "get_order_history":
        return "aggregation", "partial", "The list exposes order keys but not the order objects."
    if tool == "get_transaction_history":
        return "filter", "partial", "A date-window projection can omit records; an unbounded call is a special case, not a general inverse."
    if tool == "get_watchlist" and branch == "success_watchlist_long_context":
        return "collection", "conditional", "Given the known appended extension and its length, remove that suffix to recover the source list."
    if tool == "notify_price_change":
        return "comparison", "partial", "Selected symbols constrain percentage changes but omit exact values and other stock fields."
    if tool == "place_order":
        if branch == "error_insufficient_funds_buy":
            return "format", "partial", "The available balance is formatted to two decimal places in the error message."
        if branch == "success_placed_pending_order" and result_path == "$.result.order_id":
            return "selection", "partial", "The first free ID constrains the counter and occupied IDs but cannot reconstruct either source."
    raise ValueError(f"Unclassified return source: {tool} {branch} {result_path} {path}")


def mutation_rule(tool, branch, mutation, path):
    target = mutation["target"]
    if tool == "execute_order":
        if target == "$.state_after.account_info.balance":
            return "arithmetic", "partial", "The trade value and resulting balance are rounded to cents, losing unrestricted source precision."
        if target.startswith("$.state_after.holdings"):
            if mutation["operation"] == "delete":
                return "collection", "none", "The deleted target is absent after the write and does not expose the prior quantity."
            if path == "$.state_before.holdings":
                return "collection", "partial", "One updated holding cannot reconstruct the entire prior holdings mapping."
            return "arithmetic", "conditional", "Given the other source and the filled amount, invert the integer share addition/subtraction."
        if target.endswith(".filled_price"):
            return "arithmetic", "conditional", "Float conversion preserves the price only on the exactly representable numeric domain."
    if tool in {"fund_account", "withdraw_funds"}:
        if target.endswith(".balance"):
            return "arithmetic", "conditional", "Given the argument amount, reverse the addition/subtraction on the bounded currency domain."
        if target == "$.state_after.transaction_history":
            return "collection", "conditional", "The operation appends exactly one record; remove the last element of the observed list."
    if tool == "add_to_watchlist" and path == "$.state_before.watch_list":
        return "collection", "conditional", "The operation appends one item; drop the last item from the observed list."
    if tool == "remove_stock_from_watchlist" and path == "$.state_before.watch_list":
        return "collection", "partial", "Removing an item loses its original position in the list."
    if tool == "place_order":
        return "selection", "partial", "The generated ID/new order constrains but does not uniquely reveal prior counter and order contents."
    raise ValueError(f"Unclassified mutation source: {tool} {branch} {target} {path}")


def observation_relations(tool, branch, field, mutations):
    """Backend-reviewed post-state correspondences, separate from provenance."""
    result_path = field["path"]
    relations = []
    for source in field["state_source_relations"]:
        if source["path"].startswith("$.state_before") and any(
            overlaps(source["path"], mutation["target"]) for mutation in mutations
        ):
            continue
        paths = [source["path"], *source["given"]]
        if any(path.startswith("$.state_before") and any(
            overlaps(path, mutation["target"]) for mutation in mutations
        ) for path in paths):
            continue
        relations.append(relation(source["path"].replace("$.state_before", "$.state_after", 1),
            source["transform"], source["recoverability"],
            "This state region is unchanged on this branch. " + source["reason"],
            [path.replace("$.state_before", "$.state_after", 1) for path in source["given"]]))

    def observed(path, transform="copy", recovery="exact", reason=None):
        if not any(item["path"] == path for item in relations):
            relations.append(relation(path, transform, recovery,
                reason or "The returned value equals this stored post-call field when present."))

    if branch.startswith("success_") and tool == "place_order":
        name = result_path.rsplit(".", 1)[-1]
        stored = "id" if name == "order_id" else name
        observed(f"$.state_after.orders['{{result.order_id}}'].{stored}")
        if name == "order_id":
            observed("$.state_after.order_counter", "arithmetic", "conditional",
                     "The backend sets the post-call counter to returned order_id + 1.")
    elif branch.startswith("success_") and tool in {"activate_order", "cancel_order", "execute_order"}:
        order = "$.state_after.orders['{args.order_id}']"
        if result_path == "$.result.status":
            observed(order + ".status")
        elif tool == "execute_order" and result_path == "$.result.filled_price":
            observed(order + ".filled_price")
    elif tool in {"trading_login", "trading_logout"} and result_path == "$.result.status":
        authenticated = tool == "trading_login"
        observed("$.state_after.authenticated", "selection", "conditional",
                 f"On this branch the returned message identifies the final boolean authentication value as {str(authenticated).lower()}.")
    return relations


def annotate(spec):
    tool = spec["tool"]
    if spec["schema_version"] not in {"1.0", "1.1", CURRENT_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported input version: {spec['schema_version']}")
    for mutation in spec["mutations"]:
        target = mutation["target"]
        if tool in {"fund_account", "withdraw_funds"} and target == "$.state_after.transaction_history":
            if "$.state_before.transaction_history" not in mutation["value_from"]:
                mutation["value_from"].insert(0, "$.state_before.transaction_history")
        if tool in {"add_to_watchlist", "remove_stock_from_watchlist"} and target == "$.state_after.watch_list":
            if "$.state_before.watch_list" not in mutation["value_from"]:
                mutation["value_from"].insert(0, "$.state_before.watch_list")
        state_paths = [path for path in mutation["value_from"] if path.startswith("$.state_before.")]
        mutation["state_source_relations"] = []
        for path in state_paths:
            transform, recovery, reason = mutation_rule(tool, mutation["branch"], mutation, path)
            mutation["state_source_relations"].append(relation(path, transform, recovery, reason,
                (other for other in state_paths if other != path) if recovery == "conditional" else ()))
    for result in spec["returns"]:
        for field in result["fields"]:
            if tool == "get_order_details" and result["branch"] == "error_keyerror_formatting":
                field["value_from"] = [path for path in field["value_from"] if not path.startswith("$.state_")]
                field["logic"] = "The formatting KeyError reports a missing template key, not the order symbol's value."
            if tool == "get_available_stocks" and result["branch"] == "success_extended_context":
                field["value_from"] = [path for path in field["value_from"] if path != "$.state_before.long_context"]
                field["external_from"] = ["sector-specific imported stock-list extension constant"]
                field["logic"] = "The sector argument selects a hardcoded list, then the imported extension is appended. The long_context flag selects this branch only."
            if tool == "get_watchlist" and result["branch"] == "success_watchlist_long_context":
                field["external_from"] = ["WATCH_LIST_EXTENSION imported constant"]
            paths = ([field["source"]] if field["source_kind"] in {"state_direct", "arg_direct"}
                     else field.get("value_from", []))
            state_paths = [path for path in paths if path.startswith(("$.state_before.", "$.state_after."))]
            field["state_source_relations"] = []
            for path in state_paths:
                transform, recovery, reason = return_rule(tool, result["branch"], field, path)
                field["state_source_relations"].append(relation(path, transform, recovery, reason,
                    (other for other in state_paths if other != path) if recovery == "conditional" else ()))
            mutations = [item for item in spec["mutations"] if item["branch"] == result["branch"]]
            field["state_observation_relations"] = observation_relations(tool, result["branch"], field, mutations)
    spec["schema_version"] = CURRENT_SCHEMA_VERSION
    validate_spec(spec, tool)
    return spec


def migrate(directory):
    aggregate = directory / "all_tools.json"
    specs = json.loads(aggregate.read_text(encoding="utf-8"))
    names = [spec["tool"] for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate tool specifications")
    annotated = [annotate(spec) for spec in specs]
    # Verify every individual file agrees with the aggregate before overwriting either.
    for spec in annotated:
        original = json.loads((directory / f"{spec['tool']}.json").read_text(encoding="utf-8"))
        if annotate(original) != spec:
            raise ValueError(f"Individual specification differs from aggregate: {spec['tool']}")
    for spec in annotated:
        _write_json_atomic(directory / f"{spec['tool']}.json", spec)
    _write_json_atomic(aggregate, annotated)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = CURRENT_SCHEMA_VERSION
    manifest["source_relation_annotation"] = "deterministic_backend_review"
    manifest["source_relation_annotated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["post_state_observation_annotation"] = "deterministic_backend_review"
    _write_json_atomic(manifest_path, manifest)
    return {"tools": len(annotated), "mutations": sum(len(s["mutations"]) for s in annotated),
            "return_fields": sum(len(r["fields"]) for s in annotated for r in s["returns"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    print(json.dumps(migrate(args.spec_dir)))


if __name__ == "__main__":
    main()
