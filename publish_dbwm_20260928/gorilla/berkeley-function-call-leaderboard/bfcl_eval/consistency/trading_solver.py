"""Existential state check for successful TradingBot observation trajectories.

Run with ``python -m bfcl_eval.consistency.trading_solver trace.json``.
The input is {"steps": [{"tool": "get_account_info", "args": {},
"observation": {...}}, ...]}. A SAT answer includes a concrete initial state
that has been replayed against the real TradingBot implementation.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import re
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import z3

from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.trading_bot import (
    CURRENT_TIME,
    DEFAULT_STATE,
    TradingBot,
)


SUPPORTED_TOOLS = {
    "add_to_watchlist",
    "remove_stock_from_watchlist",
    "filter_stocks_by_price",
    "notify_price_change",
    "get_order_history",
    "get_transaction_history",
    "get_account_info",
    "get_holdings",
    "get_market_status",
    "get_stock_info",
    "get_watchlist",
    "get_current_time",
    "get_symbol_by_name",
    "get_available_stocks",
    "trading_get_login_status",
    "trading_login",
    "trading_logout",
    "fund_account",
    "withdraw_funds",
    "place_order",
    "get_order_details",
    "activate_order",
    "execute_order",
    "cancel_order",
}


class UnsupportedTrace(ValueError):
    pass


def _cents(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnsupportedTrace("Money and prices must be JSON numbers")
    amount = Decimal(str(value)) * 100
    if amount != amount.to_integral_value():
        raise UnsupportedTrace("This solver accepts monetary values with at most two decimals")
    return int(amount)


def _money(cents: int) -> float:
    return cents / 100


def _tool_name(raw: str) -> str:
    name = raw.removeprefix("TradingBot.")
    if name not in SUPPORTED_TOOLS:
        raise UnsupportedTrace(f"Tool {raw!r} is not modeled")
    return name


def _symbols(steps: list[dict]) -> set[str]:
    result = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        args = step.get("args", {})
        if not isinstance(args, dict):
            continue
        obs = step.get("observation", {})
        if not isinstance(obs, dict):
            continue
        if isinstance(args.get("symbol"), str):
            result.add(args["symbol"])
        if isinstance(args.get("stock"), str):
            result.add(args["stock"])
        if isinstance(args.get("stocks"), list):
            result.update(symbol for symbol in args["stocks"] if isinstance(symbol, str))
        if isinstance(obs.get("symbol"), str):
            result.add(obs["symbol"])
        if isinstance(obs.get("holdings"), dict):
            result.update(obs["holdings"])
        for field in ("watchlist", "filtered_stocks"):
            if isinstance(obs.get(field), list):
                result.update(symbol for symbol in obs[field] if isinstance(symbol, str))
    return result


def _watchlist_candidates(steps: list[dict]) -> list[list[str]]:
    """Infer finite initial lists from the first full list observation."""
    removed_before_anchor = []
    bases = None
    for step in steps:
        if not isinstance(step, dict):
            continue
        name = step.get("tool", "").removeprefix("TradingBot.")
        observation = step.get("observation")
        if name in {"get_watchlist", "add_to_watchlist"} and isinstance(observation, dict):
            reported = observation.get("watchlist")
            if isinstance(reported, list) and all(isinstance(item, str) for item in reported):
                bases = [list(reported)]
                if name == "add_to_watchlist":
                    symbol = step.get("args", {}).get("stock")
                    if reported and reported[-1] == symbol:
                        bases.append(list(reported[:-1]))
                break
        if name == "remove_stock_from_watchlist" and isinstance(observation, dict):
            symbol = step.get("args", {}).get("symbol")
            if observation.get("status") == f"Stock {symbol} removed from watchlist successfully.":
                removed_before_anchor.append(symbol)
    if bases is None:
        bases = [[]]
    candidates = []
    for base in bases:
        initial = list(base)
        for symbol in reversed(removed_before_anchor):
            initial.insert(0, symbol)
        if initial not in candidates:
            candidates.append(initial)
    return candidates


def _initial_order_ids(steps: list[dict]) -> list[int]:
    """Choose a minimal initial order-key list consistent with observed IDs."""
    placed = []
    inferred = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        name = step.get("tool", "").removeprefix("TradingBot.")
        obs = step.get("observation")
        args = step.get("args", {})
        complete = None
        if name == "get_order_history" and isinstance(obs, dict):
            complete = obs.get("history")
        if name == "get_order_details" and isinstance(obs, dict):
            error = obs.get("error", "")
            if isinstance(error, str) and "Here is the list of orders_id: " in error:
                try:
                    complete = ast.literal_eval(error.split("orders_id: ", 1)[1])
                except (SyntaxError, ValueError):
                    pass
        if isinstance(complete, list) and all(isinstance(item, int) for item in complete):
            return [item for item in complete if item not in placed]

        if name == "place_order" and isinstance(obs, dict) and "order_id" in obs:
            order_id = obs["order_id"]
            if isinstance(order_id, int) and not isinstance(order_id, bool):
                if placed:
                    inferred.extend(item for item in range(placed[-1] + 1, order_id)
                                    if item not in inferred)
                placed.append(order_id)
        elif name in {"get_order_details", "activate_order", "execute_order", "cancel_order"}:
            order_id = args.get("order_id") if isinstance(args, dict) else None
            if not isinstance(order_id, int) or order_id in placed or order_id in inferred:
                continue
            if isinstance(obs, dict):
                error = obs.get("error", "")
                if isinstance(error, str) and error.startswith(f"Order with ID {order_id} not found."):
                    continue
                inferred.append(order_id)
    return inferred


class TradingConsistencySolver:
    """Solve for a valid initial cash balance, holdings, market and login state."""

    def __init__(self, steps: list[dict], initial_watchlist: list[str] | None = None):
        self.steps = steps
        self.solver = z3.Solver()
        self.solver.set(timeout=3000)
        self.labels: dict[str, int] = {}
        self.label_counter = 0
        self.initial_balance = z3.Int("initial_balance_cents")
        self.balance = self.initial_balance
        self.initial_auth = z3.Bool("initial_authenticated")
        self.auth = self.initial_auth
        self.initial_market_open = z3.Bool("initial_market_open")
        self.market_open = self.initial_market_open
        self.symbols = sorted(_symbols(steps))
        self.initial_holdings = {
            symbol: z3.Int(f"initial_holdings_{index}")
            for index, symbol in enumerate(self.symbols)
        }
        self.holdings = dict(self.initial_holdings)
        self.prices = {
            symbol: z3.Int(f"initial_price_{index}_cents")
            for index, symbol in enumerate(self.symbols)
        }
        self.stock_exists = {
            symbol: z3.Bool(f"initial_stock_exists_{index}")
            for index, symbol in enumerate(self.symbols)
        }
        self.percent_changes = {
            symbol: z3.Real(f"initial_percent_change_{index}")
            for index, symbol in enumerate(self.symbols)
        }
        self.orders: dict[int, dict[str, Any]] = {}
        self.initial_orders: dict[int, dict[str, Any]] = {}
        self.initial_order_ids = self._sequence(_initial_order_ids(steps), z3.SeqSort(z3.IntSort()))
        self.order_ids = self.initial_order_ids
        self.placed_ids: list[int] = []
        self.first_account_info: dict | None = None
        self.first_stock_info: dict[str, dict] = {}
        watchlist = initial_watchlist if initial_watchlist is not None else _watchlist_candidates(steps)[0]
        self.initial_watchlist = self._sequence(watchlist, z3.SeqSort(z3.StringSort()))
        self.watchlist = self.initial_watchlist
        self.history_candidates: list[dict] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("tool", "").removeprefix("TradingBot.") != "get_transaction_history":
                continue
            observation = step.get("observation", {})
            if isinstance(observation, dict) and isinstance(observation.get("transaction_history"), list):
                for record in observation.get("transaction_history", []):
                    if record not in self.history_candidates:
                        self.history_candidates.append(deepcopy(record))
        self.initial_history_count = z3.Int("initial_history_count")
        self.initial_history_indices = [
            z3.Int(f"initial_history_record_{index}")
            for index in range(sum(
                len(step.get("observation", {}).get("transaction_history", []))
                for step in steps
                if isinstance(step, dict)
                if isinstance(step.get("observation"), dict)
                and isinstance(step.get("observation", {}).get("transaction_history", []), list)
                and step.get("tool", "").removeprefix("TradingBot.") == "get_transaction_history"
            ))
        ]
        self.history_writes: list[dict] = []
        self.solver.add(self.initial_balance >= 0)
        for symbol in self.symbols:
            self.solver.add(self.initial_holdings[symbol] >= 0)
            self.solver.add(self.prices[symbol] > 0)
        self.solver.add(
            self.initial_history_count >= 0,
            self.initial_history_count <= len(self.initial_history_indices),
        )
        for record_index in self.initial_history_indices:
            self.solver.add(record_index >= 0, record_index < len(self.history_candidates))

    @staticmethod
    def _sequence(items: list, sort: z3.SeqSortRef) -> z3.SeqRef:
        if not items:
            return z3.Empty(sort)
        units = [
            z3.Unit(z3.IntVal(item) if isinstance(item, int) else z3.StringVal(item))
            for item in items
        ]
        return units[0] if len(units) == 1 else z3.Concat(*units)

    @staticmethod
    def _sequence_value(value: z3.SeqRef) -> list:
        value = z3.simplify(value)
        if z3.is_app_of(value, z3.Z3_OP_SEQ_EMPTY):
            return []
        if z3.is_app_of(value, z3.Z3_OP_SEQ_UNIT):
            item = value.arg(0)
            return [item.as_long() if z3.is_int_value(item) else item.as_string()]
        if z3.is_app_of(value, z3.Z3_OP_SEQ_CONCAT):
            return [item for part in value.children() for item in TradingConsistencySolver._sequence_value(part)]
        raise UnsupportedTrace(f"Cannot decode symbolic sequence {value}")

    def _require(self, condition: Any, index: int) -> None:
        label = z3.Bool(f"step_{index + 1}_constraint_{self.label_counter}")
        self.label_counter += 1
        if isinstance(condition, bool):
            condition = z3.BoolVal(condition)
        self.solver.assert_and_track(condition, label)
        self.labels[str(label)] = index + 1

    def _same_static(self, first: Any, current: Any, index: int) -> None:
        self._require(z3.BoolVal(first == current), index)

    def _status(self, order: dict, expected: str, index: int) -> None:
        self._require(order["status"] == z3.StringVal(expected), index)

    def _filter_list(
        self, inputs: list[str], outputs: list[str], predicates: list[z3.BoolRef], index: int
    ) -> None:
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            raise UnsupportedTrace("List-valued tool arguments and observations must be arrays")
        count = z3.IntVal(0)
        for symbol, predicate in zip(inputs, predicates):
            matches = [z3.And(count == position, symbol == value)
                       for position, value in enumerate(outputs)]
            self._require(z3.Implies(predicate, z3.Or(*matches)), index)
            count = count + z3.If(predicate, 1, 0)
        self._require(count == len(outputs), index)

    @staticmethod
    def _timestamp_number(value: str) -> int:
        moment = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        return int((moment - datetime(2024, 9, 1)).total_seconds())

    def _record_history_write(self, kind: str, amount: Any) -> None:
        timestamp = z3.Int(f"generated_transaction_time_{len(self.history_writes)}")
        start = int((CURRENT_TIME - datetime(2024, 9, 1)).total_seconds())
        self.solver.add(timestamp >= start, timestamp <= start + 86400)
        self.history_writes.append({"type": kind, "amount": amount, "timestamp": timestamp})

    def _check_transaction_history(self, args: dict, obs: dict, index: int) -> None:
        self._require(self.auth, index)
        output = obs["transaction_history"]
        if not isinstance(output, list):
            raise UnsupportedTrace("transaction_history must be an array")
        start = datetime.strptime(args["start_date"], "%Y-%m-%d") if args.get("start_date") else datetime.min
        end = datetime.strptime(args["end_date"], "%Y-%m-%d") if args.get("end_date") else datetime.max
        low = int((start - datetime(2024, 9, 1)).total_seconds())
        high = int((end - datetime(2024, 9, 1)).total_seconds())
        output_times = [self._timestamp_number(record["timestamp"]) for record in output]
        candidate_times = [self._timestamp_number(record["timestamp"]) for record in self.history_candidates]
        position = z3.IntVal(0)
        for slot, candidate_index in enumerate(self.initial_history_indices):
            included = z3.And(
                slot < self.initial_history_count,
                z3.Or(*[
                    z3.And(candidate_index == choice, low <= moment, moment <= high)
                    for choice, moment in enumerate(candidate_times)
                ]),
            )
            matches = []
            for output_index, record in enumerate(output):
                choices = [candidate_index == choice for choice, candidate in enumerate(self.history_candidates)
                           if candidate == record]
                matches.append(z3.And(position == output_index, z3.Or(*choices)))
            self._require(z3.Implies(included, z3.Or(*matches)), index)
            position = position + z3.If(included, 1, 0)
        for write in self.history_writes:
            moment = write["timestamp"]
            included = z3.And(moment >= low, moment <= high)
            matches = []
            for output_index, record in enumerate(output):
                equal = (
                    set(record) == {"type", "amount", "timestamp"}
                    and record["type"] == write["type"]
                    and record["amount"] == write["amount"]
                )
                matches.append(z3.And(
                    position == output_index,
                    moment == output_times[output_index],
                    z3.BoolVal(equal),
                ))
            self._require(z3.Implies(included, z3.Or(*matches)), index)
            position = position + z3.If(included, 1, 0)
        self._require(position == len(output), index)

    def _process_error(self, name: str, args: dict, obs: Any, index: int) -> bool:
        unauthenticated_lists = {
            "get_watchlist": ["Error: User not authenticated. Please log in to view the watchlist."],
            "get_order_history": [{"error": "User not authenticated. Please log in to view order history."}],
            "get_transaction_history": [{"error": "User not authenticated. Please log in to view transaction history."}],
        }
        if isinstance(obs, list):
            if name not in unauthenticated_lists:
                self._require(z3.BoolVal(False), index)
            else:
                self._require(z3.Not(self.auth), index)
                self._same_static(unauthenticated_lists[name], obs, index)
            return True
        if "error" not in obs:
            return False
        error = obs["error"]
        if not isinstance(error, str):
            self._require(z3.BoolVal(False), index)
            return True

        simple_unauthenticated = {
            "get_account_info": "User not authenticated. Please log in to view account information.",
            "get_holdings": "User not authenticated. Please log in to view holdings.",
            "place_order": "User not authenticated. Please log in to place an order.",
            "fund_account": "User not authenticated. Please log in to fund the account.",
            "withdraw_funds": "User not authenticated. Please log in to make a transaction.",
            "remove_stock_from_watchlist": "User not authenticated. Please log in to modify the watchlist.",
        }
        if name in simple_unauthenticated and error == simple_unauthenticated[name]:
            self._require(z3.Not(self.auth), index)
        elif name == "get_stock_info" and error == f"Stock with symbol '{args['symbol']}' not found.":
            self._require(z3.Not(self.stock_exists[args["symbol"]]), index)
        elif name == "remove_stock_from_watchlist" and error == f"Stock {args['symbol']} not found in watchlist.":
            self._require(self.auth, index)
            self._require(
                z3.IndexOf(self.watchlist, z3.Unit(z3.StringVal(args["symbol"])), 0) < 0,
                index,
            )
        elif name == "fund_account" and error == "Funding amount must be positive.":
            self._require(self.auth, index)
            self._require(z3.BoolVal(args["amount"] <= 0), index)
        elif name == "withdraw_funds":
            self._require(self.auth, index)
            if error == "Market is closed. Transactions are not allowed.":
                self._require(z3.Not(self.market_open), index)
            elif error == "Transaction amount must be positive.":
                self._require(self.market_open, index)
                self._require(z3.BoolVal(args["amount"] <= 0), index)
            elif error == "Insufficient funds for withdrawal.":
                self._require(self.market_open, index)
                self._require(z3.BoolVal(args["amount"] > 0), index)
                self._require(self.balance < _cents(args["amount"]), index)
            else:
                self._require(z3.BoolVal(False), index)
        elif name == "place_order":
            symbol = args["symbol"]
            self._require(self.auth, index)
            if error == f"Invalid stock symbol: {symbol}":
                self._require(z3.Not(self.stock_exists[symbol]), index)
            else:
                self._require(self.stock_exists[symbol], index)
                if error == "Price and amount must be positive values.":
                    self._require(z3.BoolVal(args["price"] <= 0 or args["amount"] <= 0), index)
                else:
                    self._require(z3.BoolVal(args["price"] > 0 and args["amount"] > 0), index)
                    order_type = args["order_type"].lower()
                    if error == "Order type must be Buy or Sell.":
                        self._require(z3.BoolVal(order_type not in {"buy", "sell"}), index)
                    else:
                        self._require(z3.BoolVal(order_type in {"buy", "sell"}), index)
                        if order_type == "sell" and error == "Insufficient shares to place the sell order.":
                            self._require(self.holdings[symbol] < args["amount"], index)
                        elif order_type == "buy":
                            cost = float(args["price"]) * int(args["amount"])
                            prefix = f"Insufficient funds: required ${cost:.2f} but only $"
                            match = re.fullmatch(re.escape(prefix) + r"(-?\d+\.\d{2}) available\.", error)
                            self._require(z3.BoolVal(match is not None), index)
                            if match:
                                self._require(self.balance == _cents(float(match.group(1))), index)
                                self._require(self.balance < _cents(cost), index)
                        else:
                            self._require(z3.BoolVal(False), index)
        elif name in {"get_order_details", "cancel_order", "activate_order", "execute_order"}:
            order_id = args["order_id"]
            if name == "get_order_details" and error.startswith(f"Order with ID {order_id} not found.Here is the list of orders_id: "):
                raw_ids = error.split("orders_id: ", 1)[1]
                try:
                    ids = ast.literal_eval(raw_ids)
                except (SyntaxError, ValueError):
                    ids = None
                self._require(z3.BoolVal(isinstance(ids, list)), index)
                if isinstance(ids, list):
                    self._require(self.order_ids == self._sequence(ids, self.order_ids.sort()), index)
                    self._require(z3.BoolVal(order_id not in ids), index)
            elif error == f"Order with ID {order_id} not found.":
                self._require(
                    z3.IndexOf(self.order_ids, z3.Unit(z3.IntVal(order_id)), 0) < 0,
                    index,
                )
            else:
                if order_id not in self.orders:
                    initial_status = None
                    if name == "cancel_order":
                        if error.endswith("already completed."):
                            initial_status = "Completed"
                        elif error.endswith("already cancelled."):
                            initial_status = "Cancelled"
                    elif name == "activate_order" and "Order is already " in error:
                        initial_status = error.rsplit("Order is already ", 1)[1].rstrip(".").capitalize()
                    elif name == "execute_order" and "Order is " in error:
                        initial_status = error.rsplit("Order is ", 1)[1].rstrip(".").capitalize()
                    elif name == "execute_order" and error == "Market is closed. Orders cannot be executed.":
                        initial_status = "Open"
                    self._ensure_initial_order(name, order_id, index, initial_status)
                self._require(
                    z3.IndexOf(self.order_ids, z3.Unit(z3.IntVal(order_id)), 0) >= 0,
                    index,
                )
                order = self.orders[order_id]
                if name == "cancel_order":
                    if error == f"Can't cancel order {order_id}. Order is already completed.":
                        self._status(order, "Completed", index)
                    elif error == f"Can't cancel order {order_id}. Order is already cancelled.":
                        self._status(order, "Cancelled", index)
                    else:
                        self._require(z3.BoolVal(False), index)
                elif name == "activate_order":
                    prefix = f"Can't activate order {order_id}. Order is already "
                    if error.startswith(prefix) and error.endswith("."):
                        status = error[len(prefix):-1].capitalize()
                        self._require(z3.BoolVal(status != "Pending"), index)
                        self._status(order, status, index)
                    else:
                        self._require(z3.BoolVal(False), index)
                elif name == "execute_order":
                    self._process_execute_error(order, order_id, error, index)
                else:
                    self._require(z3.BoolVal(False), index)
        else:
            self._require(z3.BoolVal(False), index)
        self._same_static({"error": error}, obs, index)
        return True

    def _process_execute_error(self, order: dict, order_id: int, error: str, index: int) -> None:
        prefix = f"Can't execute order {order_id}. Order is "
        if error.startswith(prefix) and error.endswith("."):
            status = error[len(prefix):-1].capitalize()
            self._require(z3.BoolVal(status != "Open"), index)
            self._status(order, status, index)
            return
        self._status(order, "Open", index)
        if error == "Market is closed. Orders cannot be executed.":
            self._require(z3.Not(self.market_open), index)
            return
        self._require(self.market_open, index)
        symbol = order["symbol"]
        if error == f"Stock with symbol '{symbol}' not found.":
            self._require(z3.Not(self.stock_exists[symbol]), index)
            return
        self._require(self.stock_exists[symbol], index)
        price = self.prices[symbol]
        limit = order["limit"]
        kind = order["order_type"]
        if kind in {"Buy", "Sell"}:
            label = "Buy" if kind == "Buy" else "Sell"
            phrase = "below" if kind == "Buy" else "above"
            prefix = f"{label} limit price of ${_money(limit):.2f} is {phrase} the current stock price of $"
            match = re.fullmatch(re.escape(prefix) + r"(-?\d+\.\d{2})\.", error)
            if match:
                self._require(price == _cents(float(match.group(1))), index)
                self._require(price > limit if kind == "Buy" else price < limit, index)
                return
            self._require(price <= limit if kind == "Buy" else price >= limit, index)
            if kind == "Buy" and error == "Insufficient funds to execute the buy order.":
                self._require(self.balance < price * order["amount"], index)
                return
            if kind == "Sell" and error == "Insufficient shares to execute the sell order.":
                self._require(self.holdings[symbol] < order["amount"], index)
                return
        elif error == f"Invalid order type: {kind}":
            return
        self._require(z3.BoolVal(False), index)

    def _ensure_initial_order(
        self, name: str, order_id: int, index: int, initial_status: str | None = None
    ) -> dict:
        if order_id in self.orders:
            return self.orders[order_id]
        future_record = None
        for later in self.steps[index + 1:]:
            if not isinstance(later, dict):
                continue
            if later.get("tool", "").removeprefix("TradingBot.") != "get_order_details":
                continue
            if later.get("args", {}).get("order_id") == order_id:
                candidate = later.get("observation")
                if isinstance(candidate, dict) and "error" not in candidate:
                    future_record = deepcopy(candidate)
                    break
        if name == "execute_order" and future_record is None and initial_status is None:
            raise UnsupportedTrace("An existing order must be read before execution or detailed afterward")
        record = future_record or {
            "id": order_id, "order_type": "Buy", "symbol": "AAPL",
            "price": 1.0, "amount": 1,
        }
        if initial_status != "Completed":
            record.pop("filled_price", None)
        record["status"] = initial_status or ("Open" if name == "execute_order" else "Pending")
        self.initial_orders[order_id] = deepcopy(record)
        self.orders[order_id] = {
            "id": order_id,
            "order_type": record["order_type"],
            "symbol": record["symbol"],
            "limit": _cents(record["price"]),
            "amount": record["amount"],
            "status": z3.StringVal(record["status"]),
            "filled_price": _cents(record["filled_price"]) if "filled_price" in record else None,
        }
        self._require(
            z3.IndexOf(self.initial_order_ids, z3.Unit(z3.IntVal(order_id)), 0) >= 0,
            index,
        )
        return self.orders[order_id]

    def _process(self, index: int, step: dict) -> None:
        if not isinstance(step, dict):
            raise UnsupportedTrace("Each step must be an object")
        name = _tool_name(step.get("tool", ""))
        args = step.get("args", {})
        obs = step.get("observation", {})
        if not isinstance(args, dict) or not isinstance(obs, (dict, list)):
            raise UnsupportedTrace("args must be an object and observation an object or error list")
        try:
            inspect.signature(getattr(TradingBot, name)).bind(None, **args)
        except TypeError as exc:
            raise UnsupportedTrace(f"Invalid tool arguments: {exc}") from exc
        if self._process_error(name, args, obs, index):
            return
        exact_fields = {
            "get_holdings": {"holdings"},
            "get_market_status": {"market_status"},
            "get_watchlist": {"watchlist"},
            "add_to_watchlist": {"watchlist"},
            "remove_stock_from_watchlist": {"status"},
            "filter_stocks_by_price": {"filtered_stocks"},
            "notify_price_change": {"notification"},
            "get_order_history": {"history"},
            "get_transaction_history": {"transaction_history"},
            "trading_get_login_status": {"status"},
            "trading_login": {"status"},
            "trading_logout": {"status"},
            "fund_account": {"status", "new_balance"},
            "withdraw_funds": {"status", "new_balance"},
            "place_order": {"order_id", "order_type", "status", "price", "amount"},
            "activate_order": {"order_id", "status"},
            "cancel_order": {"order_id", "status"},
            "execute_order": {"order_id", "status", "filled_price", "amount", "new_balance", "shares_held"},
        }
        if name in exact_fields:
            self._require(z3.BoolVal(set(obs) == exact_fields[name]), index)

        if name == "get_account_info":
            self._require(self.auth, index)
            self._require(self.balance == _cents(obs["balance"]), index)
            static = {key: value for key, value in obs.items() if key != "balance"}
            if self.first_account_info is None:
                self.first_account_info = static
            else:
                self._same_static(self.first_account_info, static, index)
        elif name == "get_holdings":
            self._require(self.auth, index)
            observed = obs["holdings"]
            if not isinstance(observed, dict):
                raise UnsupportedTrace("holdings must be an object")
            for symbol in self.symbols:
                count = observed.get(symbol, 0)
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    raise UnsupportedTrace("Holdings must be nonnegative integer share counts")
                self._require(self.holdings[symbol] == count, index)
        elif name == "get_market_status":
            status = obs["market_status"]
            if status not in {"Open", "Closed"}:
                self._require(z3.BoolVal(False), index)
            else:
                self._require(self.market_open == (status == "Open"), index)
        elif name == "get_stock_info":
            symbol = args["symbol"]
            self._require(self.stock_exists[symbol], index)
            self._require(self.prices[symbol] == _cents(obs["price"]), index)
            self._require(
                self.percent_changes[symbol] == z3.RealVal(str(obs["percent_change"])),
                index,
            )
            if symbol in self.first_stock_info:
                self._same_static(self.first_stock_info[symbol], obs, index)
            else:
                self.first_stock_info[symbol] = deepcopy(obs)
        elif name == "get_watchlist":
            self._require(self.auth, index)
            watchlist = obs["watchlist"]
            if not isinstance(watchlist, list):
                raise UnsupportedTrace("watchlist must be an array")
            self._require(
                self.watchlist == self._sequence(watchlist, self.watchlist.sort()), index
            )
            self.watchlist = self._sequence(watchlist, self.watchlist.sort())
        elif name == "add_to_watchlist":
            symbol = args["stock"]
            member = z3.IndexOf(self.watchlist, z3.Unit(z3.StringVal(symbol)), 0) >= 0
            self.watchlist = z3.If(
                z3.Or(member, z3.Not(self.stock_exists[symbol])),
                self.watchlist,
                z3.Concat(self.watchlist, z3.Unit(z3.StringVal(symbol))),
            )
            self._require(
                self.watchlist == self._sequence(obs["watchlist"], self.watchlist.sort()),
                index,
            )
            self.watchlist = self._sequence(obs["watchlist"], self.watchlist.sort())
        elif name == "remove_stock_from_watchlist":
            symbol = args["symbol"]
            needle = z3.Unit(z3.StringVal(symbol))
            position = z3.IndexOf(self.watchlist, needle, 0)
            self._require(self.auth, index)
            self._require(position >= 0, index)
            self.watchlist = z3.Concat(
                z3.SubSeq(self.watchlist, 0, position),
                z3.SubSeq(self.watchlist, position + 1, z3.Length(self.watchlist)),
            )
            self.watchlist = z3.simplify(self.watchlist)
            self._same_static(
                {"status": f"Stock {symbol} removed from watchlist successfully."},
                obs,
                index,
            )
        elif name == "filter_stocks_by_price":
            inputs = args["stocks"]
            outputs = obs["filtered_stocks"]
            low = z3.RealVal(str(args["min_price"]))
            high = z3.RealVal(str(args["max_price"]))
            predicates = []
            for symbol in inputs:
                price = z3.If(
                    self.stock_exists[symbol], z3.ToReal(self.prices[symbol]) / 100,
                    z3.RealVal(0),
                )
                predicates.append(z3.And(price >= low, price <= high))
            self._filter_list(inputs, outputs, predicates, index)
        elif name == "notify_price_change":
            inputs = args["stocks"]
            notification = obs["notification"]
            prefix = "Stocks "
            suffix = " have significant price changes."
            if notification == "No significant price changes in the selected stocks.":
                outputs = []
            elif notification.startswith(prefix) and notification.endswith(suffix):
                outputs = notification[len(prefix):-len(suffix)].split(", ")
            else:
                self._require(z3.BoolVal(False), index)
                outputs = []
            threshold = z3.RealVal(str(args["threshold"]))
            predicates = []
            for symbol in inputs:
                change = self.percent_changes[symbol]
                absolute = z3.If(change >= 0, change, -change)
                predicates.append(z3.And(self.stock_exists[symbol], absolute >= threshold))
            self._filter_list(inputs, outputs, predicates, index)
        elif name == "get_order_history":
            self._require(self.auth, index)
            history = obs["history"]
            self._require(z3.BoolVal(
                isinstance(history, list)
                and all(isinstance(item, int) and not isinstance(item, bool) for item in history)
                and len(history) == len(set(history))
            ), index)
            if not isinstance(history, list) or any(not isinstance(item, int) for item in history):
                return
            self._require(
                self.order_ids == self._sequence(history, self.order_ids.sort()),
                index,
            )
        elif name == "get_transaction_history":
            self._check_transaction_history(args, obs, index)
        elif name == "get_current_time":
            self._same_static({"current_time": CURRENT_TIME.strftime("%I:%M %p")}, obs, index)
        elif name in {"get_symbol_by_name", "get_available_stocks"}:
            # These methods use fixed tables and do not read mutable backend state.
            reference = TradingBot()
            reference._load_scenario({})
            self._same_static(getattr(reference, name)(**args), obs, index)
        elif name == "trading_get_login_status":
            self._require(self.auth == z3.BoolVal(obs["status"]), index)
        elif name == "trading_login":
            expected = z3.If(
                self.auth,
                z3.StringVal("Already logged in"),
                z3.StringVal("Logged in successfully"),
            )
            self._require(expected == z3.StringVal(obs["status"]), index)
            self.auth = z3.BoolVal(True)
        elif name == "trading_logout":
            expected = z3.If(
                self.auth,
                z3.StringVal("Logged out successfully"),
                z3.StringVal("No user is currently logged in"),
            )
            self._require(expected == z3.StringVal(obs["status"]), index)
            self.auth = z3.BoolVal(False)
        elif name == "fund_account":
            amount = _cents(args["amount"])
            self._require(self.auth, index)
            self._require(z3.BoolVal(amount > 0), index)
            self.balance = self.balance + amount
            self._same_static("Account funded successfully", obs["status"], index)
            self._require(self.balance == _cents(obs["new_balance"]), index)
            self._record_history_write("deposit", args["amount"])
        elif name == "withdraw_funds":
            amount = _cents(args["amount"])
            self._require(self.auth, index)
            self._require(self.market_open, index)
            self._require(z3.BoolVal(amount > 0), index)
            self._require(self.balance >= amount, index)
            self.balance = self.balance - amount
            self._same_static("Withdrawal successful", obs["status"], index)
            self._require(self.balance == _cents(obs["new_balance"]), index)
            self._record_history_write("withdrawal", args["amount"])
        elif name == "place_order":
            symbol = args["symbol"]
            order_type = args["order_type"].capitalize()
            amount = args["amount"]
            limit = _cents(args["price"])
            order_id = obs["order_id"]
            if not isinstance(order_id, int) or isinstance(order_id, bool):
                raise UnsupportedTrace("Placed order IDs must be integers")
            if self.placed_ids:
                next_id = self.placed_ids[-1] + 1
                self._require(z3.BoolVal(order_id >= next_id), index)
                for skipped_id in range(next_id, order_id):
                    self._require(
                        z3.IndexOf(self.order_ids, z3.Unit(z3.IntVal(skipped_id)), 0) >= 0,
                        index,
                    )
            if order_id in self.orders:
                self._require(z3.BoolVal(False), index)
            self._require(
                z3.IndexOf(self.order_ids, z3.Unit(z3.IntVal(order_id)), 0) < 0,
                index,
            )
            self._require(self.auth, index)
            self._require(self.stock_exists[symbol], index)
            self._require(z3.BoolVal(order_type in {"Buy", "Sell"}), index)
            self._require(z3.BoolVal(amount > 0 and limit > 0), index)
            if order_type == "Buy":
                self._require(self.balance >= limit * amount, index)
            elif order_type == "Sell":
                self._require(self.holdings[symbol] >= amount, index)
            self._same_static(
                {"order_id": order_id, "order_type": order_type, "status": "Pending",
                 "price": args["price"], "amount": amount},
                obs,
                index,
            )
            self.orders[order_id] = {
                "id": order_id, "order_type": order_type, "symbol": symbol,
                "limit": limit, "amount": amount,
                "status": z3.StringVal("Pending"), "filled_price": None,
            }
            self.placed_ids.append(order_id)
            self.order_ids = z3.Concat(self.order_ids, z3.Unit(z3.IntVal(order_id)))
        elif name == "get_order_details":
            order_id = args["order_id"]
            if order_id not in self.orders:
                self._require(
                    z3.IndexOf(self.initial_order_ids, z3.Unit(z3.IntVal(order_id)), 0) >= 0,
                    index,
                )
                needed = {"id", "order_type", "symbol", "price", "amount", "status"}
                if not needed <= obs.keys():
                    raise UnsupportedTrace("An initial order read must include its full order record")
                if obs["id"] != order_id:
                    self._require(z3.BoolVal(False), index)
                self.initial_orders[order_id] = deepcopy(obs)
                self.orders[order_id] = {
                    "id": order_id, "order_type": obs["order_type"],
                    "symbol": obs["symbol"], "limit": _cents(obs["price"]),
                    "amount": obs["amount"],
                    "status": z3.String(f"initial_order_{order_id}_status"),
                    "filled_price": _cents(obs["filled_price"]) if "filled_price" in obs else None,
                }
            order = self.orders[order_id]
            self._status(order, obs["status"], index)
            self._same_static(order_id, obs["id"], index)
            self._same_static(order["order_type"], obs["order_type"], index)
            self._same_static(order["symbol"], obs["symbol"], index)
            self._same_static(order["limit"], _cents(obs["price"]), index)
            self._same_static(order["amount"], obs["amount"], index)
            if order["filled_price"] is None:
                self._require(z3.BoolVal("filled_price" not in obs), index)
            else:
                self._require(z3.BoolVal("filled_price" in obs), index)
                if "filled_price" in obs:
                    self._require(order["filled_price"] == _cents(obs["filled_price"]), index)
        elif name in {"activate_order", "cancel_order", "execute_order"}:
            order_id = args["order_id"]
            if order_id not in self.orders:
                self._ensure_initial_order(name, order_id, index)
            order = self.orders[order_id]
            if name == "activate_order":
                self._status(order, "Pending", index)
                self._same_static({"order_id": order_id, "status": "Open"}, obs, index)
                order["status"] = z3.StringVal("Open")
            elif name == "cancel_order":
                self._require(z3.Or(order["status"] == "Pending", order["status"] == "Open"), index)
                self._same_static({"order_id": order_id, "status": "Cancelled"}, obs, index)
                order["status"] = z3.StringVal("Cancelled")
            else:
                self._status(order, "Open", index)
                self._require(self.market_open, index)
                symbol = order["symbol"]
                self._require(self.stock_exists[symbol], index)
                price = self.prices[symbol]
                amount = order["amount"]
                if order["order_type"] == "Buy":
                    self._require(price <= order["limit"], index)
                    self._require(self.balance >= price * amount, index)
                    self.balance = self.balance - price * amount
                    self.holdings[symbol] = self.holdings[symbol] + amount
                elif order["order_type"] == "Sell":
                    self._require(price >= order["limit"], index)
                    self._require(self.holdings[symbol] >= amount, index)
                    self.balance = self.balance + price * amount
                    self.holdings[symbol] = self.holdings[symbol] - amount
                else:
                    self._require(z3.BoolVal(False), index)
                order["status"] = z3.StringVal("Completed")
                order["filled_price"] = price
                self._same_static(order_id, obs["order_id"], index)
                self._same_static("Completed", obs["status"], index)
                self._same_static(amount, obs["amount"], index)
                self._require(price == _cents(obs["filled_price"]), index)
                self._require(self.balance == _cents(obs["new_balance"]), index)
                self._require(self.holdings[symbol] == obs["shares_held"], index)

    def _witness(self, model: z3.ModelRef) -> tuple[dict, dict]:
        def value(expr: Any) -> int:
            return model.eval(expr, model_completion=True).as_long()

        def real_value(expr: Any) -> float:
            number = model.eval(expr, model_completion=True)
            return number.numerator_as_long() / number.denominator_as_long()

        balance = value(self.initial_balance)
        holdings = {
            symbol: value(expr)
            for symbol, expr in self.initial_holdings.items()
            if value(expr) > 0
        }
        stock_prices = {symbol: _money(value(expr)) for symbol, expr in self.prices.items()}
        authenticated = z3.is_true(model.eval(self.initial_auth, model_completion=True))
        market_open = z3.is_true(model.eval(self.initial_market_open, model_completion=True))
        initial_order_ids = self._sequence_value(model.eval(self.initial_order_ids, model_completion=True))
        order_counter = self.placed_ids[0] if self.placed_ids else max(initial_order_ids, default=0) + 1

        scenario = deepcopy(DEFAULT_STATE)
        scenario["orders"] = {
            order_id: deepcopy(self.initial_orders.get(order_id, {
                "id": order_id, "order_type": "Buy", "symbol": "AAPL",
                "price": 1.0, "amount": 1, "status": "Pending",
            }))
            for order_id in initial_order_ids
        }
        scenario["order_counter"] = order_counter
        scenario["authenticated"] = authenticated
        scenario["market_status"] = "Open" if market_open else "Closed"
        scenario["holdings"] = holdings
        scenario["account_info"]["balance"] = _money(balance)
        if self.first_account_info:
            scenario["account_info"].update(self.first_account_info)
        scenario["watch_list"] = self._sequence_value(
            model.eval(self.initial_watchlist, model_completion=True)
        )
        scenario["transaction_history"] = [
            deepcopy(self.history_candidates[value(self.initial_history_indices[index])])
            for index in range(value(self.initial_history_count))
        ]
        generated_timestamps = [
            (datetime(2024, 9, 1) + timedelta(seconds=value(write["timestamp"])))
            .strftime("%Y-%m-%d %H:%M:%S")
            for write in self.history_writes
        ]
        scenario["stocks"] = {}
        for symbol, price in stock_prices.items():
            if not z3.is_true(model.eval(self.stock_exists[symbol], model_completion=True)):
                continue
            info = deepcopy(DEFAULT_STATE["stocks"].get(symbol, {
                "volume": 0.0, "MA(5)": price, "MA(20)": price,
            }))
            info["price"] = price
            info["percent_change"] = real_value(self.percent_changes[symbol])
            if symbol in self.first_stock_info:
                info = deepcopy(self.first_stock_info[symbol])
            scenario["stocks"][symbol] = info
        summary = {
            "balance": _money(balance), "holdings": holdings,
            "authenticated": authenticated, "market_status": scenario["market_status"],
            "stock_prices": stock_prices, "initial_orders": self.initial_orders,
            "order_counter": order_counter,
            "watchlist": scenario["watch_list"],
            "generated_transaction_timestamps": generated_timestamps,
        }
        return scenario, summary

    def solve(self) -> dict:
        for index, step in enumerate(self.steps):
            try:
                self._process(index, step)
            except KeyError as exc:
                return {
                    "status": "unsat", "first_conflicting_step": index + 1,
                    "conflicting_steps": [index + 1],
                    "reason": f"Successful observation is missing required field {exc}",
                }
            except (UnsupportedTrace, TypeError, ValueError) as exc:
                return {"status": "unsupported", "step": index + 1, "reason": str(exc)}
            result = self.solver.check()
            if result == z3.unsat:
                conflicting = sorted({self.labels[str(x)] for x in self.solver.unsat_core() if str(x) in self.labels})
                return {"status": "unsat", "first_conflicting_step": index + 1, "conflicting_steps": conflicting}
            if result == z3.unknown:
                return {"status": "unknown", "step": index + 1, "reason": self.solver.reason_unknown()}

        scenario, summary = self._witness(self.solver.model())
        backend = TradingBot()
        backend._load_scenario(scenario)
        timestamps = iter(summary["generated_transaction_timestamps"])
        backend._generate_transaction_timestamp = lambda: next(timestamps)
        for index, step in enumerate(self.steps):
            name = _tool_name(step["tool"])
            try:
                actual = getattr(backend, name)(**step.get("args", {}))
            except Exception as exc:
                return {
                    "status": "unknown", "step": index + 1,
                    "reason": f"SMT witness raised {type(exc).__name__} during backend replay: {exc}",
                }
            if actual != step["observation"]:
                return {
                    "status": "unknown", "step": index + 1,
                    "reason": "SMT witness did not replay exactly on the backend",
                    "backend_observation": actual,
                }
        return {"status": "sat", "initial_state": summary}


def check_trace(steps: list[dict]) -> dict:
    """Check one observation trace without assuming a fixed initial balance."""
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        name = step.get("tool", "").removeprefix("TradingBot.")
        observation = step.get("observation")
        if name == "get_watchlist" and isinstance(observation, list):
            if observation != ["Error: User not authenticated. Please log in to view the watchlist."]:
                return {"status": "unsupported", "step": index + 1,
                        "reason": "long_context watchlist observations are not modeled"}
        if not isinstance(observation, dict):
            continue
        if name == "get_stock_info" and isinstance(observation.get("MA(5)"), str):
            return {"status": "unsupported", "step": index + 1,
                    "reason": "long_context stock observations are not modeled"}
        if name == "get_order_details" and "metadata" in observation:
            return {"status": "unsupported", "step": index + 1,
                    "reason": "long_context order observations are not modeled"}
        if name == "get_transaction_history" and isinstance(observation.get("transaction_history"), list):
            if len(observation["transaction_history"]) >= 1000:
                return {"status": "unsupported", "step": index + 1,
                        "reason": "large or long_context transaction histories are not modeled"}
        if name == "get_available_stocks" and "stock_list" in observation:
            reference = TradingBot()
            reference._load_scenario({}, long_context=True)
            try:
                long_output = reference.get_available_stocks(**step.get("args", {}))
            except TypeError:
                continue
            if len(long_output["stock_list"]) > 10 and observation == long_output:
                return {"status": "unsupported", "step": index + 1,
                        "reason": "long_context stock lists are not modeled"}
    outcomes = [
        TradingConsistencySolver(steps, watchlist).solve()
        for watchlist in _watchlist_candidates(steps)
    ]
    for outcome in outcomes:
        if outcome["status"] == "sat":
            return outcome
    for status in ("unknown", "unsupported"):
        for outcome in outcomes:
            if outcome["status"] == status:
                return outcome
    return max(outcomes, key=lambda outcome: outcome.get("first_conflicting_step", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="JSON file containing a steps array")
    args = parser.parse_args()
    data = json.loads(args.trace.read_text(encoding="utf-8-sig"))
    print(json.dumps(check_trace(data["steps"]), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
