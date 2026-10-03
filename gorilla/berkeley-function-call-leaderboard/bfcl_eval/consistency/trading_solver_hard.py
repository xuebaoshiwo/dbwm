"""Existential state check for successful TradingBot observation trajectories.

Run with ``python -m bfcl_eval.consistency.trading_solver_hard trace.json``.
The input is {"steps": [{"tool": "get_account_info", "args": {},
"observation": {...}}, ...]}. A SAT answer includes a concrete initial state
that has been replayed against the real TradingBot implementation, with
symbolic transaction timestamps and a 1e-9 numeric comparison tolerance.
Holdings observations treat missing symbols and zero-share entries as equivalent.
Fixed time/name/stock-list tools are excluded by default. See HARD_SOLVER.md
for the checking policies and limits; invalid_format/unknown are not UNSAT.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import re
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
from pathlib import Path
from typing import Any

import z3

from bfcl_eval.consistency.trading_structures import Watchlist, OrderKeys, order_ids
from bfcl_eval.consistency.trading_history import History
from bfcl_eval.consistency.trading_trace_format import validate
from bfcl_eval.consistency.trading_float import (
    F64, RNE, as_float, finite, fp, observed as number_matches, rounded, three_decimals,
)

from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.trading_bot_hard import (
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

# These calls neither read nor change the inferred mutable state.
STATIC_TOOLS = {"get_current_time", "get_symbol_by_name", "get_available_stocks"}


def _observations_equal(left: Any, right: Any) -> bool:
    """Ignore only floating-point residue; keep a 0.001 difference visible."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right or abs(Decimal(str(left)) - Decimal(str(right))) <= Decimal("0.000000001")
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_observations_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_observations_equal(a, b) for a, b in zip(left, right))
    return left == right


def _replay_observations_equal(tool: str, actual: Any, observed: Any) -> bool:
    """Ignore zero-key representation only in successful holdings observations."""
    if tool == "get_holdings":
        def normalize(value):
            if (isinstance(value, dict) and "error" not in value
                    and isinstance(value.get("holdings"), dict)):
                return {**value, "holdings": {
                    symbol: count for symbol, count in value["holdings"].items()
                    if not (type(count) is int and count == 0)
                }}
            return value
        actual, observed = normalize(actual), normalize(observed)
    return _observations_equal(actual, observed)


class UnsupportedTrace(ValueError):
    pass


def _units(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnsupportedTrace("Money and prices must be JSON numbers")
    amount = Decimal(str(value)) * 1000
    if not amount.is_finite():
        raise UnsupportedTrace("Money and prices must be finite")
    nearest = amount.to_integral_value()
    # Accommodate binary64 residue, not an extra decimal of model error.
    if abs(amount - nearest) > Decimal("0.000001"):
        raise UnsupportedTrace("This solver accepts monetary values with at most three decimals")
    return int(nearest)


def _money(units: int) -> float:
    return units / 1000


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
        error = obs.get("error", "")
        if isinstance(error, str):
            match = re.fullmatch(r"Stock with symbol '(.*)' not found\.", error)
            if match:
                result.add(match.group(1))
    return result


class TradingConsistencySolver:
    """Solve for a valid initial cash balance, holdings, market and login state."""

    def __init__(self, steps: list[dict], initial_watchlist: list[str] | None = None,
                 *, check_static_tools: bool = False, timestamp_policy: str = "symbolic"):
        if timestamp_policy not in {"symbolic", "backend"}:
            raise ValueError("timestamp_policy must be 'symbolic' or 'backend'")
        self.steps = steps
        self.check_static_tools = check_static_tools
        self.timestamp_policy = timestamp_policy
        self.solver = z3.Solver()
        self.solver.set(timeout=0)
        self.labels: dict[str, int] = {}
        self.label_counter = 0
        self.initial_balance = z3.FP("initial_balance", F64)
        cash_tools = {"get_account_info", "fund_account", "withdraw_funds", "place_order", "execute_order"}
        if not any(step.get("tool", "").removeprefix("TradingBot.") in cash_tools for step in steps):
            self.initial_balance = fp(0)
        for step in steps:
            name = step.get("tool", "").removeprefix("TradingBot.")
            obs = step.get("observation", {})
            if not isinstance(obs, dict) or "error" in obs:
                continue
            if name == "get_account_info" and "balance" in obs:
                # On the declared 0.001 grid, a 1e-9 observation interval has
                # at most one initial value. Substitute that proven constant.
                self.initial_balance = fp(_money(_units(obs["balance"])))
                break
            if name in {"fund_account", "withdraw_funds", "execute_order"}:
                break
        # If cash is never observed and no cash-dependent error occurs, a
        # sufficiently funded account is an existential witness for every
        # successful debit. This removes a free floating-point search variable.
        cash_unobserved = True
        budget = 1.0
        for step in steps:
            name = step.get("tool", "").removeprefix("TradingBot.")
            args, obs = step.get("args", {}), step.get("observation", {})
            if name == "get_account_info" or (name in cash_tools and isinstance(obs, dict) and "error" in obs):
                cash_unobserved = False
            if isinstance(obs, dict) and "error" not in obs:
                if name == "withdraw_funds":
                    budget += abs(args["amount"])
                elif name == "place_order":
                    budget += abs(float(args["price"]) * args["amount"])
                elif name == "execute_order":
                    budget += abs(float(obs["filled_price"]) * obs["amount"])
        if cash_unobserved and budget < 1e12 and all(
            abs(step.get("args", {}).get("amount", 0)) < 1e12
            for step in steps if step.get("tool", "").removeprefix("TradingBot.") == "fund_account"
        ):
            self.initial_balance = fp(math.ceil(budget * 2))
        self.balance = self.initial_balance
        self.initial_auth = z3.Bool("initial_authenticated")
        self.auth = self.initial_auth
        self.initial_market_open = z3.Bool("initial_market_open")
        self.market_open = self.initial_market_open
        self.symbols = sorted(_symbols([
            step for step in steps
            if not self._ignored(step)
        ]))
        # Unobserved symbols are existential too. One fresh representative per
        # initial order suffices: only equality and the per-symbol state matter.
        for order_id in order_ids(steps):
            fresh = f"__unobserved_stock_{order_id}"
            while fresh in self.symbols:
                fresh += "_"
            self.symbols.append(fresh)
        self.initial_holdings = {
            symbol: z3.Int(f"initial_holdings_{index}")
            for index, symbol in enumerate(self.symbols)
        }
        self.holdings = dict(self.initial_holdings)
        self.initial_holding_present = {
            symbol: z3.Bool(f"initial_holding_present_{index}")
            for index, symbol in enumerate(self.symbols)
        }
        self.holding_present = dict(self.initial_holding_present)
        stock_reads = {}
        price_needed, change_needed = set(), set()
        any_execution = False
        for step in steps:
            name = step.get("tool", "").removeprefix("TradingBot.")
            args, obs = step.get("args", {}), step.get("observation", {})
            if name == "get_stock_info" and isinstance(obs, dict) and "price" in obs:
                stock_reads.setdefault(args["symbol"], obs)
            if name == "filter_stocks_by_price":
                price_needed.update(args.get("stocks", []))
            if name == "notify_price_change":
                change_needed.update(args.get("stocks", []))
            if name == "execute_order":
                any_execution = True
        fill_prices = sorted({_money(_units(step["observation"]["filled_price"]))
                              for step in steps if step.get("tool", "").removeprefix("TradingBot.") == "execute_order"
                              and isinstance(step.get("observation"), dict) and "filled_price" in step["observation"]})
        fill_only = any_execution and not any(
            step.get("tool", "").removeprefix("TradingBot.") == "execute_order"
            and isinstance(step.get("observation"), dict) and "error" in step["observation"] for step in steps)
        known_order_symbols, fixed_fill_prices = {}, {}
        for step in steps:
            name = step.get("tool", "").removeprefix("TradingBot.")
            obs, args = step.get("observation", {}), step.get("args", {})
            if not isinstance(obs, dict) or "error" in obs:
                continue
            if name == "get_order_details" and "symbol" in obs:
                known_order_symbols[args["order_id"]] = obs["symbol"]
            elif name == "place_order" and "order_id" in obs:
                known_order_symbols[obs["order_id"]] = args["symbol"]
        for step in steps:
            obs, args = step.get("observation", {}), step.get("args", {})
            if step.get("tool", "").removeprefix("TradingBot.") == "execute_order" and isinstance(obs, dict) and "filled_price" in obs:
                symbol = known_order_symbols.get(args["order_id"])
                if symbol is not None:
                    fixed_fill_prices.setdefault(symbol, _money(_units(obs["filled_price"])))
        self.prices = {
            symbol: (fp(_money(_units(stock_reads[symbol]["price"]))) if symbol in stock_reads else
                     z3.FP(f"initial_price_{index}", F64) if any_execution or symbol in price_needed else fp(1))
            for index, symbol in enumerate(self.symbols)
        }
        thresholds = []
        for step in steps:
            if step.get("tool", "").removeprefix("TradingBot.") == "filter_stocks_by_price":
                thresholds.extend([step["args"]["min_price"], step["args"]["max_price"]])
        if not any_execution or fill_only:
            options = {1.0, .001, *fill_prices}
            for threshold in thresholds:
                decimal = Decimal(str(threshold)) * 1000
                for integral in (decimal.to_integral_value(rounding=ROUND_FLOOR), decimal.to_integral_value(rounding=ROUND_CEILING)):
                    for offset in (-1, 0, 1):
                        candidate = float((integral + offset) / 1000)
                        if candidate > 0 and math.isfinite(candidate):
                            options.add(candidate)
                for direction in (-math.inf, math.inf):
                    candidate = round(math.nextafter(float(threshold), direction), 3)
                    if candidate > 0 and math.isfinite(candidate):
                        options.add(candidate)
            options = sorted(options)
            for i, symbol in enumerate(self.symbols):
                if symbol in stock_reads or symbol in fixed_fill_prices:
                    continue
                choice = z3.Int(f"stock_price_choice_{i}")
                self.solver.add(choice >= 0, choice < len(options))
                expression = fp(options[-1])
                for j in reversed(range(len(options) - 1)):
                    expression = z3.If(choice == j, fp(options[j]), expression)
                self.prices[symbol] = expression
        for symbol, price in fixed_fill_prices.items():
            if symbol not in stock_reads:
                self.prices[symbol] = fp(price)
        self.stock_exists = {
            symbol: z3.Bool(f"initial_stock_exists_{index}")
            for index, symbol in enumerate(self.symbols)
        }
        self.percent_changes = {
            symbol: (fp(_money(_units(stock_reads[symbol]["percent_change"]))) if symbol in stock_reads else
                     z3.FP(f"initial_percent_change_{index}", F64) if symbol in change_needed else fp(0))
            for index, symbol in enumerate(self.symbols)
        }
        self.orders: dict[int, dict[str, Any]] = {}
        self.initial_orders: dict[int, dict[str, Any]] = {}
        self.order_keys = OrderKeys(order_ids(steps))
        self.placed_ids: list[int] = []
        self.first_account_info: dict | None = None
        self.first_stock_info: dict[str, dict] = {}
        self.watchlist = Watchlist(steps, self.symbols, self.solver, initial_watchlist)
        self.history = History(steps, self.solver)
        self.history_writes: list[dict] = []
        self.solver.add(self.initial_balance >= fp(0), three_decimals(self.initial_balance))
        for symbol in self.symbols:
            self.solver.add(self.initial_holdings[symbol] >= 0)
            self.solver.add(z3.Implies(z3.Not(self.initial_holding_present[symbol]),
                                     self.initial_holdings[symbol] == 0))
            self.solver.add(self.prices[symbol] > fp(0), three_decimals(self.prices[symbol]),
                            three_decimals(self.percent_changes[symbol]))
    def _require(self, condition: Any, index: int) -> None:
        label = z3.Bool(f"step_{index + 1}_constraint_{self.label_counter}")
        self.label_counter += 1
        if isinstance(condition, bool):
            condition = z3.BoolVal(condition)
        self.solver.assert_and_track(condition, label)
        self.labels[str(label)] = index + 1

    def _same_static(self, first: Any, current: Any, index: int) -> None:
        self._require(z3.BoolVal(_observations_equal(first, current)), index)

    def _ignored(self, step: Any) -> bool:
        return (not self.check_static_tools and isinstance(step, dict)
                and step.get("tool", "").removeprefix("TradingBot.") in STATIC_TOOLS)

    def _cash_guard(self, cost: Any, *, insufficient: bool) -> Any:
        return fp(cost) > self.balance if insufficient else fp(cost) <= self.balance

    def _formatted_money(self, value: Any, text: str, index: int) -> None:
        # .2f and round(..., 2) use the same exact decimal ties-to-even rule.
        target = fp(float(text))
        self._require(z3.fpEQ(rounded(value, 2), target), index)
        if float(text) == 0:
            self._require(z3.fpIsNegative(value) == text.startswith("-"), index)

    def _float_amount(self, amount: Any, index: int) -> Any:
        if isinstance(amount, int):
            try:
                return fp(float(amount))
            except OverflowError:
                self._require(False, index)
                return fp(0)
        # Any larger integer raises OverflowError when Python multiplies it by
        # a float. This bound is imposed by execution, not a search heuristic.
        self._require(z3.And(amount >= 0, amount < 2 ** 1024), index)
        value = z3.fpToFPUnsigned(RNE, z3.Int2BV(amount, 1024), F64)
        self._require(finite(value), index)
        return value

    def _status(self, order: dict, expected: str, index: int) -> None:
        self._require(order["status"] == z3.StringVal(expected), index)

    def _symbol_matches(self, symbol: Any, concrete: str) -> Any:
        if isinstance(symbol, str):
            return symbol == concrete
        return symbol == self.symbols.index(concrete)

    def _select_symbol(self, mapping: dict, symbol: Any) -> Any:
        if isinstance(symbol, str):
            return mapping[symbol]
        result = mapping[self.symbols[-1]]
        for position in range(len(self.symbols) - 2, -1, -1):
            result = z3.If(symbol == position, mapping[self.symbols[position]], result)
        return result

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
        if self.timestamp_policy == "backend":
            start = int((CURRENT_TIME - datetime(2024, 9, 1)).total_seconds())
            self.solver.add(timestamp >= start, timestamp <= start + 86400)
        else:
            epoch = datetime(2024, 9, 1)
            self.solver.add(timestamp >= int((datetime.min - epoch).total_seconds()),
                            timestamp <= (datetime.max - epoch).days * 86400 + 86399)
        self.history_writes.append({"type": kind, "amount": amount, "timestamp": timestamp})

    def _check_transaction_history(self, args: dict, obs: dict, index: int) -> None:
        self._require(self.auth, index)
        self.history.observe(args, obs["transaction_history"], self.history_writes,
                             self._require, index, _observations_equal)

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
                z3.Not(self.watchlist.contains(args["symbol"])),
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
                self._require(self._cash_guard(args["amount"], insufficient=True), index)
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
                                self._formatted_money(self.balance, match.group(1), index)
                                self._require(self._cash_guard(cost, insufficient=True), index)
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
                    if any(type(item) is not int for item in ids):
                        self._require(False, index)
                        return True
                    self._require(error == f"Order with ID {order_id} not found.Here is the list of orders_id: {ids}", index)
                    self._require(self.order_keys.observe(ids), index)
                    self._require(z3.BoolVal(order_id not in ids), index)
            elif error == f"Order with ID {order_id} not found.":
                if name == "get_order_details":
                    self._require(False, index)
                    return True
                self._require(
                    z3.Not(self.order_keys.contains(order_id)),
                    index,
                )
            else:
                if order_id not in self.orders:
                    self._ensure_initial_order(name, order_id, index)
                self._require(
                    self.order_keys.contains(order_id),
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
                        self._require(error == f"{prefix}{status.lower()}.", index)
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
            self._require(error == f"{prefix}{status.lower()}.", index)
            self._require(status != "Open", index)
            self._status(order, status, index)
            return
        self._status(order, "Open", index)
        if error == "Market is closed. Orders cannot be executed.":
            self._require(z3.Not(self.market_open), index)
            return
        self._require(self.market_open, index)
        symbol = order["symbol"]
        exists = self._select_symbol(self.stock_exists, symbol)
        match = re.fullmatch(r"Stock with symbol '(.*)' not found\.", error)
        if match:
            self._require(self._symbol_matches(symbol, match.group(1)), index)
            self._require(z3.Not(exists), index)
            return
        self._require(exists, index)
        price = self._select_symbol(self.prices, symbol)
        limit = order["limit"]
        kind = order["order_type"]
        buy, sell = kind == "Buy", kind == "Sell"
        match = re.fullmatch(
            r"(Buy|Sell) limit price of \$(-?\d+\.\d{2}) is (below|above) the current stock price of \$(-?\d+\.\d{2})\.", error)
        if match:
            side, shown_limit, direction, shown_price = match.groups()
            self._require(kind == side, index)
            self._require(direction == ("below" if side == "Buy" else "above"), index)
            self._formatted_money(limit, shown_limit, index)
            self._formatted_money(price, shown_price, index)
            self._require(price > limit if side == "Buy" else price < limit, index)
            return
        self._require(z3.Implies(buy, price <= limit), index)
        self._require(z3.Implies(sell, price >= limit), index)
        converted_amount = self._float_amount(order["amount"], index)
        if error == "Insufficient funds to execute the buy order.":
            self._require(buy, index)
            info = self.first_stock_info.get(symbol) if isinstance(symbol, str) else None
            if info is not None and isinstance(order["amount"], int):
                cost = round(float(info["price"]) * order["amount"], 2)
                self._require(self._cash_guard(cost, insufficient=True), index)
            else:
                trade_value = rounded(price * converted_amount, 2)
                self._require(self._cash_guard(trade_value, insufficient=True), index)
        elif error == "Insufficient shares to execute the sell order.":
            self._require(sell, index)
            self._require(self._select_symbol(self.holdings, symbol) < order["amount"], index)
        elif error.startswith("Invalid order type: "):
            self._require(z3.And(z3.Not(buy), z3.Not(sell)), index)
            self._require(kind == error[len("Invalid order type: "):], index)
        else:
            self._require(False, index)

    def _ensure_initial_order(
        self, name: str, order_id: int, index: int
    ) -> dict:
        if order_id in self.orders:
            return self.orders[order_id]
        prefix = f"initial_order_{order_id}"
        # Immutable fields can be fixed by a later full read. Without one they
        # remain variables, constrained jointly by fills, cash and holdings.
        details = None
        observed_amount = None
        fill_price = None
        has_execute_error = False
        has_execution = False
        for step in self.steps:
            if not isinstance(step, dict) or step.get("args", {}).get("order_id") != order_id:
                continue
            obs = step.get("observation", {})
            if not isinstance(obs, dict):
                continue
            tool = step.get("tool", "").removeprefix("TradingBot.")
            has_execution |= tool == "execute_order"
            if "error" in obs:
                has_execute_error |= tool == "execute_order"
                continue
            if tool == "get_order_details" and {"symbol", "order_type", "price", "amount"} <= obs.keys():
                details = obs
                break
            if tool == "execute_order" and "amount" in obs:
                observed_amount = obs["amount"]
                fill_price = obs["filled_price"]
        symbol = details["symbol"] if details else z3.Int(prefix + "_symbol")
        kind = details["order_type"] if details else z3.String(prefix + "_type") if has_execution else "Buy"
        limit = (fp(details["price"]) if details else fp(_money(_units(fill_price))) if fill_price is not None and not has_execute_error
                 else z3.FP(prefix + "_limit", F64) if has_execution else fp(1))
        amount = details["amount"] if details else (
            observed_amount if observed_amount is not None else z3.Int(prefix + "_amount") if has_execution else 1
        )
        if not isinstance(symbol, str):
            self._require(z3.And(symbol >= 0, symbol < len(self.symbols)), index)
        self._require(z3.And(limit > fp(0), three_decimals(limit)), index)
        self._require(z3.Or(kind == "Buy", kind == "Sell"), index)
        self._require(amount > 0, index)
        record = {"id": order_id, "symbol": symbol, "order_type": kind,
                  "limit": limit, "amount": amount,
                  "status": z3.String(prefix + "_status"),
                  "filled_present": z3.Bool(prefix + "_filled_present"),
                  "filled_price": fp(details["filled_price"]) if details and "filled_price" in details else fp(1),
                  "extras": {k: deepcopy(v) for k, v in (details or {}).items()
                             if k not in {"id", "symbol", "order_type", "price", "amount", "status", "filled_price"}}}
        self._require(z3.And(record["filled_price"] > fp(0), three_decimals(record["filled_price"])), index)
        self._require(z3.Or(*[record["status"] == state for state in ("Pending", "Open", "Completed", "Cancelled")]), index)
        self._require(z3.Implies(record["filled_present"], record["status"] == "Completed"), index)
        self.initial_orders[order_id] = dict(record)
        self.orders[order_id] = record
        self._require(
            self.order_keys.contains(order_id, initial=True),
            index,
        )
        return record

    def _process(self, index: int, step: dict) -> None:
        if not isinstance(step, dict):
            raise UnsupportedTrace("Each step must be an object")
        name = _tool_name(step.get("tool", ""))
        if self._ignored(step):
            return
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
            "add_to_watchlist": {"status"},
            "remove_stock_from_watchlist": {"status"},
            "filter_stocks_by_price": {"filtered_stocks"},
            "notify_price_change": {"notification"},
            "get_order_history": {"history"},
            "get_transaction_history": {"transaction_history"},
            "trading_get_login_status": {"status"},
            "trading_login": {"status"},
            "trading_logout": {"status"},
            "fund_account": {"status"},
            "withdraw_funds": {"status"},
            "place_order": {"order_id", "order_type", "status", "price", "amount"},
            "activate_order": {"order_id", "status"},
            "cancel_order": {"order_id", "status"},
            "execute_order": {"order_id", "status", "filled_price", "amount"},
        }
        if name in exact_fields:
            self._require(z3.BoolVal(set(obs) == exact_fields[name]), index)

        if name == "get_account_info":
            self._require(self.auth, index)
            _units(obs["balance"])
            self._require(number_matches(self.balance, obs["balance"]), index)
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
                if not isinstance(count, int) or isinstance(count, bool):
                    raise UnsupportedTrace("Holdings must be nonnegative integer share counts")
                self._require(count >= 0, index)
                self._require(self.holdings[symbol] == count, index)
                # Missing and explicit-zero entries both observe zero shares.
                # Backend key presence still evolves normally for the witness;
                # it is no longer constrained by this representation choice.
        elif name == "get_market_status":
            status = obs["market_status"]
            if status not in {"Open", "Closed"}:
                self._require(z3.BoolVal(False), index)
            else:
                self._require(self.market_open == (status == "Open"), index)
        elif name == "get_stock_info":
            symbol = args["symbol"]
            self._require(self.stock_exists[symbol], index)
            _units(obs["price"])
            self._require(number_matches(self.prices[symbol], obs["price"]), index)
            self._require(
                number_matches(self.percent_changes[symbol], obs["percent_change"]),
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
            self._require(self.watchlist.observe(watchlist), index)
        elif name == "add_to_watchlist":
            symbol = args["stock"]
            member = self.watchlist.contains(symbol)
            if obs["status"] == f"Stock {symbol} added to watchlist successfully.":
                self._require(z3.And(z3.Not(member), self.stock_exists[symbol]), index)
                self.watchlist.append(symbol)
            elif obs["status"] == f"Stock {symbol} is already in the watchlist.":
                self._require(member, index)
            elif obs["status"] == f"Stock {symbol} is not a valid stock symbol.":
                self._require(z3.And(z3.Not(member), z3.Not(self.stock_exists[symbol])), index)
            else:
                self._require(False, index)
        elif name == "remove_stock_from_watchlist":
            symbol = args["symbol"]
            self._require(self.auth, index)
            self._require(self.watchlist.contains(symbol), index)
            self.watchlist.remove_first(symbol)
            self._same_static({"status": f"Stock {symbol} removed from watchlist successfully."}, obs, index)
        elif name == "filter_stocks_by_price":
            inputs = args["stocks"]
            outputs = obs["filtered_stocks"]
            low = fp(args["min_price"])
            high = fp(args["max_price"])
            predicates = []
            for symbol in inputs:
                price = z3.If(
                    self.stock_exists[symbol], self.prices[symbol], fp(0),
                )
                predicates.append(z3.And(price >= low, price <= high))
            self._filter_list(inputs, outputs, predicates, index)
        elif name == "notify_price_change":
            threshold = fp(args["threshold"])
            joined, any_match = z3.StringVal(""), z3.BoolVal(False)
            for symbol in args["stocks"]:
                selected = z3.And(self.stock_exists[symbol], z3.fpAbs(self.percent_changes[symbol]) >= threshold)
                joined = z3.If(selected, z3.Concat(joined, z3.If(any_match, z3.StringVal(", "), z3.StringVal("")), z3.StringVal(symbol)), joined)
                any_match = z3.Or(any_match, selected)
            expected = z3.If(any_match, z3.Concat(z3.StringVal("Stocks "), joined, z3.StringVal(" have significant price changes.")),
                             z3.StringVal("No significant price changes in the selected stocks."))
            self._require(expected == obs["notification"], index)
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
                self.order_keys.observe(history),
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
            _units(args["amount"])
            amount = fp(args["amount"])
            self._require(self.auth, index)
            self._require(amount > fp(0), index)
            self.balance = self.balance + amount
            self._same_static("Account funded successfully", obs["status"], index)
            self._record_history_write("deposit", args["amount"])
        elif name == "withdraw_funds":
            _units(args["amount"])
            amount = fp(args["amount"])
            self._require(self.auth, index)
            self._require(self.market_open, index)
            self._require(amount > fp(0), index)
            self._require(self._cash_guard(args["amount"], insufficient=False), index)
            self.balance = self.balance - amount
            self._same_static("Withdrawal successful", obs["status"], index)
            self._record_history_write("withdrawal", args["amount"])
        elif name == "place_order":
            symbol = args["symbol"]
            order_type = args["order_type"].capitalize()
            amount = args["amount"]
            _units(args["price"])
            limit = fp(args["price"])
            order_id = obs["order_id"]
            if not isinstance(order_id, int) or isinstance(order_id, bool):
                raise UnsupportedTrace("Placed order IDs must be integers")
            self._require(self.order_keys.place(order_id), index)
            self._require(self.auth, index)
            self._require(self.stock_exists[symbol], index)
            self._require(z3.BoolVal(order_type in {"Buy", "Sell"}), index)
            self._require(z3.And(amount > 0, limit > fp(0)), index)
            if order_type == "Buy":
                self._require(self._cash_guard(float(args["price"]) * int(amount), insufficient=False), index)
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
                "status": z3.StringVal("Pending"), "filled_price": None, "filled_present": False,
            }
            self.placed_ids.append(order_id)
        elif name == "get_order_details":
            order_id = args["order_id"]
            order = self._ensure_initial_order(name, order_id, index)
            self._same_static(order.get("extras", {}), {k: v for k, v in obs.items()
                             if k not in {"id", "symbol", "order_type", "price", "amount", "status", "filled_price"}}, index)
            self._status(order, obs["status"], index)
            self._same_static(order_id, obs["id"], index)
            self._require(order["order_type"] == obs["order_type"], index)
            self._require(self._symbol_matches(order["symbol"], obs["symbol"]), index)
            self._require(number_matches(order["limit"], obs["price"]), index)
            self._require(order["amount"] == obs["amount"], index)
            self._require(order["filled_present"] == ("filled_price" in obs), index)
            if "filled_price" in obs and order["filled_price"] is not None:
                self._require(number_matches(order["filled_price"], obs["filled_price"]), index)
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
                self._require(self._select_symbol(self.stock_exists, symbol), index)
                price = self._select_symbol(self.prices, symbol)
                amount = obs["amount"]
                if not isinstance(amount, int) or isinstance(amount, bool):
                    raise UnsupportedTrace("Executed share counts must be positive integers")
                if amount <= 0:
                    self._require(False, index)
                    return
                self._float_amount(amount, index)
                self._require(order["amount"] == amount, index)
                _units(obs["filled_price"])
                trade_fp = fp(round(_money(_units(obs["filled_price"])) * amount, 2))
                kind = order["order_type"]
                buy, sell = kind == "Buy", kind == "Sell"
                self._require(z3.Or(buy, sell), index)
                self._require(z3.Implies(buy, price <= order["limit"]), index)
                self._require(z3.Implies(sell, price >= order["limit"]), index)
                self._require(z3.Implies(buy, self._cash_guard(trade_fp, insufficient=False)), index)
                self._require(z3.Implies(sell, self._select_symbol(self.holdings, symbol) >= amount), index)
                self.balance = rounded(self.balance + z3.If(buy, -trade_fp, trade_fp), 2)
                for stock in self.symbols:
                    selected = self._symbol_matches(symbol, stock)
                    old = self.holdings[stock]
                    self.holdings[stock] = z3.simplify(z3.If(selected, old + z3.If(buy, amount, -amount), old))
                    self.holding_present[stock] = z3.simplify(z3.If(
                        selected, z3.Or(buy, old - amount != 0), self.holding_present[stock]))
                order["status"] = z3.StringVal("Completed")
                order["filled_price"] = price
                order["filled_present"] = True
                self._same_static(order_id, obs["order_id"], index)
                self._same_static("Completed", obs["status"], index)
                self._require(number_matches(price, obs["filled_price"]), index)

    def _witness(self, model: z3.ModelRef) -> tuple[dict, dict]:
        def value(expr: Any) -> int:
            return expr if isinstance(expr, int) else model.eval(expr, model_completion=True).as_long()

        balance = as_float(self.initial_balance, model)
        holdings = {
            symbol: value(expr)
            for symbol, expr in self.initial_holdings.items()
            if z3.is_true(model.eval(self.initial_holding_present[symbol], model_completion=True))
        }
        stock_prices = {symbol: as_float(expr, model) for symbol, expr in self.prices.items()}
        authenticated = z3.is_true(model.eval(self.initial_auth, model_completion=True))
        market_open = z3.is_true(model.eval(self.initial_market_open, model_completion=True))
        initial_order_ids = self.order_keys.witness(model)
        order_counter = self.placed_ids[0] if self.placed_ids else max(initial_order_ids, default=0) + 1

        scenario = deepcopy(DEFAULT_STATE)
        def text_value(expr: Any) -> str:
            return expr if isinstance(expr, str) else model.eval(expr, model_completion=True).as_string()

        scenario["orders"] = {}
        for order_id in initial_order_ids:
            source = self.initial_orders.get(order_id)
            if source is None:
                scenario["orders"][order_id] = {
                    "id": order_id, "order_type": "Buy", "symbol": "AAPL",
                    "price": 1.0, "amount": 1, "status": "Pending",
                }
                continue
            symbol = source["symbol"]
            record = {"id": order_id, "order_type": text_value(source["order_type"]),
                      "symbol": symbol if isinstance(symbol, str) else self.symbols[value(symbol)],
                      "price": as_float(source["limit"], model), "amount": value(source["amount"]),
                      "status": text_value(source["status"])}
            if z3.is_true(model.eval(source["filled_present"], model_completion=True)):
                record["filled_price"] = as_float(source["filled_price"], model)
            record.update(deepcopy(source.get("extras", {})))
            scenario["orders"][order_id] = record
        scenario["order_counter"] = order_counter
        scenario["authenticated"] = authenticated
        scenario["market_status"] = "Open" if market_open else "Closed"
        scenario["holdings"] = holdings
        scenario["account_info"]["balance"] = balance
        if self.first_account_info is not None:
            scenario["account_info"] = {"balance": balance, **self.first_account_info}
        scenario["watch_list"] = self.watchlist.witness(model)
        scenario["transaction_history"] = self.history.witness(model)
        generated_timestamps = []
        for write in self.history_writes:
            moment = datetime(2024, 9, 1) + timedelta(seconds=value(write["timestamp"]))
            # strftime('%Y') does not zero-pad years below 1000 on every OS.
            generated_timestamps.append(
                f"{moment.year:04d}-{moment.month:02d}-{moment.day:02d} "
                f"{moment.hour:02d}:{moment.minute:02d}:{moment.second:02d}")
        scenario["stocks"] = {}
        for symbol, price in stock_prices.items():
            if not z3.is_true(model.eval(self.stock_exists[symbol], model_completion=True)):
                continue
            info = deepcopy(DEFAULT_STATE["stocks"].get(symbol, {
                "volume": 0.0, "MA(5)": price, "MA(20)": price,
            }))
            info["price"] = price
            info["percent_change"] = as_float(self.percent_changes[symbol], model)
            if symbol in self.first_stock_info:
                info = deepcopy(self.first_stock_info[symbol])
                info["price"] = price
                info["percent_change"] = as_float(self.percent_changes[symbol], model)
            scenario["stocks"][symbol] = info
        summary = {
            "balance": balance, "holdings": holdings,
            "authenticated": authenticated, "market_status": scenario["market_status"],
            "stock_prices": stock_prices, "initial_orders": deepcopy(scenario["orders"]),
            "order_counter": order_counter,
            "watchlist": scenario["watch_list"],
            "generated_transaction_timestamps": generated_timestamps,
        }
        return scenario, summary

    def _check_formula(self):
        """Try cheap witnesses first; a failed guess never proves UNSAT."""
        leaves = {}
        amounts = {}
        pending, visited = list(self.solver.assertions()), set()
        while pending:
            expression = pending.pop()
            if expression.get_id() in visited:
                continue
            visited.add(expression.get_id())
            if z3.is_const(expression) and expression.decl().kind() == z3.Z3_OP_UNINTERPRETED and isinstance(expression, z3.FPRef):
                leaves[str(expression)] = expression
            if z3.is_const(expression) and expression.decl().kind() == z3.Z3_OP_UNINTERPRETED and str(expression).startswith("initial_order_") and str(expression).endswith("_amount"):
                amounts[str(expression)] = expression
            pending.extend(expression.children())
        if leaves or amounts:
            balances = [0.0, 1.0, 1000000.0]
            for step in self.steps:
                obs = step.get("observation", {})
                if isinstance(obs, dict) and isinstance(obs.get("error"), str):
                    match = re.search(r"but only \$(-?\d+\.\d{2}) available", obs["error"])
                    if match:
                        balances.insert(0, float(match.group(1)))
            candidates = {0.0}
            for step in self.steps:
                name = step.get("tool", "").removeprefix("TradingBot.")
                args, obs = step.get("args", {}), step.get("observation", {})
                if not isinstance(obs, dict) or "error" in obs:
                    continue
                if name == "get_account_info":
                    balances = [obs["balance"] - delta for delta in candidates] + balances
                    break
                if name in {"fund_account", "withdraw_funds"}:
                    delta = args["amount"] * (1 if name == "fund_account" else -1)
                    candidates = {v + delta for v in candidates}
                elif name == "execute_order":
                    delta = round(float(obs["filled_price"]) * obs["amount"], 2)
                    candidates = {v + direction * delta for v in candidates for direction in (-1, 1)}
                    # This caps only speculative witnesses, never the formula.
                    candidates = set(sorted(candidates)[:32])
            labels = [(z3.Bool(label), z3.BoolVal(True)) for label in self.labels]
            amount_hint = 1 + sum(
                sum(obs.get("holdings", {}).values()) + abs(obs.get("amount", 0))
                for step in self.steps if isinstance(obs := step.get("observation"), dict)
            )
            for balance in dict.fromkeys(balances):
                if balance < 0 or not math.isfinite(balance):
                    continue
                replacements = [(leaf, fp(round(balance, 3) if name == "initial_balance" else 1))
                                for name, leaf in leaves.items()]
                replacements.extend((leaf, z3.IntVal(max(1, amount_hint))) for leaf in amounts.values())
                trial = z3.Solver()
                trial.set(timeout=500)
                for expression in self.solver.assertions():
                    trial.add(z3.simplify(z3.substitute(expression, *(labels + replacements))))
                trial.add(*[leaf == value for leaf, value in replacements])
                if trial.check() == z3.sat:
                    return z3.sat, trial.model()
        result = self.solver.check()
        return result, self.solver.model() if result == z3.sat else None

    def solve(self) -> dict:
        # An empty trace (or only ignored calls) is a valid unconstrained case.
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
                return {"status": "invalid_format", "step": index + 1, "reason": str(exc)}
        result, model = self._check_formula()
        if result == z3.unsat:
            conflicting = sorted({self.labels[str(x)] for x in self.solver.unsat_core() if str(x) in self.labels})
            return {"status": "unsat", "first_conflicting_step": max(conflicting, default=len(self.steps)), "conflicting_steps": conflicting}
        if result == z3.unknown:
            return {"status": "unknown", "reason": self.solver.reason_unknown()}

        scenario, summary = self._witness(model)
        backend = TradingBot()
        backend._load_scenario(scenario)
        timestamps = iter(summary["generated_transaction_timestamps"])
        backend._generate_transaction_timestamp = lambda: next(timestamps)
        for index, step in enumerate(self.steps):
            name = _tool_name(step["tool"])
            if self._ignored(step):
                continue
            try:
                actual = getattr(backend, name)(**step.get("args", {}))
            except Exception as exc:
                return {
                    "status": "unknown", "step": index + 1,
                    "reason": f"SMT witness raised {type(exc).__name__} during backend replay: {exc}",
                }
            if not _replay_observations_equal(name, actual, step["observation"]):
                return {
                    "status": "unknown", "step": index + 1,
                    "reason": "SMT witness did not replay under numeric and zero-holding equivalence policies",
                    "backend_observation": actual,
                }
        return {"status": "sat", "initial_state": summary,
                "witness_state": scenario,
                "random_choices": {"transaction_timestamps": summary["generated_transaction_timestamps"]},
                "replay_verified": True}


def check_trace(steps: list[dict], *, check_static_tools: bool = False,
                timestamp_policy: str = "symbolic") -> dict:
    """Check consistency, excluding fixed lookup tools by default.

    symbolic timestamps may be any valid datetime, but remain identical on
    repeated reads and obey date filters. backend restricts them to the
    backend's hardcoded generation window. Neither mode predicts an RNG seed.
    get_holdings treats missing keys and explicit integer-zero counts as
    equivalent in both constraints and witness replay; nonzero counts must match.
    """
    try:
        validate(steps, TradingBot, SUPPORTED_TOOLS, set() if check_static_tools else STATIC_TOOLS, _units)
        return TradingConsistencySolver(steps, check_static_tools=check_static_tools,
                                        timestamp_policy=timestamp_policy).solve()
    except (UnsupportedTrace, ValueError, TypeError, KeyError, OverflowError) as exc:
        return {"status": "invalid_format", "reason": str(exc)}



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="JSON file containing a steps array")
    parser.add_argument("--check-static-tools", action=argparse.BooleanOptionalAction, default=False,
                        help="Check fixed time/name/stock-list outputs (default: ignore these three tools)")
    parser.add_argument("--timestamp-policy", choices=("symbolic", "backend"), default="symbolic",
                        help="Use free consistent timestamps, or enforce the backend's fixed one-day window")
    args = parser.parse_args()
    data = json.loads(args.trace.read_text(encoding="utf-8-sig"))
    print(json.dumps(check_trace(data["steps"], check_static_tools=args.check_static_tools,
                                 timestamp_policy=args.timestamp_policy), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
