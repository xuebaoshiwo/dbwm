"""TradingBot adapter. Add another adapter for a new backend/domain."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from bfcl_eval.consistency.data_generator.core import Call, Job
from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.trading_bot import TradingBot


REPO_ROOT = Path(__file__).resolve().parents[3]
STATE_POOL = REPO_ROOT / "bfcl_eval/consistency/initial_states_pools/trading_bot/trading_states.jsonl"
SCHEMA_PATH = REPO_ROOT / "bfcl_eval/data/multi_turn_func_doc/trading_bot.json"


def load_initial_states(path: Path = STATE_POOL) -> list[dict[str, Any]]:
    """Read concatenated JSON objects, including pretty-printed 'JSONL' files."""
    content = path.read_text(encoding="utf-8-sig")
    decoder = json.JSONDecoder()
    position = 0
    states: list[dict[str, Any]] = []
    while position < len(content):
        while position < len(content) and content[position].isspace():
            position += 1
        if position == len(content):
            break
        state, position = decoder.raw_decode(content, position)
        if not isinstance(state, dict):
            raise ValueError(f"Initial state #{len(states) + 1} is not an object")
        states.append(state)
    if not states:
        raise ValueError("No initial states found")
    return states


class TradingAdapter:
    name = "trading_bot"

    def target_read_call(self, job: Job) -> Call:
        if job.target == "account_info.balance":
            return Call("get_account_info", {})
        if job.target == "watch_list":
            return Call("get_watchlist", {})
        raise ValueError(f"Unsupported TradingBot target: {job.target}")

    def make_skeleton(self, job: Job) -> list[Call | None]:
        if job.length < 4 or not 1 <= job.write_count <= job.read_gap:
            raise ValueError("Need room for login, two reads, and effective writes")
        if not 1 <= job.read_gap <= job.length - 3:
            raise ValueError("read_gap must fit between login and the last call")
        read = self.target_read_call(job)
        if job.target == "account_info.balance":
            writes = self._balance_writes(job)
        elif job.target == "watch_list":
            writes = self._watchlist_writes(job)
        else:
            raise ValueError(f"Unsupported TradingBot target: {job.target}")

        skeleton: list[Call | None] = [Call("trading_login", {"username": "alice", "password": "demo"}), read]
        # Spread writes through the intervening window. None slots are for
        # model-selected read-only calls. No extra target reads are inserted.
        write_slots = [((index + 1) * job.read_gap) // (job.write_count + 1) for index in range(job.write_count)]
        if len(set(write_slots)) != len(write_slots):
            raise ValueError("Too many writes for the requested read gap")
        for slot in range(job.read_gap):
            skeleton.append(writes[write_slots.index(slot)] if slot in write_slots else None)
        skeleton.append(read)
        skeleton.extend([None] * (job.length - len(skeleton)))
        return skeleton

    def _balance_writes(self, job: Job) -> list[Call]:
        balance = float(job.initial_state["account_info"]["balance"])
        calls: list[Call] = []
        for index in range(job.write_count):
            if index % 2 == 0 or balance < 25.0 or job.initial_state["market_status"] != "Open":
                amount = float(50 + 25 * (index // 2))
                calls.append(Call("fund_account", {"amount": amount}))
                balance += amount
            else:
                amount = float(min(25.0, balance / 2))
                calls.append(Call("withdraw_funds", {"amount": amount}))
                balance -= amount
        return calls

    def _watchlist_writes(self, job: Job) -> list[Call]:
        state = job.initial_state
        symbols = list(state["stocks"])
        if not symbols:
            raise ValueError("Watchlist target requires at least one stock")
        symbol = symbols[0]
        present = symbol in state["watch_list"]
        calls: list[Call] = []
        for _ in range(job.write_count):
            if present:
                calls.append(Call("remove_stock_from_watchlist", {"symbol": symbol}))
            else:
                calls.append(Call("add_to_watchlist", {"stock": symbol}))
            present = not present
        return calls

    def distractor_candidates(self, job: Job) -> list[Call]:
        state = job.initial_state
        symbols = list(state["stocks"])
        candidates = [
            Call("get_current_time", {}),
            Call("get_market_status", {}),
            Call("trading_get_login_status", {}),
            Call("get_holdings", {}),
            Call("get_order_history", {}),
            Call("get_transaction_history", {}),
            Call("get_available_stocks", {"sector": "Technology"}),
            Call("get_symbol_by_name", {"name": "Apple Inc."}),
        ]
        for symbol in symbols[:2]:
            candidates.append(Call("get_stock_info", {"symbol": symbol}))
        if symbols:
            candidates.extend(
                [
                    Call("filter_stocks_by_price", {"stocks": symbols[:3], "min_price": 0.0, "max_price": 5000.0}),
                    Call("notify_price_change", {"stocks": symbols[:3], "threshold": 1.0}),
                ]
            )
        if job.target == "watch_list":
            candidates.append(Call("get_account_info", {}))
        else:
            candidates.append(Call("get_watchlist", {}))
        return candidates

    def rollout(self, initial_state: Mapping[str, Any], calls: list[Call]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        bot = TradingBot()
        bot._load_scenario(dict(initial_state))
        observations: list[dict[str, Any]] = []
        states = [self._snapshot(bot)]
        for call in calls:
            observation = getattr(bot, call.tool)(**deepcopy(call.args))
            if not isinstance(observation, dict):
                raise ValueError(f"{call.tool} returned a non-object observation")
            observations.append(deepcopy(observation))
            states.append(self._snapshot(bot))
        return observations, states

    def _snapshot(self, bot: TradingBot) -> dict[str, Any]:
        return deepcopy(
            {
                "account_info": bot.account_info,
                "watch_list": bot.watch_list,
                "orders": bot.orders,
                "holdings": bot.holdings,
                "authenticated": bot.authenticated,
                "market_status": bot.market_status,
                "transaction_history": bot.transaction_history,
            }
        )

    def target_value(self, state: Mapping[str, Any], target: str) -> Any:
        if target == "account_info.balance":
            return state["account_info"]["balance"]
        if target == "watch_list":
            return state["watch_list"]
        raise ValueError(target)

    def describe_call(self, call: Call) -> str:
        tool, args = call.tool, call.args
        descriptions = {
            "trading_login": f"使用用户名 {args.get('username')} 和密码 {args.get('password')} 登录股票账户",
            "get_account_info": "查询账户信息和现金余额",
            "get_watchlist": "查询自选股列表",
            "get_holdings": "查询当前持仓",
            "get_market_status": "查询市场开闭状态",
            "get_current_time": "查询当前时间",
            "get_order_history": "查询订单历史",
            "get_transaction_history": "查询资金流水",
            "trading_get_login_status": "查询登录状态",
            "get_available_stocks": f"查询 {args.get('sector')} 板块的股票",
            "get_symbol_by_name": f"查询 {args.get('name')} 对应的股票代码",
            "get_stock_info": f"查询 {args.get('symbol')} 的股票信息",
            "filter_stocks_by_price": f"在 {args.get('stocks')} 中筛选价格介于 {args.get('min_price')} 和 {args.get('max_price')} 的股票",
            "notify_price_change": f"查看 {args.get('stocks')} 中涨跌幅达到 {args.get('threshold')} 的股票提醒",
            "fund_account": f"向账户充值 {args.get('amount')} 元",
            "withdraw_funds": f"从账户提现 {args.get('amount')} 元",
            "add_to_watchlist": f"把 {args.get('stock')} 加入自选股",
            "remove_stock_from_watchlist": f"把 {args.get('symbol')} 从自选股移除",
        }
        return descriptions[tool]

    def tool_schema(self) -> list[dict[str, Any]]:
        schemas = [
            json.loads(line)
            for line in SCHEMA_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for schema in schemas:
            if schema["name"] == "get_order_history":
                # The upstream response schema says order_history, while the
                # backend returns history. Keep generated policy schemas aligned
                # with the observations used by this benchmark.
                properties = schema["response"]["properties"]
                if "order_history" in properties:
                    properties["history"] = properties.pop("order_history")
        return schemas

    def validate_trace(self, steps: list[dict[str, Any]]) -> dict[str, Any]:
        from bfcl_eval.consistency.trading_solver import check_trace

        return check_trace(steps)
