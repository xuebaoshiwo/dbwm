"""Validate the public trace format before constructing constraints."""

import inspect
import math
from datetime import datetime


def validate(steps, backend, supported, ignored, money):
    if not isinstance(steps, list):
        raise ValueError("steps must be an array")

    def number(value):
        return type(value) in (int, float) and math.isfinite(value)

    def json_value(value):
        if value is None or type(value) in (str, bool, int):
            return
        if isinstance(value, float) and math.isfinite(value):
            return
        if isinstance(value, list):
            for child in value:
                json_value(child)
            return
        if isinstance(value, dict) and all(isinstance(k, str) for k in value):
            for child in value.values():
                json_value(child)
            return
        raise ValueError("Values must be finite JSON data")

    integer_args = {"amount", "order_id"}
    numeric_args = {"price", "min_price", "max_price", "threshold"}
    for i, step in enumerate(steps, 1):
        if not isinstance(step, dict) or not isinstance(step.get("tool"), str):
            raise ValueError(f"Step {i} must have a string tool name")
        name = step["tool"].removeprefix("TradingBot.")
        if name not in supported:
            raise ValueError(f"Step {i}: unknown tool {name}")
        if name in ignored:
            continue
        args, obs = step.get("args", {}), step.get("observation")
        if not isinstance(args, dict) or not isinstance(obs, (dict, list)):
            raise ValueError(f"Step {i}: args must be an object; observation an object or error array")
        json_value(args)
        json_value(obs)
        inspect.signature(getattr(backend, name)).bind(None, **args)
        for field, value in args.items():
            if field == "stocks":
                valid = isinstance(value, list) and all(isinstance(v, str) for v in value)
            elif field == "amount" and name in {"fund_account", "withdraw_funds"}:
                valid = number(value)
                if valid:
                    money(value)
            elif field in integer_args:
                valid = type(value) is int
            elif field in numeric_args:
                valid = number(value)
                if valid:
                    money(value)
            elif field in {"start_date", "end_date"}:
                valid = value is None or isinstance(value, str)
                if valid and value:
                    datetime.strptime(value, "%Y-%m-%d")
            else:
                valid = isinstance(value, str)
            if not valid:
                raise ValueError(f"Step {i}: invalid type for argument {field}")
        if not isinstance(obs, dict) or "error" in obs:
            continue
        if name == "trading_get_login_status" and "status" in obs and type(obs["status"]) is not bool:
            raise ValueError(f"Step {i}: login status must be a boolean")
        if "status" in obs and name != "trading_get_login_status" and not isinstance(obs["status"], str):
            raise ValueError(f"Step {i}: status must be a string")
        for field in ("market_status", "order_type", "symbol", "notification"):
            if field in obs and not isinstance(obs[field], str):
                raise ValueError(f"Step {i}: {field} must be a string")
        for field in ("history", "watchlist", "filtered_stocks", "transaction_history"):
            if field in obs and not isinstance(obs[field], list):
                raise ValueError(f"Step {i}: {field} must be an array")
        for field in ("watchlist", "filtered_stocks"):
            if field in obs and any(not isinstance(v, str) for v in obs[field]):
                raise ValueError(f"Step {i}: {field} must contain strings")
        if "history" in obs and any(type(v) is not int for v in obs["history"]):
            raise ValueError(f"Step {i}: history must contain integer IDs")
        if name == "get_holdings" and "holdings" in obs:
            if not isinstance(obs["holdings"], dict) or any(type(v) is not int for v in obs["holdings"].values()):
                raise ValueError(f"Step {i}: holdings must map symbols to integers")
        for field in ("balance", "price", "filled_price", "percent_change"):
            if field in obs:
                money(obs[field])
        for field in ("id", "order_id", "amount"):
            if field in obs and type(obs[field]) is not int:
                raise ValueError(f"Step {i}: {field} must be an integer")
        for record in obs.get("transaction_history", []):
            if not isinstance(record, dict) or not {"type", "amount", "timestamp"} <= record.keys():
                raise ValueError(f"Step {i}: invalid transaction record")
            money(record["amount"])
            datetime.strptime(record["timestamp"], "%Y-%m-%d %H:%M:%S")
