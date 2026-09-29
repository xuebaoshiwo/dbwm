"""Seeded mixed tool traces replayed from the real TradingBot backend."""

import random
import unittest
from copy import deepcopy

from bfcl_eval.consistency.trading_solver import check_trace
from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.trading_bot import TradingBot


class TradingMixedTraceTests(unittest.TestCase):
    def test_eighty_valid_mixed_traces(self):
        rng = random.Random(42)
        names = [
            "get_account_info", "get_holdings", "get_market_status", "get_stock_info",
            "get_watchlist", "get_current_time", "get_symbol_by_name",
            "get_available_stocks", "trading_get_login_status", "trading_login",
            "trading_logout", "fund_account", "withdraw_funds", "place_order",
            "get_order_details", "activate_order", "execute_order", "cancel_order",
            "add_to_watchlist", "remove_stock_from_watchlist", "filter_stocks_by_price",
            "notify_price_change", "get_order_history", "get_transaction_history",
        ]

        def args(name, bot):
            if name == "get_stock_info":
                return {"symbol": rng.choice(["AAPL", "BAD", "GOOG"])}
            if name == "get_symbol_by_name":
                return {"name": rng.choice(["Apple", "Google", "BAD"])}
            if name == "get_available_stocks":
                return {"sector": rng.choice(["Technology", "Automobile", "BAD"])}
            if name == "trading_login":
                return {"username": "u", "password": "p"}
            if name in {"fund_account", "withdraw_funds"}:
                return {"amount": rng.choice([1, 10, 10000, 0])}
            if name == "place_order":
                return {"order_type": rng.choice(["Buy", "Sell"]),
                        "symbol": rng.choice(["AAPL", "GOOG", "BAD"]),
                        "price": rng.choice([1, 230, 3000]), "amount": rng.choice([1, 2])}
            if name in {"get_order_details", "activate_order", "execute_order", "cancel_order"}:
                return {"order_id": rng.choice(list(bot.orders) + [999])}
            if name == "add_to_watchlist":
                return {"stock": rng.choice(["AAPL", "BAD", "NVDA"])}
            if name == "remove_stock_from_watchlist":
                return {"symbol": rng.choice(["AAPL", "BAD", "NVDA"])}
            if name == "filter_stocks_by_price":
                return {"stocks": ["AAPL", "GOOG", "BAD"],
                        "min_price": rng.choice([0, 100, 1000]),
                        "max_price": rng.choice([100, 300, 3000])}
            if name == "notify_price_change":
                return {"stocks": ["AAPL", "GOOG", "BAD"],
                        "threshold": rng.choice([0.1, 0.2, 0.3])}
            return {}

        for run in range(80):
            bot = TradingBot()
            bot._load_scenario({"authenticated": bool(rng.randrange(2)),
                                "market_status": rng.choice(["Open", "Closed"])})
            steps = []
            for _ in range(12):
                name = rng.choice(names)
                arguments = args(name, bot)
                steps.append({"tool": name, "args": arguments,
                              "observation": deepcopy(getattr(bot, name)(**arguments))})
            result = check_trace(steps)
            self.assertEqual(result["status"], "sat", f"Run {run}: {result}; steps={steps}")


if __name__ == "__main__":
    unittest.main()
