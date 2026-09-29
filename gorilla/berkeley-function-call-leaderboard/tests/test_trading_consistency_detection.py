"""Deterministic mutation corpus for the TradingBot consistency checker."""

import unittest
from copy import deepcopy

from bfcl_eval.consistency.trading_solver import check_trace
from bfcl_eval.consistency.evaluate import evaluate_corpus
from tests.test_trading_consistency_solver import recorded_trace


class TradingDetectionCorpusTests(unittest.TestCase):
    def test_labeled_corpus_reports_detection_and_false_positives(self):
        valid = recorded_trace({"authenticated": True},
                               [("get_account_info", {}), ("get_account_info", {})])
        invalid = deepcopy(valid)
        invalid[-1]["observation"]["balance"] += 1
        report = evaluate_corpus([
            {"steps": valid, "inconsistent": False},
            {"steps": invalid, "inconsistent": True},
        ])["metrics"]
        self.assertEqual(report["detection_rate"], 1.0)
        self.assertEqual(report["false_positive_rate"], 0.0)

    def test_one_hundred_known_inconsistencies(self):
        cases = []
        for offset in range(1, 11):
            account = recorded_trace(
                {"authenticated": True, "market_status": "Open"},
                [("get_account_info", {}), ("fund_account", {"amount": 10}), ("get_account_info", {})],
            )
            account[-1]["observation"]["balance"] += offset
            cases.append(("balance", account))

            holdings = recorded_trace(
                {"authenticated": True, "market_status": "Open", "orders": {}, "order_counter": 1},
                [("get_holdings", {}),
                 ("place_order", {"order_type": "Buy", "symbol": "AAPL", "price": 230, "amount": 1}),
                 ("activate_order", {"order_id": 1}), ("execute_order", {"order_id": 1}),
                 ("get_holdings", {})],
            )
            holdings[-1]["observation"]["holdings"]["AAPL"] += offset
            cases.append(("holdings", holdings))

            market = recorded_trace({}, [("get_market_status", {}), ("get_market_status", {})])
            market[-1]["observation"]["market_status"] = "Open"
            cases.append(("market", market))

            stock = recorded_trace({}, [("get_stock_info", {"symbol": "AAPL"}),
                                        ("get_stock_info", {"symbol": "AAPL"})])
            stock[-1]["observation"]["price"] += offset / 100
            cases.append(("stock", stock))

            watchlist = recorded_trace(
                {"authenticated": True},
                [("get_watchlist", {}), ("add_to_watchlist", {"stock": "AAPL"}), ("get_watchlist", {})],
            )
            watchlist[-1]["observation"]["watchlist"].append(f"BAD{offset}")
            cases.append(("watchlist", watchlist))

            order = recorded_trace(
                {"authenticated": True},
                [("get_order_details", {"order_id": 12446}),
                 ("activate_order", {"order_id": 12446}),
                 ("get_order_details", {"order_id": 12446})],
            )
            order[-1]["observation"]["status"] = "Pending"
            cases.append(("order status", order))

            order_history = recorded_trace(
                {"authenticated": True},
                [("get_order_history", {}),
                 ("place_order", {"order_type": "Buy", "symbol": "AAPL", "price": 230, "amount": 1}),
                 ("get_order_history", {})],
            )
            order_history[-1]["observation"]["history"].append(20000 + offset)
            cases.append(("order history", order_history))

            transactions = recorded_trace(
                {"authenticated": True},
                [("fund_account", {"amount": 10}),
                 ("get_transaction_history", {}),
                 ("get_transaction_history", {})],
            )
            transactions[-1]["observation"]["transaction_history"][0]["amount"] += offset
            cases.append(("transaction history", transactions))

            filtered = recorded_trace(
                {},
                [("get_stock_info", {"symbol": "AAPL"}),
                 ("filter_stocks_by_price", {"stocks": ["AAPL"], "min_price": offset, "max_price": 300})],
            )
            filtered[-1]["observation"]["filtered_stocks"] = []
            cases.append(("price filter", filtered))

            notification = recorded_trace(
                {},
                [("get_stock_info", {"symbol": "AAPL"}),
                 ("notify_price_change", {"stocks": ["AAPL"], "threshold": offset / 100})],
            )
            notification[-1]["observation"]["notification"] = "No significant price changes in the selected stocks."
            cases.append(("price notification", notification))

        self.assertEqual(len(cases), 100)
        detected = 0
        failures = []
        for category, steps in cases:
            result = check_trace(deepcopy(steps))
            if result["status"] == "unsat":
                detected += 1
            else:
                failures.append((category, result))
        self.assertEqual(detected, 100, f"Detected {detected}/100; misses: {failures}")


if __name__ == "__main__":
    unittest.main()
