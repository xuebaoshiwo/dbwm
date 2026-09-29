import unittest
from copy import deepcopy
import json
from pathlib import Path

from bfcl_eval.consistency.trading_solver import check_trace
from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.trading_bot import TradingBot


def recorded_trace(initial_state, calls):
    bot = TradingBot()
    bot._load_scenario(initial_state)
    steps = []
    for tool, args in calls:
        steps.append(
            {
                "tool": tool,
                "args": args,
                "observation": deepcopy(getattr(bot, tool)(**args)),
            }
        )
    return steps


class TradingConsistencySolverTests(unittest.TestCase):
    def test_finds_free_initial_balance_through_reads_and_writes(self):
        steps = recorded_trace(
            {"authenticated": True, "market_status": "Open", "account_info": {"account_id": 7, "balance": 700.0, "binding_card": 99}},
            [
                ("get_account_info", {}),
                ("get_stock_info", {"symbol": "AAPL"}),
                ("fund_account", {"amount": 100.0}),
                ("get_watchlist", {}),
                ("withdraw_funds", {"amount": 25.0}),
                ("get_account_info", {}),
            ],
        )
        result = check_trace(steps)
        self.assertEqual(result["status"], "sat")
        self.assertEqual(result["initial_state"]["balance"], 700.0)

        steps[-1]["observation"]["balance"] = 776.0
        result = check_trace(steps)
        self.assertEqual(result["status"], "unsat")
        self.assertEqual(result["first_conflicting_step"], 6)

    def test_deduces_an_unseen_initial_balance(self):
        steps = [{"tool": "fund_account", "args": {"amount": 100}, "observation": {"status": "Account funded successfully", "new_balance": 150}}]
        result = check_trace(steps)
        self.assertEqual(result["status"], "sat")
        self.assertEqual(result["initial_state"]["balance"], 50.0)

    def test_buy_then_sell_trace_has_a_witness(self):
        steps = recorded_trace(
            {"authenticated": True, "market_status": "Open", "orders": {}, "order_counter": 1, "holdings": {}},
            [
                ("get_account_info", {}),
                ("get_holdings", {}),
                ("get_market_status", {}),
                ("get_stock_info", {"symbol": "AAPL"}),
                ("place_order", {"order_type": "Buy", "symbol": "AAPL", "price": 230.0, "amount": 2}),
                ("get_order_details", {"order_id": 1}),
                ("activate_order", {"order_id": 1}),
                ("execute_order", {"order_id": 1}),
                ("place_order", {"order_type": "Sell", "symbol": "AAPL", "price": 220.0, "amount": 1}),
                ("activate_order", {"order_id": 2}),
                ("execute_order", {"order_id": 2}),
                ("get_account_info", {}),
                ("get_holdings", {}),
            ],
        )
        result = check_trace(steps)
        self.assertEqual(result["status"], "sat")
        self.assertEqual(result["initial_state"]["holdings"], {})
        self.assertEqual(result["initial_state"]["market_status"], "Open")

        steps[-1]["observation"]["holdings"] = {"AAPL": 2}
        result = check_trace(steps)
        self.assertEqual(result["status"], "unsat")
        self.assertEqual(result["first_conflicting_step"], 13)

    def test_impossible_sell_is_unsat(self):
        steps = [
            {"tool": "get_holdings", "args": {}, "observation": {"holdings": {}}},
            {"tool": "place_order", "args": {"order_type": "Sell", "symbol": "AAPL", "price": 200, "amount": 1},
             "observation": {"order_id": 1, "order_type": "Sell", "status": "Pending", "price": 200.0, "amount": 1}},
        ]
        result = check_trace(steps)
        self.assertEqual(result["status"], "unsat")
        self.assertEqual(result["first_conflicting_step"], 2)

    def test_business_errors_are_checked(self):
        steps = [{"tool": "withdraw_funds", "args": {"amount": 10}, "observation": {"error": "Insufficient funds for withdrawal."}}]
        self.assertEqual(check_trace(steps)["status"], "sat")
        steps.insert(0, {"tool": "get_account_info", "args": {}, "observation": {"balance": 30, "account_id": 1, "binding_card": 2}})
        self.assertEqual(check_trace(steps)["status"], "unsat")

    def test_watchlist_stock_filter_notification_and_histories(self):
        steps = recorded_trace(
            {"authenticated": True, "market_status": "Open"},
            [
                ("get_watchlist", {}),
                ("add_to_watchlist", {"stock": "AAPL"}),
                ("get_watchlist", {}),
                ("remove_stock_from_watchlist", {"symbol": "NVDA"}),
                ("get_watchlist", {}),
                ("filter_stocks_by_price", {"stocks": ["AAPL", "GOOG", "BAD"], "min_price": 200, "max_price": 300}),
                ("get_stock_info", {"symbol": "AAPL"}),
                ("notify_price_change", {"stocks": ["AAPL", "TSLA"], "threshold": 0.15}),
                ("get_order_history", {}),
                ("get_transaction_history", {}),
                ("fund_account", {"amount": 25}),
                ("get_transaction_history", {}),
                ("get_transaction_history", {"start_date": "2024-09-01", "end_date": "2024-09-03"}),
            ],
        )
        result = check_trace(steps)
        self.assertEqual(result["status"], "sat", result)
        modified = deepcopy(steps)
        modified[4]["observation"]["watchlist"].append("NVDA")
        self.assertEqual(check_trace(modified)["status"], "unsat")
        modified = deepcopy(steps)
        modified[11]["observation"]["transaction_history"][0]["amount"] = 30
        self.assertEqual(check_trace(modified)["status"], "unsat")

    def test_first_call_writes_watchlist(self):
        steps = recorded_trace(
            {"authenticated": True},
            [
                ("add_to_watchlist", {"stock": "AAPL"}),
                ("get_watchlist", {}),
                ("remove_stock_from_watchlist", {"symbol": "NVDA"}),
                ("get_watchlist", {}),
            ],
        )
        self.assertEqual(check_trace(steps)["status"], "sat")
        steps = recorded_trace(
            {"authenticated": True},
            [
                ("remove_stock_from_watchlist", {"symbol": "NVDA"}),
                ("get_watchlist", {}),
            ],
        )
        self.assertEqual(check_trace(steps)["status"], "sat")

    def test_all_schema_tools_in_one_valid_trace(self):
        calls = [
            ("trading_login", {"username": "user", "password": "pass"}),
            ("trading_get_login_status", {}),
            ("get_current_time", {}),
            ("get_symbol_by_name", {"name": "Apple"}),
            ("get_available_stocks", {"sector": "Technology"}),
            ("get_stock_info", {"symbol": "AAPL"}),
            ("filter_stocks_by_price", {"stocks": ["AAPL", "TSLA", "BAD"], "min_price": 100, "max_price": 300}),
            ("notify_price_change", {"stocks": ["AAPL", "TSLA"], "threshold": 0.15}),
            ("get_market_status", {}),
            ("get_account_info", {}),
            ("get_holdings", {}),
            ("get_watchlist", {}),
            ("add_to_watchlist", {"stock": "AAPL"}),
            ("remove_stock_from_watchlist", {"symbol": "NVDA"}),
            ("get_order_history", {}),
            ("get_order_details", {"order_id": 12446}),
            ("place_order", {"order_type": "Buy", "symbol": "AAPL", "price": 230, "amount": 1}),
            ("activate_order", {"order_id": 12447}),
            ("execute_order", {"order_id": 12447}),
            ("cancel_order", {"order_id": 12446}),
            ("fund_account", {"amount": 25}),
            ("withdraw_funds", {"amount": 10}),
            ("get_transaction_history", {}),
            ("trading_logout", {}),
        ]
        schema_path = Path(__file__).parents[1] / "bfcl_eval/data/multi_turn_func_doc/trading_bot.json"
        schema_names = {json.loads(line)["name"] for line in schema_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        self.assertEqual({tool for tool, _ in calls}, schema_names)
        steps = recorded_trace({"market_status": "Open"}, calls)
        result = check_trace(steps)
        self.assertEqual(result["status"], "sat", result)

    def test_backend_business_error_examples_have_witnesses(self):
        known_order = {"id": 1, "order_type": "Buy", "symbol": "AAPL", "price": 230.0, "amount": 1, "status": "Open"}
        base_order_state = {"authenticated": True, "market_status": "Open", "orders": {1: known_order}, "order_counter": 2}
        cases = [
            ({}, [("get_account_info", {})]),
            ({}, [("get_holdings", {})]),
            ({}, [("get_watchlist", {})]),
            ({}, [("get_order_history", {})]),
            ({}, [("get_transaction_history", {})]),
            ({}, [("place_order", {"order_type": "Buy", "symbol": "AAPL", "price": 230, "amount": 1})]),
            ({}, [("fund_account", {"amount": 1})]),
            ({}, [("withdraw_funds", {"amount": 1})]),
            ({}, [("remove_stock_from_watchlist", {"symbol": "NVDA"})]),
            ({}, [("get_stock_info", {"symbol": "BAD"})]),
            ({"authenticated": True}, [("remove_stock_from_watchlist", {"symbol": "BAD"})]),
            ({"authenticated": True}, [("fund_account", {"amount": 0})]),
            ({"authenticated": True}, [("withdraw_funds", {"amount": 1})]),
            ({"authenticated": True, "market_status": "Open"}, [("withdraw_funds", {"amount": 0})]),
            ({"authenticated": True, "market_status": "Open"}, [("withdraw_funds", {"amount": 10001})]),
            ({"authenticated": True}, [("place_order", {"order_type": "Buy", "symbol": "BAD", "price": 1, "amount": 1})]),
            ({"authenticated": True}, [("place_order", {"order_type": "Buy", "symbol": "AAPL", "price": 0, "amount": 1})]),
            ({"authenticated": True}, [("place_order", {"order_type": "Hold", "symbol": "AAPL", "price": 1, "amount": 1})]),
            ({"authenticated": True}, [("place_order", {"order_type": "Buy", "symbol": "AAPL", "price": 20000, "amount": 1})]),
            ({"authenticated": True, "holdings": {}}, [("place_order", {"order_type": "Sell", "symbol": "AAPL", "price": 1, "amount": 1})]),
            ({}, [("get_order_details", {"order_id": 999})]),
            ({}, [("cancel_order", {"order_id": 999})]),
            ({}, [("activate_order", {"order_id": 999})]),
            ({}, [("execute_order", {"order_id": 999})]),
            (base_order_state, [("get_order_details", {"order_id": 1}), ("activate_order", {"order_id": 1})]),
            (base_order_state, [("get_order_details", {"order_id": 1}), ("execute_order", {"order_id": 1}), ("cancel_order", {"order_id": 1})]),
            ({"orders": {1: {**known_order, "status": "Completed", "filled_price": 225.0}}},
             [("cancel_order", {"order_id": 1}), ("get_order_details", {"order_id": 1})]),
            ({**base_order_state, "market_status": "Closed"}, [("get_order_details", {"order_id": 1}), ("execute_order", {"order_id": 1})]),
            ({**base_order_state, "account_info": {"account_id": 1, "balance": 0, "binding_card": 2}}, [("get_order_details", {"order_id": 1}), ("execute_order", {"order_id": 1})]),
            ({**base_order_state, "stocks": {}}, [("get_order_details", {"order_id": 1}), ("execute_order", {"order_id": 1})]),
        ]
        for initial_state, calls in cases:
            with self.subTest(calls=calls, initial_state=initial_state):
                steps = recorded_trace(initial_state, calls)
                result = check_trace(steps)
                self.assertEqual(result["status"], "sat", result)

    def test_history_filtered_before_full_read(self):
        initial_state = {
            "authenticated": True,
            "transaction_history": [
                {"type": "deposit", "amount": 20, "timestamp": "2024-08-30 12:00:00"},
                {"type": "withdrawal", "amount": 5, "timestamp": "2024-09-01 00:00:00"},
            ],
        }
        steps = recorded_trace(initial_state, [
            ("get_transaction_history", {"start_date": "2024-09-01", "end_date": "2024-09-02"}),
            ("get_transaction_history", {}),
        ])
        self.assertEqual(check_trace(steps)["status"], "sat")
        steps[-1]["observation"]["transaction_history"][1]["amount"] = 6
        self.assertEqual(check_trace(steps)["status"], "unsat")

    def test_first_order_action_can_use_later_details(self):
        initial_state = {
            "authenticated": True,
            "market_status": "Open",
            "orders": {1: {"id": 1, "order_type": "Sell", "symbol": "AAPL", "price": 200.0, "amount": 1, "status": "Pending"}},
            "order_counter": 2,
        }
        steps = recorded_trace(initial_state, [
            ("activate_order", {"order_id": 1}),
            ("get_order_details", {"order_id": 1}),
            ("execute_order", {"order_id": 1}),
            ("get_order_details", {"order_id": 1}),
        ])
        self.assertEqual(check_trace(steps)["status"], "sat")
        initial_state["orders"][1]["status"] = "Open"
        steps = recorded_trace(initial_state, [
            ("execute_order", {"order_id": 1}),
            ("get_order_details", {"order_id": 1}),
        ])
        self.assertEqual(check_trace(steps)["status"], "sat")

    def test_fixed_lookup_distractors_are_checked(self):
        steps = recorded_trace(
            {},
            [
                ("get_symbol_by_name", {"name": "Apple"}),
                ("get_available_stocks", {"sector": "Technology"}),
            ],
        )
        self.assertEqual(check_trace(steps)["status"], "sat")
        steps[0]["observation"]["symbol"] = "GOOG"
        self.assertEqual(check_trace(steps)["status"], "unsat")

    def test_long_context_mode_is_reported_as_unsupported(self):
        bot = TradingBot()
        bot._load_scenario({"authenticated": True}, long_context=True)
        steps = [{"tool": "get_watchlist", "args": {}, "observation": bot.get_watchlist()}]
        self.assertEqual(check_trace(steps)["status"], "unsupported")


if __name__ == "__main__":
    unittest.main()
