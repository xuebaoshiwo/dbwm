import json
import unittest
from pathlib import Path

from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.trading_bot import TradingBot


class TradingOrderLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.bot = TradingBot()
        self.bot._load_scenario(
            {
                "authenticated": True,
                "orders": {},
                "order_counter": 1,
                "holdings": {},
            }
        )

    def place_order(self):
        result = self.bot.place_order("Buy", "AAPL", 10.0, 2)
        self.assertEqual(result["status"], "Pending")
        return result["order_id"]

    def test_order_changes_only_after_explicit_activation(self):
        order_id = self.place_order()
        balance = self.bot.get_account_info()["balance"]
        self.assertEqual(self.bot.get_order_details(order_id)["status"], "Pending")
        self.assertEqual(self.bot.activate_order(order_id), {"order_id": order_id, "status": "Open"})
        self.assertEqual(self.bot.get_order_details(order_id)["status"], "Open")
        self.assertEqual(self.bot.get_account_info()["balance"], balance)
        self.assertIn("already open", self.bot.activate_order(order_id)["error"])
        self.assertEqual(self.bot.cancel_order(order_id), {"order_id": order_id, "status": "Cancelled"})

    def test_cancellation_is_terminal(self):
        order_id = self.place_order()
        self.assertEqual(self.bot.cancel_order(order_id), {"order_id": order_id, "status": "Cancelled"})
        self.assertEqual(self.bot.get_order_details(order_id)["status"], "Cancelled")
        self.assertIn("already cancelled", self.bot.cancel_order(order_id)["error"])
        self.assertIn("cancelled", self.bot.activate_order(order_id)["error"])

    def test_completed_and_unknown_orders_cannot_advance(self):
        self.assertIn("not found", self.bot.activate_order(999)["error"])
        self.bot.orders[2] = {"status": "Completed"}
        self.assertIn("completed", self.bot.activate_order(2)["error"])
        self.assertIn("already completed", self.bot.cancel_order(2)["error"])

    def test_buy_fill_updates_cash_holdings_and_order(self):
        self.bot.market_status = "Open"
        placed = self.bot.place_order("Buy", "AAPL", 230.0, 2)
        order_id = placed["order_id"]
        self.assertEqual(self.bot.get_holdings(), {"holdings": {}})
        self.assertIn("pending", self.bot.execute_order(order_id)["error"])
        self.bot.activate_order(order_id)

        result = self.bot.execute_order(order_id)
        self.assertEqual(result["status"], "Completed")
        self.assertEqual(result["filled_price"], 227.16)
        self.assertEqual(result["new_balance"], 9545.68)
        self.assertEqual(result["shares_held"], 2)
        self.assertEqual(self.bot.get_holdings(), {"holdings": {"AAPL": 2}})
        self.assertEqual(self.bot.get_order_details(order_id)["filled_price"], 227.16)
        self.assertIn("completed", self.bot.execute_order(order_id)["error"])
        self.assertIn("already completed", self.bot.cancel_order(order_id)["error"])

    def test_sell_fill_updates_cash_and_removes_empty_position(self):
        self.bot.market_status = "Open"
        self.bot.holdings = {"AAPL": 2}
        order_id = self.bot.place_order("Sell", "AAPL", 220.0, 2)["order_id"]
        self.bot.activate_order(order_id)

        result = self.bot.execute_order(order_id)
        self.assertEqual(result["new_balance"], 10454.32)
        self.assertEqual(result["shares_held"], 0)
        self.assertEqual(self.bot.get_holdings(), {"holdings": {}})
        self.assertIn("Insufficient shares", self.bot.place_order("Sell", "AAPL", 220.0, 1)["error"])

    def test_failed_fills_leave_order_and_finances_unchanged(self):
        order_id = self.bot.place_order("Buy", "AAPL", 200.0, 2)["order_id"]
        self.bot.activate_order(order_id)
        self.assertEqual(self.bot.get_market_status(), {"market_status": "Closed"})
        self.assertIn("Market is closed", self.bot.execute_order(order_id)["error"])
        self.bot.market_status = "Open"
        self.assertIn("limit price", self.bot.execute_order(order_id)["error"])
        self.assertEqual(self.bot.get_order_details(order_id)["status"], "Open")
        self.assertEqual(self.bot.get_account_info()["balance"], 10000.0)
        self.assertEqual(self.bot.get_holdings(), {"holdings": {}})

    def test_balance_and_holdings_are_checked_again_at_fill(self):
        self.bot.market_status = "Open"
        buy_id = self.bot.place_order("Buy", "AAPL", 230.0, 2)["order_id"]
        self.bot.activate_order(buy_id)
        self.bot.withdraw_funds(9900.0)
        self.assertIn("Insufficient funds", self.bot.execute_order(buy_id)["error"])
        self.assertEqual(self.bot.get_order_details(buy_id)["status"], "Open")

        self.bot.holdings = {"AAPL": 2}
        first_id = self.bot.place_order("Sell", "AAPL", 220.0, 2)["order_id"]
        second_id = self.bot.place_order("Sell", "AAPL", 220.0, 2)["order_id"]
        self.bot.activate_order(first_id)
        self.bot.activate_order(second_id)
        self.bot.execute_order(first_id)
        self.assertIn("Insufficient shares", self.bot.execute_order(second_id)["error"])
        self.assertEqual(self.bot.get_order_details(second_id)["status"], "Open")

    def test_default_order_ids_are_not_overwritten(self):
        bot = TradingBot()
        bot._load_scenario({"authenticated": True})
        existing_order = bot.get_order_details(12446).copy()
        new_id = bot.place_order("Buy", "AAPL", 230.0, 1)["order_id"]
        self.assertEqual(new_id, 12447)
        self.assertEqual(bot.get_order_details(12446), existing_order)

    def test_scenario_state_is_not_shared_between_instances(self):
        scenario = {
            "authenticated": True,
            "market_status": "Open",
            "orders": {},
            "order_counter": 1,
            "account_info": {"balance": 10000.0},
            "holdings": {},
        }
        first = TradingBot()
        second = TradingBot()
        first._load_scenario(scenario)
        second._load_scenario(scenario)
        order_id = first.place_order("Buy", "AAPL", 230.0, 1)["order_id"]
        first.activate_order(order_id)
        first.execute_order(order_id)
        self.assertEqual(second.get_holdings(), {"holdings": {}})
        self.assertEqual(second.get_account_info()["balance"], 10000.0)
        self.assertEqual(scenario["holdings"], {})

    def test_tool_schema_describes_activation(self):
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "bfcl_eval/data/multi_turn_func_doc/trading_bot.json"
        )
        schemas = {
            entry["name"]: entry
            for entry in map(json.loads, schema_path.read_text(encoding="utf-8").splitlines())
        }
        self.assertEqual(schemas["activate_order"]["parameters"]["required"], ["order_id"])
        self.assertIn("Pending", schemas["activate_order"]["description"])
        self.assertIn("activate_order", schemas["place_order"]["description"])
        self.assertEqual(schemas["execute_order"]["parameters"]["required"], ["order_id"])
        self.assertIn("holdings", schemas["get_holdings"]["response"]["properties"])
        self.assertIn("market_status", schemas["get_market_status"]["response"]["properties"])


if __name__ == "__main__":
    unittest.main()
