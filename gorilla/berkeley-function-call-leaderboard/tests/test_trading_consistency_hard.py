import unittest
from copy import deepcopy

from bfcl_eval.consistency.trading_solver_hard import check_trace, _replay_observations_equal
from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.trading_bot_hard import TradingBot


def record(calls, **state):
    bot = TradingBot()
    bot._load_scenario({"authenticated": True, "market_status": "Open",
                        "orders": {}, "order_counter": 1, "transaction_history": [],
                        **state})
    steps = []
    for tool, args in calls:
        steps.append({"tool": tool, "args": args,
                      "observation": deepcopy(getattr(bot, tool)(**args))})
    return steps


def cash(balance):
    return {"account_id": 7, "binding_card": 99, "balance": balance}


class TradingConsistencyHardTests(unittest.TestCase):
    def assert_status(self, steps, status="sat", **options):
        result = check_trace(steps, **options)
        self.assertEqual(result["status"], status, result)
        return result

    def test_fixed_outputs_are_ignored_by_default_and_can_be_enabled(self):
        cases = [("get_current_time", {}, {"current_time": "11:59 PM"}),
                 ("get_symbol_by_name", {"name": "Apple"}, {"symbol": "OTHER"}),
                 ("get_available_stocks", {"sector": "Technology"}, {"stock_list": ["OTHER"]})]
        for tool, args, obs in cases:
            with self.subTest(tool=tool):
                steps = [{"tool": "TradingBot." + tool, "args": args, "observation": obs}]
                self.assert_status(steps)
                self.assert_status(steps, "unsat", check_static_tools=True)

    def test_fixed_outputs_do_not_pollute_state_or_step_numbers(self):
        steps = record([("get_holdings", {}), ("get_holdings", {})], holdings={"AAPL": 2})
        steps.insert(1, {"tool": "get_symbol_by_name", "args": {},
                         "observation": {"holdings": {"FAKE": 20}}})
        self.assert_status(steps)
        steps[-1]["observation"]["holdings"]["AAPL"] = 3
        result = self.assert_status(steps, "unsat")
        self.assertEqual(result["first_conflicting_step"], 3)

    def test_empty_trace(self):
        self.assert_status([])

    def history_trace(self, timestamp):
        steps = record([("get_transaction_history", {}), ("fund_account", {"amount": 1.123}),
                        ("get_transaction_history", {}), ("get_transaction_history", {})])
        for step in steps[2:]:
            step["observation"]["transaction_history"][0]["timestamp"] = timestamp
        return steps

    def test_timestamps_need_not_match_seed_or_hardcoded_date(self):
        self.assert_status(self.history_trace("2024-09-01 17:18:19"), timestamp_policy="backend")
        steps = self.history_trace("2026-10-01 17:18:19")
        self.assert_status(steps)
        self.assert_status(steps, "unsat", timestamp_policy="backend")

    def test_random_timestamp_still_must_be_consistent(self):
        steps = self.history_trace("2026-10-01 17:18:19")
        steps[-1]["observation"]["transaction_history"][0]["timestamp"] = "2026-10-01 17:18:20"
        self.assert_status(steps, "unsat")

    def test_random_timestamp_filters_still_apply(self):
        steps = self.history_trace("2026-10-01 17:18:19")
        steps[-1]["args"] = {"start_date": "2026-10-02"}
        steps[-1]["observation"]["transaction_history"] = []
        self.assert_status(steps)
        steps[-1]["observation"] = deepcopy(steps[-2]["observation"])
        self.assert_status(steps, "unsat")

    def test_three_decimals_and_float_residue(self):
        steps = record([("get_account_info", {}), ("fund_account", {"amount": .1}),
                        ("fund_account", {"amount": .003}), ("withdraw_funds", {"amount": .002}),
                        ("get_account_info", {})], account_info=cash(.2))
        self.assert_status(steps)
        steps[-1]["observation"]["balance"] = .301
        self.assert_status(steps)
        steps[-1]["observation"]["balance"] = .302
        self.assert_status(steps, "unsat")

    def test_four_decimal_money_is_outside_declared_format(self):
        steps = record([("get_account_info", {})], account_info=cash(1.0001))
        self.assert_status(steps, "invalid_format")

    def test_trade_and_cash_rounding_match_backend(self):
        for price in [2.675, 2.685, 1.234, .005, .015]:
            for kind in ["Buy", "Sell"]:
                for amount in [1, 3]:
                    with self.subTest(price=price, kind=kind, amount=amount):
                        steps = record(
                            [("get_account_info", {}), ("get_stock_info", {"symbol": "AAPL"}),
                             ("place_order", {"order_type": kind, "symbol": "AAPL", "price": price, "amount": amount}),
                             ("activate_order", {"order_id": 1}), ("execute_order", {"order_id": 1}),
                             ("get_account_info", {})],
                            account_info=cash(100.005), holdings={"AAPL": 10},
                            stocks={"AAPL": {"price": price, "percent_change": 0, "volume": 1, "MA(5)": price, "MA(20)": price}})
                        self.assert_status(steps)
                        steps[-1]["observation"]["balance"] += .001
                        self.assert_status(steps, "unsat")

    def test_float_boundary_business_errors(self):
        # 0.3 - 0.1 is slightly below 0.2 in the backend.
        steps = record([("get_account_info", {}), ("withdraw_funds", {"amount": .1}),
                        ("withdraw_funds", {"amount": .2}), ("get_account_info", {})],
                       account_info=cash(.3))
        self.assertIn("error", steps[2]["observation"])
        self.assert_status(steps)
        # float(0.1) * 3 is slightly above float(0.3).
        steps = record([("get_account_info", {}),
                        ("place_order", {"order_type": "Buy", "symbol": "AAPL", "price": .1, "amount": 3})],
                       account_info=cash(.3))
        self.assertIn("error", steps[-1]["observation"])
        self.assert_status(steps)

    def test_formatted_error_money_is_a_rounded_value(self):
        steps = record([("get_account_info", {}),
                        ("place_order", {"order_type": "Buy", "symbol": "AAPL", "price": 1, "amount": 3})],
                       account_info=cash(1.234))
        self.assert_status(steps)
        steps = record([("get_stock_info", {"symbol": "AAPL"}),
                        ("place_order", {"order_type": "Buy", "symbol": "AAPL", "price": 1.23, "amount": 1}),
                        ("activate_order", {"order_id": 1}), ("execute_order", {"order_id": 1})],
                       stocks={"AAPL": {"price": 1.234, "percent_change": 0}})
        self.assert_status(steps)

    def test_stock_filters_use_three_decimal_prices(self):
        steps = record([("get_stock_info", {"symbol": "AAPL"}),
                        ("filter_stocks_by_price", {"stocks": ["AAPL"], "min_price": 1.234, "max_price": 1.234})],
                       stocks={"AAPL": {"price": 1.234, "percent_change": 0}})
        self.assert_status(steps)

    def test_unread_initial_order_is_solved_existentially(self):
        steps = record([("activate_order", {"order_id": 12}),
                        ("execute_order", {"order_id": 12})],
                       orders={12: {"id": 12, "symbol": "AAPL", "order_type": "Buy",
                                    "price": 300, "amount": 5, "status": "Pending"}})
        self.assert_status(steps)

    def test_unread_orders_are_jointly_constrained_by_cash_and_holdings(self):
        steps = record(
            [("get_account_info", {}), ("get_holdings", {}),
             ("execute_order", {"order_id": 11}), ("execute_order", {"order_id": 12}),
             ("get_account_info", {}), ("get_holdings", {})],
            account_info=cash(100), holdings={"ALPHA": 5, "BETA": 10},
            stocks={"ALPHA": {"price": 10}, "BETA": {"price": 20}},
            orders={11: {"id": 11, "symbol": "ALPHA", "order_type": "Buy",
                         "price": 15, "amount": 2, "status": "Open"},
                    12: {"id": 12, "symbol": "BETA", "order_type": "Sell",
                         "price": 18, "amount": 3, "status": "Open"}})
        result = self.assert_status(steps)
        first = result["initial_state"]["initial_orders"][11]
        second = result["initial_state"]["initial_orders"][12]
        self.assertEqual((first["symbol"], first["order_type"]), ("ALPHA", "Buy"))
        self.assertEqual((second["symbol"], second["order_type"]), ("BETA", "Sell"))
        # The inferred limits need not match the generating state's limits.
        self.assertGreaterEqual(first["price"], 10)
        self.assertLessEqual(second["price"], 20)
        bad = deepcopy(steps)
        bad[-2]["observation"]["balance"] += .001
        self.assert_status(bad, "unsat")
        bad = deepcopy(steps)
        bad[-1]["observation"]["holdings"]["ALPHA"] += 1
        self.assert_status(bad, "unsat")

    def test_unread_order_identity_and_lifecycle_persist(self):
        order = {"id": 12, "symbol": "AAPL", "order_type": "Buy",
                 "price": 300, "amount": 5, "status": "Open"}
        steps = record([("execute_order", {"order_id": 12})], orders={12: order})
        self.assert_status(steps)
        # Same ID cannot be executed twice, even though its fields were unknown.
        self.assert_status(steps + deepcopy(steps), "unsat")
        steps = record([("activate_order", {"order_id": 12}),
                        ("execute_order", {"order_id": 12}),
                        ("get_order_details", {"order_id": 12})],
                       orders={12: {**order, "status": "Pending"}})
        self.assert_status(steps)
        steps[-1]["observation"]["amount"] = 6
        self.assert_status(steps, "unsat")

    def test_zero_holding_key_is_equivalent_to_absence(self):
        steps = record([("get_holdings", {}), ("get_holdings", {})], holdings={"AAPL": 0})
        self.assert_status(steps)
        steps[-1]["observation"]["holdings"] = {}
        original = deepcopy(steps)
        self.assertTrue(self.assert_status(steps)["replay_verified"])
        self.assertEqual(steps, original)
        self.assertTrue(self.assert_status(list(reversed(steps)))["replay_verified"])

    def test_zero_equivalence_does_not_hide_nonzero_or_malformed_counts(self):
        for count, status in [(1, "unsat"), (-1, "unsat"), (False, "invalid_format"),
                              (0.0, "invalid_format"), ("0", "invalid_format")]:
            with self.subTest(count=count):
                steps = record([("get_holdings", {}), ("get_holdings", {})], holdings={})
                steps[-1]["observation"]["holdings"] = {"AAPL": count}
                self.assert_status(steps, status)
        steps = record([("get_holdings", {}), ("get_holdings", {})], holdings={"AAPL": 2})
        steps[-1]["observation"]["holdings"] = {}
        self.assert_status(steps, "unsat")

    def test_replay_zero_equivalence_is_scoped_to_holdings(self):
        zero, empty = {"holdings": {"AAPL": 0}}, {"holdings": {}}
        self.assertTrue(_replay_observations_equal("get_holdings", zero, empty))
        self.assertTrue(_replay_observations_equal("get_holdings", empty, zero))
        self.assertFalse(_replay_observations_equal("get_account_info", zero, empty))
        self.assertFalse(_replay_observations_equal("get_holdings", {**zero, "extra": 0}, empty))

    def test_buy_creates_and_sell_removes_holding_key(self):
        steps = record(
            [("get_holdings", {}), ("execute_order", {"order_id": 1}),
             ("get_holdings", {}), ("execute_order", {"order_id": 2}), ("get_holdings", {})],
            holdings={}, stocks={"AAPL": {"price": 10}},
            orders={1: {"id": 1, "symbol": "AAPL", "order_type": "Buy",
                        "price": 10, "amount": 2, "status": "Open"},
                    2: {"id": 2, "symbol": "AAPL", "order_type": "Sell",
                        "price": 10, "amount": 2, "status": "Open"}})
        self.assert_status(steps)
        steps[-1]["observation"]["holdings"] = {"AAPL": 0}
        self.assertTrue(self.assert_status(steps)["replay_verified"])
        steps[-1]["tool"] = "TradingBot.get_holdings"
        self.assert_status(steps)
        steps[-1]["observation"]["holdings"] = {"AAPL": 1}
        self.assert_status(steps, "unsat")

    def test_unread_order_errors_constrain_initial_attributes(self):
        steps = record([("get_holdings", {}), ("execute_order", {"order_id": 1})],
                       holdings={}, orders={1: {"id": 1, "symbol": "AAPL", "order_type": "Sell",
                                                "price": 200, "amount": 2, "status": "Open"}})
        self.assertIn("Insufficient shares", steps[-1]["observation"]["error"])
        self.assert_status(steps)
        steps = record([("execute_order", {"order_id": 1})],
                       orders={1: {"id": 1, "symbol": "MISSING", "order_type": "Buy",
                                   "price": 100, "amount": 2, "status": "Open"}})
        self.assert_status(steps)

    def test_add_then_multiple_removes(self):
        steps = record([("get_watchlist", {}), ("add_to_watchlist", {"stock": "AAPL"}),
                        ("remove_stock_from_watchlist", {"symbol": "MSFT"}),
                        ("remove_stock_from_watchlist", {"symbol": "TSLA"}),
                        ("remove_stock_from_watchlist", {"symbol": "NVDA"}),
                        ("get_watchlist", {})], watch_list=["NVDA", "MSFT", "TSLA"])
        self.assert_status(steps)
        steps[-1]["observation"]["watchlist"] = []
        self.assert_status(steps, "unsat")


if __name__ == "__main__":
    unittest.main()
