import itertools
import random
import unittest
from copy import deepcopy
from datetime import datetime, timedelta

import z3

from bfcl_eval.consistency.trading_float import F64, as_float, fp, rounded
from bfcl_eval.consistency.trading_solver_hard import check_trace
from tests.test_trading_consistency_hard import cash, record


class AdversarialConsistencyTests(unittest.TestCase):
    def status(self, steps, expected="sat"):
        result = check_trace(steps)
        self.assertEqual(result["status"], expected, (steps, result))

    def test_previous_watchlist_false_unsat(self):
        steps = record([("add_to_watchlist", {"stock": "AAPL"}),
                        ("remove_stock_from_watchlist", {"symbol": "AAPL"}),
                        ("get_watchlist", {})], watch_list=[])
        self.status(steps)

    def test_watchlist_exhaustive_three_operation_traces(self):
        operations = [("add_to_watchlist", {"stock": s}) for s in ("AAPL", "GOOG")]
        operations += [("remove_stock_from_watchlist", {"symbol": s}) for s in ("AAPL", "GOOG")]
        states = [[], ["AAPL"], ["GOOG"], ["AAPL", "GOOG"], ["GOOG", "AAPL"], ["AAPL", "AAPL"]]
        for initial, calls in itertools.product(states, itertools.product(operations, repeat=3)):
            with self.subTest(initial=initial, calls=calls):
                self.status(record([*calls, ("get_watchlist", {})], watch_list=initial))

    def test_numeric_boundary_without_early_balance_anchor(self):
        steps = [{"tool": "withdraw_funds", "args": {"amount": .2},
                  "observation": {"error": "Insufficient funds for withdrawal."}},
                 {"tool": "get_account_info", "args": {}, "observation": cash(.2)}]
        self.status(steps, "unsat")
        self.status(record([("fund_account", {"amount": .1}), ("get_account_info", {})],
                           account_info=cash(.2)))

    def test_order_counter_and_missing_ids(self):
        initial = {4: {"id": 4, "order_type": "Buy", "symbol": "AAPL", "price": 230,
                       "amount": 1, "status": "Pending"}}
        calls = [("place_order", {"order_type": "Buy", "symbol": "AAPL", "price": 230, "amount": 1})] * 2
        steps = record([*calls, ("get_order_history", {})], orders=initial, order_counter=3)
        self.status(steps)
        steps[-1]["observation"]["history"] = [3, 4, 5]
        self.status(steps, "unsat")
        steps = record([("get_order_details", {"order_id": 3}), *calls,
                        ("get_order_history", {})], orders={}, order_counter=3)
        self.status(steps)

    def test_history_thousand_records_and_duplicates(self):
        start = datetime(2024, 9, 1)
        for distinct in (False, True):
            history = [{"type": "deposit", "amount": 1,
                        "timestamp": (start + timedelta(seconds=i if distinct else 0)).strftime("%Y-%m-%d %H:%M:%S")}
                       for i in range(1000)]
            self.status(record([("get_transaction_history", {})], transaction_history=history))

    def test_overlapping_history_filters_preserve_global_order(self):
        history = [{"type": "deposit", "amount": i + 1,
                    "timestamp": f"2024-09-0{i + 1} 00:00:00"} for i in range(3)]
        calls = [("get_transaction_history", {"start_date": "2024-09-01", "end_date": "2024-09-02"}),
                 ("get_transaction_history", {"start_date": "2024-09-02", "end_date": "2024-09-03"}),
                 ("get_transaction_history", {})]
        steps = record(calls, transaction_history=history)
        self.status(steps)
        steps[-1]["observation"]["transaction_history"].reverse()
        self.status(steps, "unsat")

    def test_notification_symbols_with_commas(self):
        self.status(record([("get_stock_info", {"symbol": "A, B"}),
                            ("notify_price_change", {"stocks": ["A, B"], "threshold": .5})],
                           stocks={"A, B": {"price": 1, "percent_change": .6}}))

    def test_business_error_text_matches_the_executed_branch(self):
        self.status([{"tool": "get_order_details", "args": {"order_id": 1},
                      "observation": {"error": "Order with ID 1 not found."}}], "unsat")
        self.status([{"tool": "get_order_details", "args": {"order_id": 1},
                      "observation": {"error": "Order with ID 1 not found.Here is the list of orders_id: [2,3]"}}], "unsat")
        self.status([{"tool": "activate_order", "args": {"order_id": 1},
                      "observation": {"error": "Can't activate order 1. Order is already OPEN."}}], "unsat")
        self.status([{"tool": "execute_order", "args": {"order_id": 1},
                      "observation": {"error": "Can't execute order 1. Order is COMPLETED."}}], "unsat")

    def test_random_timestamp_year_padding(self):
        self.status([
            {"tool": "get_transaction_history", "args": {}, "observation": {"transaction_history": []}},
            {"tool": "fund_account", "args": {"amount": 1}, "observation": {"status": "Account funded successfully"}},
            {"tool": "get_transaction_history", "args": {},
             "observation": {"transaction_history": [{"type": "deposit", "amount": 1,
                                                        "timestamp": "0001-01-01 00:00:00"}]}},
        ])

    def test_rounding_bitvector_matches_python(self):
        rng = random.Random(1731)
        values = [rng.uniform(-1e8, 1e8) for _ in range(400)]
        values += [i / 1000 for i in range(-3000, 3000, 5)]
        values += [0., -0., 5e-324, -5e-324, 1e308, -1e308]
        variable = z3.FP("rounding_test_value", F64)
        for digits in (2, 3):
            expression = rounded(variable, digits)
            for value in values:
                actual = as_float(z3.simplify(z3.substitute(expression, (variable, fp(value)))))
                self.assertEqual(actual, round(value, digits), (value, digits, actual))

    def test_seeded_backend_financial_traces_and_mutations(self):
        for seed in range(40):
            rng = random.Random(seed)
            price = round(rng.uniform(1, 30), 3)
            initial = round(rng.uniform(200, 500), 3)
            side = rng.choice(["Buy", "Sell"])
            calls = [("get_account_info", {}), ("get_holdings", {}),
                     ("get_stock_info", {"symbol": "AAPL"}),
                     ("fund_account", {"amount": round(rng.uniform(.001, 3), 3)}),
                     ("execute_order", {"order_id": 1}),
                     ("withdraw_funds", {"amount": round(rng.uniform(.001, 3), 3)}),
                     ("get_account_info", {}), ("get_holdings", {})]
            steps = record(calls, account_info=cash(initial), holdings={"AAPL": 10},
                           stocks={"AAPL": {"price": price, "percent_change": 0}},
                           orders={1: {"id": 1, "order_type": side, "symbol": "AAPL", "price": price,
                                       "amount": rng.randint(1, 5), "status": "Open"}})
            with self.subTest(seed=seed):
                self.status(steps)
                bad = deepcopy(steps)
                bad[-2]["observation"]["balance"] = round(bad[-2]["observation"]["balance"] + .001, 3)
                self.status(bad, "unsat")

    def test_seeded_mixed_success_and_business_error_traces(self):
        for seed in range(30):
            rng = random.Random(seed + 400)
            calls = [("get_account_info", {}), ("get_stock_info", {"symbol": "AAPL"}),
                     ("get_stock_info", {"symbol": "GOOG"}),
                     ("get_order_details", {"order_id": 1}), ("get_order_details", {"order_id": 2})]
            operations = [
                ("get_account_info", {}), ("get_holdings", {}), ("get_market_status", {}),
                ("get_watchlist", {}), ("get_order_history", {}), ("get_transaction_history", {}),
                ("trading_login", {"username": "u", "password": "p"}), ("trading_logout", {}),
                ("trading_get_login_status", {}),
                ("fund_account", {"amount": rng.choice([-.1, 0, .123, 10])}),
                ("withdraw_funds", {"amount": rng.choice([-.1, 0, .123, 100000])}),
                ("add_to_watchlist", {"stock": rng.choice(["AAPL", "GOOG", "MISSING"])}),
                ("remove_stock_from_watchlist", {"symbol": rng.choice(["AAPL", "GOOG"])}),
                ("get_stock_info", {"symbol": "MISSING"}),
                ("place_order", {"order_type": rng.choice(["Buy", "Sell", "Invalid"]), "symbol": "AAPL", "price": 2.675, "amount": rng.choice([0, 1, 10])}),
                ("activate_order", {"order_id": rng.choice([1, 2, 99])}),
                ("cancel_order", {"order_id": rng.choice([1, 2, 99])}),
                ("execute_order", {"order_id": rng.choice([1, 2, 99])}),
            ]
            calls += [rng.choice(operations) for _ in range(16)]
            calls += [("trading_login", {"username": "u", "password": "p"}),
                      ("get_account_info", {}), ("get_holdings", {}), ("get_watchlist", {}),
                      ("get_order_history", {}), ("get_transaction_history", {})]
            with self.subTest(seed=seed):
                self.status(record(calls, account_info=cash(50.125), order_counter=3,
                                   market_status=rng.choice(["Open", "Closed"]),
                                   holdings={"AAPL": 3, "GOOG": 0}, watch_list=["GOOG"],
                                   stocks={"AAPL": {"price": 2.675, "percent_change": .001},
                                           "GOOG": {"price": 1.234, "percent_change": -.003}},
                                   orders={1: {"id": 1, "symbol": "AAPL", "order_type": "Buy", "price": 3,
                                               "amount": 2, "status": rng.choice(["Pending", "Open", "Completed", "Cancelled"])},
                                           2: {"id": 2, "symbol": "GOOG", "order_type": "Sell", "price": 1,
                                               "amount": 3, "status": "Open"}}))


if __name__ == "__main__":
    unittest.main()
