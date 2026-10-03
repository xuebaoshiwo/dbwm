"""Independent, hand-selected common situations; observations come from Python."""
from copy import deepcopy

from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.trading_bot_hard import DEFAULT_STATE


def fixtures():
    base = deepcopy(DEFAULT_STATE)
    base.update(authenticated=True, market_status="Open", random_seed=37, order_counter=801)
    base["account_info"] = {"account_id": 601, "balance": 12000.0, "binding_card": 730001}
    base["orders"] = {
        701: {"id": 701, "order_type": "Buy", "symbol": "AAPL", "price": 240.0, "amount": 4, "status": "Pending"},
        702: {"id": 702, "order_type": "Sell", "symbol": "GOOG", "price": 2800.0, "amount": 2, "status": "Open"},
        703: {"id": 703, "order_type": "Buy", "symbol": "MSFT", "price": 320.0, "amount": 3, "status": "Completed", "filled_price": 310.23},
    }
    base["transaction_history"] = [
        {"type": "deposit", "amount": 700.0, "timestamp": "2024-08-10 11:20:00"},
        {"type": "withdrawal", "amount": 90.5, "timestamp": "2024-08-12 13:45:00"},
    ]
    result = {}

    def add(tool, label, args=None, **state):
        scenario = deepcopy(base)
        scenario.update(deepcopy(state))
        result.setdefault(tool, []).append({"label": label, "args": args or {}, "initial_state": scenario})

    for label, state in [("logged in", {}), ("logged out", {"authenticated": False}),
                         ("different account", {"account_info": {"account_id": 602, "balance": 83.25, "binding_card": 730002}})]:
        add("get_account_info", label, **state)
    for label, state in [("nonempty holdings", {}), ("empty holdings", {"holdings": {}}), ("login required", {"authenticated": False})]:
        add("get_holdings", label, **state)
    for tool, field, empty in [("get_watchlist", "watch_list", []), ("get_order_history", "orders", {})]:
        add(tool, "nonempty collection")
        add(tool, "empty collection", **{field: empty})
        add(tool, "login required", authenticated=False)
    add("get_transaction_history", "all records")
    add("get_transaction_history", "bounded date filter", {"start_date": "2024-08-11", "end_date": "2024-08-13"})
    add("get_transaction_history", "empty history", transaction_history=[])
    for label, value in [("open", "Open"), ("closed", "Closed"), ("another open scenario", "Open")]:
        add("get_market_status", label, market_status=value)
    for label, value in [("logged in", True), ("logged out", False), ("another authenticated user", True)]:
        add("trading_get_login_status", label, authenticated=value)
    for label, authenticated, username in [("new login", False, "demo_alice"), ("already logged in", True, "demo_alice"), ("another user", False, "demo_bob")]:
        add("trading_login", label, {"username": username, "password": "example_password"}, authenticated=authenticated)
    for label, value in [("logout", True), ("already logged out", False), ("another logout", True)]:
        add("trading_logout", label, authenticated=value)
    for tool in ("fund_account", "withdraw_funds"):
        add(tool, "successful positive amount", {"amount": 325.75})
        add(tool, "nonpositive amount", {"amount": 0})
        if tool == "fund_account":
            add(tool, "login required", {"amount": 50}, authenticated=False)
        else:
            add(tool, "insufficient balance", {"amount": 15000})
    for label, stock in [("new entry", "AAPL"), ("already present", "NVDA"), ("unknown symbol", "UNKNOWN")]:
        add("add_to_watchlist", label, {"stock": stock})
    add("remove_stock_from_watchlist", "remove existing", {"symbol": "NVDA"})
    add("remove_stock_from_watchlist", "absent entry", {"symbol": "AAPL"})
    add("remove_stock_from_watchlist", "login required", {"symbol": "NVDA"}, authenticated=False)
    for label, symbol in [("first stock", "AAPL"), ("second stock", "GOOG"), ("unknown symbol", "UNKNOWN")]:
        add("get_stock_info", label, {"symbol": symbol})
    for label, name in [("first company", "Apple"), ("second company", "Google"), ("unknown company", "Unknown Company")]:
        add("get_symbol_by_name", label, {"name": name})
    for sector in ("Technology", "Automobile", "Unknown"):
        add("get_available_stocks", sector, {"sector": sector})
    for label, state in [("ordinary scenario", {}), ("closed market", {"market_status": "Closed"}), ("logged out", {"authenticated": False})]:
        add("get_current_time", label, **state)
    for label, low, high in [("subset matches", 200, 400), ("no matches", 1, 10), ("all match", 0, 5000)]:
        add("filter_stocks_by_price", label, {"stocks": ["AAPL", "GOOG", "MSFT"], "min_price": low, "max_price": high})
    for label, stocks, threshold in [("some changes", ["AAPL", "NVDA"], 0.2), ("no significant changes", ["GOOG"], 1.0), ("multiple changes", ["AAPL", "TSLA", "NVDA"], 0.1)]:
        add("notify_price_change", label, {"stocks": stocks, "threshold": threshold})
    for label, oid in [("pending order", 701), ("completed order", 703), ("missing order", 999)]:
        add("get_order_details", label, {"order_id": oid})
    for label, oid in [("activate pending", 701), ("already open", 702), ("missing order", 999)]:
        add("activate_order", label, {"order_id": oid})
    for label, oid in [("cancel pending", 701), ("cancel open", 702), ("completed order", 703)]:
        add("cancel_order", label, {"order_id": oid})
    opened = deepcopy(base["orders"])
    opened[701]["status"] = "Open"
    add("execute_order", "successful buy", {"order_id": 701}, orders=opened)
    add("execute_order", "successful sell", {"order_id": 702})
    add("execute_order", "order not open", {"order_id": 701})
    add("place_order", "buy limit order", {"order_type": "Buy", "symbol": "AAPL", "price": 240.0, "amount": 2})
    add("place_order", "sell limit order", {"order_type": "Sell", "symbol": "GOOG", "price": 2800.0, "amount": 1})
    add("place_order", "insufficient funds", {"order_type": "Buy", "symbol": "AAPL", "price": 240.0, "amount": 100})
    return result
