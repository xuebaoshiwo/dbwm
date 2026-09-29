"""Ten handcrafted TradingBot scenarios for probing world-model consistency.

Each scenario is a dict with four keys:

- ``name``: short identifier.
- ``theme``: which dimension of long-horizon consistency the scenario stresses.
- ``state``: a complete initial state, loadable via ``TradingBot()._load_scenario(state)``,
  standing in for ``DEFAULT_STATE``.
- ``probe``: a ground-truth rollout ``[(tool, args), ...]``.  Executing it against a
  backend loaded with ``state`` must reproduce a trace that ``trading_solver.check_trace``
  answers ``sat``.
- ``corrupt``: ``{"step": 1-based probe index, "path": [...], "value": ...}`` — one
  observation mutation that must flip the verdict to ``unsat``.

Money amounts deliberately use float-exact decimals (.0/.25/.5/.75) so the integer-cent
SMT model and the float backend agree during replay.
"""

STOCK_AAPL = {"price": 150.25, "percent_change": 0.17, "volume": 2.552, "MA(5)": 150.1, "MA(20)": 149.8}
STOCK_GOOG = {"price": 2800.0, "percent_change": 0.24, "volume": 1.123, "MA(5)": 2795.0, "MA(20)": 2802.0}
STOCK_TSLA = {"price": 660.0, "percent_change": -0.8, "volume": 1.654, "MA(5)": 661.0, "MA(20)": 659.5}
STOCK_MSFT = {"price": 310.5, "percent_change": 0.09, "volume": 3.234, "MA(5)": 310.0, "MA(20)": 310.2}
STOCK_NVDA = {"price": 220.34, "percent_change": 0.34, "volume": 1.234, "MA(5)": 220.4, "MA(20)": 220.6}
STOCK_ROBO = {"price": 88.5, "percent_change": 1.2, "volume": 1.1, "MA(5)": 88.0, "MA(20)": 87.5}
STOCK_LUNA = {"price": 42.5, "percent_change": -3.0, "volume": 2.2, "MA(5)": 42.0, "MA(20)": 43.0}


def _account(account_id, balance, binding_card=1974202140965533):
    return {"account_id": account_id, "balance": balance, "binding_card": binding_card}


def _order(order_id, order_type, symbol, price, amount, status):
    return {"id": order_id, "order_type": order_type, "symbol": symbol,
            "price": price, "amount": amount, "status": status}


def _tx(kind, amount, timestamp):
    return {"type": kind, "amount": amount, "timestamp": timestamp}


SCENARIOS = [
    {
        "name": "balance_chain",
        "theme": "余额长链 + 资金流水的时间窗过滤（fund/withdraw 写记录，初始记录被日期范围区分）",
        "state": {
            "orders": {},
            "account_info": _account(90001, 5000.0),
            "holdings": {"AAPL": 10},
            "authenticated": False,
            "market_status": "Open",
            "order_counter": 11,
            "stocks": {"AAPL": dict(STOCK_AAPL)},
            "watch_list": ["AAPL"],
            "transaction_history": [
                _tx("deposit", 1000.0, "2024-08-20 09:00:00"),
                _tx("withdrawal", 200.0, "2024-08-25 16:45:00"),
            ],
            "random_seed": 101,
        },
        "probe": [
            ("TradingBot.trading_login", {"username": "alice", "password": "x"}),
            ("TradingBot.fund_account", {"amount": 500.0}),
            ("TradingBot.withdraw_funds", {"amount": 300.0}),
            ("TradingBot.get_stock_info", {"symbol": "AAPL"}),
            ("TradingBot.place_order", {"order_type": "buy", "symbol": "AAPL", "price": 150.5, "amount": 10}),
            ("TradingBot.activate_order", {"order_id": 11}),
            ("TradingBot.execute_order", {"order_id": 11}),
            ("TradingBot.get_holdings", {}),
            ("TradingBot.get_transaction_history", {}),
            ("TradingBot.get_transaction_history", {"start_date": "2024-09-01"}),
            ("TradingBot.get_account_info", {}),
        ],
        "corrupt": {"step": 11, "path": ["observation", "balance"], "value": 3697.6},
    },
    {
        "name": "order_book_inference",
        "theme": "初始订单簿推断：history 全列表锚定 ID 集，details 读初始订单，跨 place/activate/execute/cancel 状态机",
        "state": {
            "orders": {
                101: _order(101, "Buy", "AAPL", 150.25, 5, "Pending"),
                102: _order(102, "Sell", "GOOG", 2800.0, 2, "Open"),
                103: _order(103, "Buy", "TSLA", 660.0, 3, "Completed"),
            },
            "account_info": _account(90002, 8000.0),
            "holdings": {"AAPL": 8, "GOOG": 5, "TSLA": 1},
            "authenticated": False,
            "market_status": "Open",
            "order_counter": 104,
            "stocks": {"AAPL": dict(STOCK_AAPL), "GOOG": dict(STOCK_GOOG), "TSLA": dict(STOCK_TSLA)},
            "watch_list": [],
            "transaction_history": [],
            "random_seed": 102,
        },
        "probe": [
            ("TradingBot.trading_login", {"username": "alice", "password": "x"}),
            ("TradingBot.get_order_history", {}),
            ("TradingBot.get_order_details", {"order_id": 101}),
            ("TradingBot.get_order_details", {"order_id": 102}),
            ("TradingBot.get_order_details", {"order_id": 103}),
            ("TradingBot.activate_order", {"order_id": 101}),
            ("TradingBot.execute_order", {"order_id": 101}),
            ("TradingBot.cancel_order", {"order_id": 103}),
            ("TradingBot.place_order", {"order_type": "sell", "symbol": "GOOG", "price": 2800.0, "amount": 2}),
            ("TradingBot.cancel_order", {"order_id": 104}),
            ("TradingBot.get_order_history", {}),
            ("TradingBot.get_account_info", {}),
        ],
        "corrupt": {"step": 7, "path": ["observation", "new_balance"], "value": 7248.8},
    },
    {
        "name": "watchlist_ordering",
        "theme": "watchlist 顺序敏感性 + add 的幂等/静默忽略（已存在、不存在的股票都不动列表）",
        "state": {
            "orders": {},
            "account_info": _account(90003, 3000.0),
            "holdings": {"TSLA": 2},
            "authenticated": False,
            "market_status": "Open",
            "order_counter": 31,
            "stocks": {"TSLA": dict(STOCK_TSLA), "AAPL": dict(STOCK_AAPL),
                       "NVDA": dict(STOCK_NVDA), "ROBO": dict(STOCK_ROBO)},
            "watch_list": ["TSLA", "AAPL"],
            "transaction_history": [],
            "random_seed": 103,
        },
        "probe": [
            ("TradingBot.trading_login", {"username": "alice", "password": "x"}),
            ("TradingBot.get_watchlist", {}),
            ("TradingBot.add_to_watchlist", {"stock": "TSLA"}),
            ("TradingBot.add_to_watchlist", {"stock": "NVDA"}),
            ("TradingBot.add_to_watchlist", {"stock": "ROBO"}),
            ("TradingBot.add_to_watchlist", {"stock": "FAKE"}),
            ("TradingBot.remove_stock_from_watchlist", {"symbol": "TSLA"}),
            ("TradingBot.get_watchlist", {}),
            ("TradingBot.remove_stock_from_watchlist", {"symbol": "FAKE"}),
        ],
        "corrupt": {"step": 8, "path": ["observation", "watchlist"], "value": ["AAPL", "ROBO", "NVDA"]},
    },
    {
        "name": "auth_flip_flop",
        "theme": "登录态翻转链：错误观测反向钉死 auth，login/logout 返回值与状态迁移互查",
        "state": {
            "orders": {},
            "account_info": _account(90004, 500.0),
            "holdings": {"AAPL": 7},
            "authenticated": False,
            "market_status": "Open",
            "order_counter": 61,
            "stocks": {"AAPL": dict(STOCK_AAPL)},
            "watch_list": ["AAPL"],
            "transaction_history": [],
            "random_seed": 104,
        },
        "probe": [
            ("TradingBot.get_account_info", {}),
            ("TradingBot.trading_get_login_status", {}),
            ("TradingBot.trading_login", {"username": "alice", "password": "x"}),
            ("TradingBot.get_holdings", {}),
            ("TradingBot.get_watchlist", {}),
            ("TradingBot.trading_logout", {}),
            ("TradingBot.get_account_info", {}),
            ("TradingBot.trading_login", {"username": "alice", "password": "x"}),
            ("TradingBot.place_order", {"order_type": "buy", "symbol": "AAPL", "price": 150.25, "amount": 1}),
            ("TradingBot.get_order_history", {}),
            ("TradingBot.get_account_info", {}),
        ],
        "corrupt": {"step": 8, "path": ["observation", "status"], "value": "Already logged in"},
    },
    {
        "name": "closed_market_errors",
        "theme": "闭市错误模板互证：withdraw/execute 的 market-closed 错误与市场状态查询互相钉死",
        "state": {
            "orders": {55: _order(55, "Buy", "GOOG", 2800.0, 2, "Open")},
            "account_info": _account(90005, 2000.0),
            "holdings": {"GOOG": 3},
            "authenticated": False,
            "market_status": "Closed",
            "order_counter": 56,
            "stocks": {"GOOG": dict(STOCK_GOOG)},
            "watch_list": [],
            "transaction_history": [],
            "random_seed": 105,
        },
        "probe": [
            ("TradingBot.get_market_status", {}),
            ("TradingBot.trading_login", {"username": "alice", "password": "x"}),
            ("TradingBot.fund_account", {"amount": 4000.0}),
            ("TradingBot.withdraw_funds", {"amount": 100.0}),
            ("TradingBot.get_stock_info", {"symbol": "GOOG"}),
            ("TradingBot.get_order_details", {"order_id": 55}),
            ("TradingBot.execute_order", {"order_id": 55}),
            ("TradingBot.place_order", {"order_type": "buy", "symbol": "GOOG", "price": 2800.0, "amount": 1}),
            ("TradingBot.activate_order", {"order_id": 56}),
            ("TradingBot.execute_order", {"order_id": 56}),
            ("TradingBot.get_market_status", {}),
            ("TradingBot.get_account_info", {}),
        ],
        "corrupt": {"step": 11, "path": ["observation", "market_status"], "value": "Open"},
    },
    {
        "name": "cross_tool_price",
        "theme": "价格跨工具一致性：get_stock_info × filter_stocks_by_price × notify_price_change × 成交 filled_price",
        "state": {
            "orders": {},
            "account_info": _account(90006, 0.0),
            "holdings": {},
            "authenticated": False,
            "market_status": "Open",
            "order_counter": 71,
            "stocks": {"AAPL": dict(STOCK_AAPL), "GOOG": dict(STOCK_GOOG), "LUNA": dict(STOCK_LUNA)},
            "watch_list": [],
            "transaction_history": [],
            "random_seed": 106,
        },
        "probe": [
            ("TradingBot.trading_login", {"username": "alice", "password": "x"}),
            ("TradingBot.get_stock_info", {"symbol": "AAPL"}),
            ("TradingBot.get_stock_info", {"symbol": "LUNA"}),
            ("TradingBot.filter_stocks_by_price", {"stocks": ["AAPL", "GOOG", "LUNA", "GHOST"], "min_price": 100.0, "max_price": 800.0}),
            ("TradingBot.notify_price_change", {"stocks": ["AAPL", "GOOG", "LUNA"], "threshold": 1.0}),
            ("TradingBot.fund_account", {"amount": 1000.0}),
            ("TradingBot.place_order", {"order_type": "buy", "symbol": "LUNA", "price": 43.0, "amount": 10}),
            ("TradingBot.activate_order", {"order_id": 71}),
            ("TradingBot.execute_order", {"order_id": 71}),
            ("TradingBot.get_account_info", {}),
        ],
        "corrupt": {"step": 4, "path": ["observation", "filtered_stocks"], "value": ["AAPL", "LUNA", "GOOG"]},
    },
    {
        "name": "sell_side_limits",
        "theme": "卖单侧：持仓不足错误、卖限价 vs 市价的错误模板（错误串里嵌的数字精确钉死价格）",
        "state": {
            "orders": {},
            "account_info": _account(90007, 2000.0),
            "holdings": {"TSLA": 3},
            "authenticated": False,
            "market_status": "Open",
            "order_counter": 81,
            "stocks": {"TSLA": dict(STOCK_TSLA)},
            "watch_list": [],
            "transaction_history": [],
            "random_seed": 107,
        },
        "probe": [
            ("TradingBot.trading_login", {"username": "alice", "password": "x"}),
            ("TradingBot.get_stock_info", {"symbol": "TSLA"}),
            ("TradingBot.place_order", {"order_type": "sell", "symbol": "TSLA", "price": 665.0, "amount": 5}),
            ("TradingBot.place_order", {"order_type": "sell", "symbol": "TSLA", "price": 665.0, "amount": 3}),
            ("TradingBot.activate_order", {"order_id": 81}),
            ("TradingBot.execute_order", {"order_id": 81}),
            ("TradingBot.cancel_order", {"order_id": 81}),
            ("TradingBot.get_holdings", {}),
        ],
        "corrupt": {"step": 6, "path": ["observation", "error"],
                    "value": "Sell limit price of $665.00 is above the current stock price of $670.00."},
    },
    {
        "name": "insufficient_funds_template",
        "theme": "买单余额不足错误模板：错误串里 'only $X available' 钉死余额，boundary 恰好充足的第二单验证链恢复",
        "state": {
            "orders": {},
            "account_info": _account(90008, 1000.0),
            "holdings": {},
            "authenticated": False,
            "market_status": "Open",
            "order_counter": 91,
            "stocks": {"AAPL": dict(STOCK_AAPL)},
            "watch_list": [],
            "transaction_history": [],
            "random_seed": 108,
        },
        "probe": [
            ("TradingBot.trading_login", {"username": "alice", "password": "x"}),
            ("TradingBot.get_account_info", {}),
            ("TradingBot.place_order", {"order_type": "buy", "symbol": "AAPL", "price": 150.25, "amount": 10}),
            ("TradingBot.fund_account", {"amount": 600.0}),
            ("TradingBot.place_order", {"order_type": "buy", "symbol": "AAPL", "price": 160.0, "amount": 10}),
            ("TradingBot.activate_order", {"order_id": 91}),
            ("TradingBot.execute_order", {"order_id": 91}),
            ("TradingBot.get_account_info", {}),
        ],
        "corrupt": {"step": 3, "path": ["observation", "error"],
                    "value": "Insufficient funds: required $1502.50 but only $900.00 available."},
    },
    {
        "name": "history_time_windows",
        "theme": "流水时间窗：三条初始记录分布在八月，fund 写记录落在九月，窗口查询的包含/排除逻辑",
        "state": {
            "orders": {},
            "account_info": _account(90009, 2000.0),
            "holdings": {"AAPL": 4},
            "authenticated": False,
            "market_status": "Open",
            "order_counter": 41,
            "stocks": {"AAPL": dict(STOCK_AAPL)},
            "watch_list": [],
            "transaction_history": [
                _tx("deposit", 800.0, "2024-08-10 09:00:00"),
                _tx("withdrawal", 150.0, "2024-08-20 14:30:00"),
                _tx("deposit", 250.0, "2024-08-30 11:15:00"),
            ],
            "random_seed": 109,
        },
        "probe": [
            ("TradingBot.trading_login", {"username": "alice", "password": "x"}),
            ("TradingBot.get_transaction_history", {}),
            ("TradingBot.get_transaction_history", {"start_date": "2024-08-15", "end_date": "2024-08-31"}),
            ("TradingBot.fund_account", {"amount": 500.0}),
            ("TradingBot.get_transaction_history", {"start_date": "2024-09-01"}),
            ("TradingBot.get_transaction_history", {}),
            ("TradingBot.get_account_info", {}),
        ],
        "corrupt": {"step": 3, "path": ["observation", "transaction_history"], "value": [
            _tx("deposit", 800.0, "2024-08-10 09:00:00"),
            _tx("withdrawal", 150.0, "2024-08-20 14:30:00"),
            _tx("deposit", 250.0, "2024-08-30 11:15:00"),
        ]},
    },
    {
        "name": "combined_long_chain",
        "theme": "综合长链：认证→入金→watchlist→买单成交→初始卖单错误/取消→卖单成交→流水→登出登入→终态对账",
        "state": {
            "orders": {201: _order(201, "Sell", "AAPL", 151.0, 1, "Open")},
            "account_info": _account(90010, 6000.0),
            "holdings": {"AAPL": 2, "MSFT": 5},
            "authenticated": False,
            "market_status": "Open",
            "order_counter": 202,
            "stocks": {"AAPL": dict(STOCK_AAPL), "MSFT": dict(STOCK_MSFT)},
            "watch_list": ["MSFT"],
            "transaction_history": [_tx("deposit", 300.0, "2024-08-28 10:00:00")],
            "random_seed": 110,
        },
        "probe": [
            ("TradingBot.trading_login", {"username": "alice", "password": "x"}),
            ("TradingBot.get_account_info", {}),
            ("TradingBot.fund_account", {"amount": 500.0}),
            ("TradingBot.add_to_watchlist", {"stock": "AAPL"}),
            ("TradingBot.get_watchlist", {}),
            ("TradingBot.place_order", {"order_type": "buy", "symbol": "AAPL", "price": 150.5, "amount": 10}),
            ("TradingBot.activate_order", {"order_id": 202}),
            ("TradingBot.execute_order", {"order_id": 202}),
            ("TradingBot.get_holdings", {}),
            ("TradingBot.get_order_details", {"order_id": 201}),
            ("TradingBot.execute_order", {"order_id": 201}),
            ("TradingBot.cancel_order", {"order_id": 201}),
            ("TradingBot.place_order", {"order_type": "sell", "symbol": "MSFT", "price": 310.0, "amount": 4}),
            ("TradingBot.activate_order", {"order_id": 203}),
            ("TradingBot.execute_order", {"order_id": 203}),
            ("TradingBot.get_holdings", {}),
            ("TradingBot.get_transaction_history", {}),
            ("TradingBot.trading_logout", {}),
            ("TradingBot.trading_login", {"username": "alice", "password": "x"}),
            ("TradingBot.get_account_info", {}),
        ],
        "corrupt": {"step": 20, "path": ["observation", "balance"], "value": 6239.6},
    },
]
