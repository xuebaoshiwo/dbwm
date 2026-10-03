# TradingBot observation consistency checker

For the **TradingBotHard** checker used with `data_generator_v2`, see
[HARD_SOLVER.md](HARD_SOLVER.md). It supports three-decimal monetary inputs,
excludes the three fixed lookup tools by default, and provides configurable
transaction timestamp policies. The description below concerns the original
`trading_solver.py` checker.

The checker asks whether **one possible initial TradingBot state** can explain a
sequence of tool calls and predicted observations. Initial cash, holdings,
authentication, market status, stock prices, order keys, and watchlist contents
are inferred from the sequence. Reads constrain the current state; successful
writes update it; business errors constrain why a write did not happen.

All 24 tools in `bfcl_eval/data/multi_turn_func_doc/trading_bot.json` are
modeled for ordinary TradingBot mode. This includes stock filtering and price
notifications, watchlist mutations, order history, and filtered transaction
history. The backend is replayed to verify each `sat` witness. Transaction
timestamps are treated as nondeterministic values within the backend's one-day
generation range and injected during replay; the checker does not prove that a
particular Python random seed produces those exact timestamps.

Install the optional dependency with `pip install -e ".[consistency]"`, then run:

```text
python -m bfcl_eval.consistency.trading_solver trace.json
```

The input is a JSON object with a `steps` array:

```json
{
  "steps": [
    {"tool": "fund_account", "args": {"amount": 100},
     "observation": {"status": "Account funded successfully", "new_balance": 150}},
    {"tool": "get_account_info", "args": {},
     "observation": {"account_id": 12345, "balance": 150, "binding_card": 1974202140965533}}
  ]
}
```

A complete 22-step example is in `examples/trading_buy_sell_trace.json`.

Tool names may have a `TradingBot.` prefix. `observation` is the full value
predicted by the world model. It can also be a backend business-error response,
including the list-shaped authentication errors returned by some history tools.

Results:

- `sat`: a concrete initial-state witness passed backend replay. The result
  includes that witness and any inferred transaction timestamps.
- `unsat`: no state satisfies the modeled constraints. The result identifies
  the first conflicting step and Z3's conflicting step set when available.
- `unsupported`: the trace uses a format or path outside the model's stated
  assumptions. This is **not** a contradiction.
- `unknown`: Z3 timed out or a candidate could not be replayed exactly. This is
  also **not** a contradiction.

The current assumptions are nonnegative initial cash and holdings, integer
share counts, monetary values and stock prices with at most two decimals,
ordinary TradingBot mode (`long_context=False`), and a three-second Z3 timeout
per prefix. A successful execution of an unseen initial order requires an
earlier or later `get_order_details` observation so its symbol and side can be
identified. The model may return `unsupported` for other unusual initial-order
or malformed-call patterns. Initial order keys and watchlist entries are
inferred from observations; unseen entries are omitted when they do not affect
any observed response.

Run the verification suite with:

```text
python -m unittest tests.test_trading_consistency_solver tests.test_trading_consistency_detection tests.test_trading_consistency_mixed tests.test_trading_order_lifecycle
```

The fixed mutation corpus contains 100 deliberately inconsistent traces across
10 state relationships. All 100 are currently detected as `unsat`; 80 seeded
12-step trajectories recorded from the real backend are accepted as `sat`.
These are regression measurements on these corpora, **not** an estimated
detection rate for arbitrary world-model outputs. A representative WM corpus
with manually labeled inconsistencies is needed to estimate that rate.

For such a corpus, save one JSON object per line with `steps` and a boolean
`inconsistent` label, then run:

```text
python -m bfcl_eval.consistency.evaluate labeled_traces.jsonl
```

`detection_rate` is `unsat` among all labeled inconsistent traces. An
`unsupported` or `unknown` result counts as a miss, so the rate cannot be
inflated by dropping hard cases. `false_positive_rate` is `unsat` among labeled
consistent traces. The report lists every result that differs from its label.
