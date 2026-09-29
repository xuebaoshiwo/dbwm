# Long-horizon consistency trajectory generator

The pipeline has a domain-independent scheduler (`core.py`) and a backend adapter
(`trading.py`). A trajectory first fixes two reads of a target state field and a
specified number of successful writes between them. `utils/model_client.py`
chooses read-only distractor calls for the remaining slots. Every call is then
executed on a fresh real backend loaded with one initial state. Invalid model
responses are retried and explicitly marked as deterministic fallback. Model
output never supplies ground-truth observations.

Run from the repository root:

```bash
python -m bfcl_eval.consistency.data_generator.generate \
  --domain trading_bot --lengths 5 8 12 --read-gaps 2 5 9 \
  --write-counts 1 2 3 --workers 4
```

The defaults use the 10 states in
`bfcl_eval/consistency/initial_states_pools/trading_bot/trading_states.jsonl`,
producing one trajectory per state at each of the three lengths (30 files).
Although named `.jsonl`, that pool contains pretty-printed JSON objects;
`load_initial_states` parses concatenated JSON values rather than individual
lines. `--model-client` can point at another compatible module with `chat_text`.

If the provider fails transiently, regenerate only files whose source was
recorded as `deterministic_fallback` while preserving successful model files:

```bash
python -m bfcl_eval.consistency.data_generator.generate \
  --retry-fallback --workers 2 --retries 4
```

`length` counts all tool calls, including login. `read_gap` counts tool calls
strictly between the two target reads. `write_count` counts calls that actually
change the target backend field within that gap. `write_frequency` in each
file is `write_count / read_gap`. These values are validated against snapshots
of the real backend after every call. TradingBot traces are also checked by
`trading_solver.check_trace` and must return `sat`.

Each JSON file contains a natural-language policy `query`, the full `tool_schema`,
initial-state snapshot/reference, controls, model proposal provenance, and
`steps` in the input format accepted by the consistency solver. The query lists
all requested actions in order and never includes observations.

The TradingBot adapter corrects one upstream response-schema typo in the
generated `tool_schema`: `get_order_history` returns `history` in the backend,
although the source schema calls it `order_history`. This avoids asking a
policy to infer the wrong response field from the supplied schema.

To add another domain, implement the `DomainAdapter` interface in `core.py`:
provide a skeleton with two explicit target reads, safe distractor candidates,
real-backend rollout, target-value extraction, human-readable call descriptions,
and schema loading. A domain can optionally provide `validate_trace`. Register
its state loader and job factory in `generate.py`; the scheduler, model proposal,
controls, and output format remain shared.

Offline tests:

```bash
python -m pytest bfcl_eval/consistency/data_generator/test_pipeline.py -q
```
