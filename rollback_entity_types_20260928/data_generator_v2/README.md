# Backward consistency trajectories

The research goal is to test whether a world model, acting as the environment
over a long tool-call sequence, returns mutually consistent observations. Each
monitored state field is anchored by an early observation, changed by several
tool calls, and read again later. Consistency is checked across all returned
observations and the implied state transitions, not only the two endpoint reads.

The active sampler is `generate_backward_trajectories.py`. Monitoring selects
entity instances and their fields. Every call has explicit symbolic entity
bindings; lifecycle rules apply to all affected instances, including unmonitored
orders introduced to change holdings. Concrete keys, arguments, initial state,
observations and backend execution remain grounding obligations.

## Start here

For a new session, read these documents in order:

1. [DESIGN_CONTEXT.md](DESIGN_CONTEXT.md): the user's research goal, accepted
   decisions, implementation status, known gaps, and rollback locations.
2. [BACKWARD_SAMPLING.md](BACKWARD_SAMPLING.md): the current backward algorithm,
   dependency fixing, observation timing, write counting, CLI and output fields.
3. The target catalog and lifecycle rules listed below, followed by the sampler
   and its tests when changing behavior.

Entity-mode write counts are best-effort goals. Fields without legal writers
retain their observation intervals and report a shortfall. Legacy catalogs
without entity types retain their original minimum-write behavior.

## Pipeline and files

| File | Responsibility |
| --- | --- |
| `generate_tool_state_specs.py` | Generate schema-1.2 dependency JSON from a supplied backend and tool schema. |
| `trading_bot_hard_symbolic.json` | Example domain catalog: specification paths, monitoring targets and reader refinements. |
| `trading_bot_hard_lifecycle.json` | Example domain rules for the monitored order's state transitions. |
| `lifecycle_rules.py` | Parse and evaluate the external lifecycle rules. |
| `entity_bindings.py` | Define entity types, expand monitored instances and bind call keys and invariant entity references. |
| `entity_lifecycle.py` | Track lifecycle frontiers per instance, add prerequisite transitions and audit complete lifecycles. |
| `generate_backward_trajectories.py` | Sample success branches backward and emit symbolic plans. |
| `backward_state_knowledge.py` | Track observed state, propagate writes and invalidate unknown regions. |
| `symbolic_dependency_graph.py` | Match symbolic paths and load dependency specifications. |
| `migrate_trading_bot_hard_sources.py` | Migrate the existing TradingBotHard specifications using reviewed backend facts. |

The generic specification generator and lifecycle engine accept domain data;
the TradingBotHard migration is intentionally specific to that backend. The
sampler currently has a fixed `long_context=false` assumption, documented in
the algorithm notes and design context.

## Run

From the repository root:

```powershell
python -m bfcl_eval.consistency.data_generator_v2.generate_backward_trajectories `
  --targets balance transaction_history orders `
  --count 10 --min-length 20 --max-length 60 --min-writes 3 `
  --output-dir bfcl_eval/consistency/trajectories_v2/trading_bot_hard/backward/new_run
```

Monitor two independent orders, with multiple fields per order:

```powershell
python -m bfcl_eval.consistency.data_generator_v2.generate_backward_trajectories `
  --monitor order_A=order:status,amount --monitor order_B=order:status,amount `
  --count 10 --min-length 20 --max-length 60 --min-writes 3 `
  --output-dir bfcl_eval/consistency/trajectories_v2/trading_bot_hard/backward/two_orders
```

For holdings alone, use `--monitor position=stock:holding` or
`--targets holdings`. Repeated executions allocate distinct orders and enforce
each order's lifecycle even though its status is not monitored. The stock type
shares one symbol identity between the stock record and its holding.

The output directory must be empty. The CLI loads the catalog's lifecycle rules
by default; `--lifecycle-rules PATH` overrides them and `--no-lifecycle-rules`
disables them. Adding a target to the catalog does not add it to the CLI default
selection: include its name in `--targets`.

`--target-min-writes NAME=N` overrides individual field counts; explicit
monitors use names such as `order_A.status=2`. `--targets` and `--monitor` are
mutually exclusive. The sampler retains its backward random selection strategy;
it does not use the catalog's legacy `writer_sequences`.

For the Python API, pass the catalog's `entity_types` alongside its loaded
lifecycle rules, then select instance monitors:

```python
config = json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8"))
specs, targets, refinements, _ = load_inputs(DEFAULT_CATALOG)
rules, _ = load_lifecycle_rules(DEFAULT_CATALOG, specs, targets)
sampler = BackwardSampler(specs, targets, reader_refinements=refinements,
                          lifecycle_rules=rules, entity_types=config["entity_types"])
plan = sampler.build(monitors=[
    {"id": "order_A", "type": "order", "fields": ["status", "amount"]},
    {"id": "order_B", "type": "order", "fields": ["status", "amount"]},
])
```

`monitored_entities` is the monitoring selection; `planning.entities` also
includes supporting instances. `planning.target_chains` contains the individual
field intervals. Use `sampler.node_for_step(step)` when auditing bound calls;
`sampler.nodes` contains unbound tool templates.

## Verify

```powershell
python -m unittest tests.test_backward_trajectory_sampling tests.test_generate_tool_state_specs tests.test_entity_trajectory_sampling -q
```
