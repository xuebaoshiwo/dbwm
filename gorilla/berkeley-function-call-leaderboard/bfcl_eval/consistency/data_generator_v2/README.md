# Backward consistency trajectories

The research goal is to test whether a world model, acting as the environment
over a long tool-call sequence, returns mutually consistent observations. Each
monitored state field is anchored by an early observation, changed by several
tool calls, and read again later. Consistency is checked across all returned
observations and the implied state transitions, not only the two endpoint reads.

The active sampler is `generate_backward_trajectories.py`. It plans symbolic
success-branch tool chains; arguments, initial state, observations, and backend
execution are left to grounding. Configured lifecycle rules filter illegal
writer order, mutually exclusive transitions, and limited branch uses per
numbered placeholder, including unmonitored side effects. A generated chain
still requires concrete arguments, initial state and backend execution.

## Start here

For a new session, read these documents in order:

1. [DESIGN_CONTEXT.md](DESIGN_CONTEXT.md): the user's research goal, accepted
   decisions, implementation status, known gaps, and rollback locations.
2. [BACKWARD_SAMPLING.md](BACKWARD_SAMPLING.md): the current backward algorithm,
   dependency fixing, observation timing, write counting, CLI and output fields.
3. The target catalog and lifecycle rules listed below, followed by the sampler
   and its tests when changing behavior.

The user's requested behavior and the current implementation differ for write
shortfalls on targets without lifecycle rules. This is an explicit open issue
in the design context; do not infer that every shortfall is already accepted.

## Pipeline and files

| File | Responsibility |
| --- | --- |
| `generate_tool_state_specs.py` | Generate schema-1.2 dependency JSON from a supplied backend and tool schema. |
| `trading_bot_hard_symbolic.json` | Example domain catalog: specification paths, monitoring targets and reader refinements. |
| `trading_bot_hard_lifecycle.json` | Example domain rules applied independently to every bound order. |
| `lifecycle_rules.py` | Parse and evaluate the external lifecycle rules. |
| `placeholder_bindings.py` | Allocate numbered placeholders and substitute call-local aliases. |
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

The output directory must be empty. The CLI loads the catalog's lifecycle rules
by default; `--lifecycle-rules PATH` overrides them and `--no-lifecycle-rules`
disables them. Adding a target to the catalog does not add it to the CLI default
selection: include its name in `--targets`.

`--min-writes` is a best-effort goal for lifecycle-constrained targets and a
required minimum for other targets. `--target-min-writes NAME=N` overrides
individual counts. The sampler retains its backward random selection strategy;
it does not use the catalog's legacy `writer_sequences`.

Repeat a placeholder target to monitor independent instances, for example
`--targets orders orders holdings`. The resulting order chains are named
`orders_1` and `orders_2`, with paths containing `{order_id_1}` and
`{order_id_2}`. Use `--target-min-writes orders_1=2 orders_2=2` to override
their counts. A static field such as `balance` cannot be repeated.
The catalog also offers `watch_list` as a static monitoring target.

The sampler binds anchors, mutations, return sources and dependencies before
planning. Different numbered placeholders cannot reuse each other's facts.
Each additional holdings execution gets its own order identity and is checked
against that order's sampled transitions, even when `orders` is omitted from
monitoring. Lifecycle filtering does not insert predecessor calls: an order's
first execution can require Open as its initial state. No entity-type
registration is required; lifecycle transitions remain domain data.

## Verify

```powershell
python -m unittest tests.test_backward_trajectory_sampling tests.test_generate_tool_state_specs tests.test_placeholder_trajectory_sampling -q
```
