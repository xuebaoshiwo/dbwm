# Backward consistency trajectories

The research goal is to test whether a world model, acting as the environment
over a long tool-call sequence, returns mutually consistent observations. Each
monitored state field is anchored by an early observation, changed by several
tool calls, and read again later. Consistency is checked across all returned
observations and the implied state transitions, not only the two endpoint reads.

The active sampler is `generate_backward_trajectories.py`. It plans symbolic
success-branch tool chains; arguments, initial state, observations, and backend
execution are left to grounding. A generated chain is not yet an executable
test case: branch guards, entity bindings, and order lifecycles may be
incompatible. Grounding must instantiate and execute the chain, then reject
and resample any case that cannot run. See `BACKWARD_SAMPLING.md` for the
sampling rules, CLI options, and limitations.

From the repository root:

```powershell
python -m bfcl_eval.consistency.data_generator_v2.generate_backward_trajectories `
  --count 10 --min-length 20 --max-length 60 --min-writes 3 `
  --output-dir bfcl_eval/consistency/trajectories_v2/trading_bot_hard/backward/new_run
```

The output directory must be empty. The retained example batch is
`trajectories_v2/trading_bot_hard/backward/demo_10_min20_forward_reuse_20260928`.

Verify with:

```powershell
python -m unittest tests.test_backward_trajectory_sampling tests.test_generate_tool_state_specs -q
```
