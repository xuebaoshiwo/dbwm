# Post-state observation rollback

This snapshot was taken before the version-1.2 post-state observation changes.
It includes the earlier lifecycle-rule implementation.

Original tool specifications are in `trading_bot_hard/`. Original generator
files are in `data_generator_v2/`. The two original test files are at this
snapshot directory's root.

To undo this change, restore these generator files from `data_generator_v2/`
to `D:/dbwm/gorilla/berkeley-function-call-leaderboard/bfcl_eval/consistency/data_generator_v2/`:

- `generate_tool_state_specs.py`
- `migrate_trading_bot_hard_sources.py`
- `generate_backward_trajectories.py`
- `symbolic_dependency_graph.py`
- `trading_bot_hard_lifecycle.json`
- `BACKWARD_SAMPLING.md`

Restore all JSON files from `trading_bot_hard/` to
`D:/dbwm/gorilla/berkeley-function-call-leaderboard/bfcl_eval/consistency/data_v2/trading_bot_hard/`.
Restore the two test files to that project's `tests/` directory.
Review any newer edits before overwriting those files. Existing trajectory
batches were not modified.
