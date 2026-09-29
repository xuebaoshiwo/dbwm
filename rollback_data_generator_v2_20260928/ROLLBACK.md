# Rollback snapshot

These four files are copies from immediately before the lifecycle-rule change:

- `generate_backward_trajectories.py`
- `trading_bot_hard_symbolic.json`
- `BACKWARD_SAMPLING.md`
- `test_backward_trajectory_sampling.py`

For a temporary behavior rollback, run the generator with
`--no-lifecycle-rules`. To restore the exact pre-change files, copy the four
snapshots back to their matching locations in
`D:/dbwm/gorilla/berkeley-function-call-leaderboard`, after checking for any
newer edits you want to keep. The newly added `lifecycle_rules.py` and
`trading_bot_hard_lifecycle.json` are unused after the catalog is restored.
