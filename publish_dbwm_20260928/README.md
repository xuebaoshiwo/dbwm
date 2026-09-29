# DBWM

Current workspace snapshot of the BFCL world-model consistency experiments.
The source layout is preserved so existing relative model-client paths work.

- `gorilla/`: Gorilla/BFCL source, including the consistency generators,
  specifications, example trajectories, backends, and tests.
- `utils/`: Shared model client. Set `MICUAPI_API_KEY` or `OPENAI_API_KEY`
  in the environment before making model calls.

The symbolic backward sampler and its current limitations are documented in
[`data_generator_v2/README.md`](gorilla/berkeley-function-call-leaderboard/bfcl_eval/consistency/data_generator_v2/README.md).

Run the generator tests from `gorilla/berkeley-function-call-leaderboard`:

```powershell
python -m unittest tests.test_backward_trajectory_sampling tests.test_generate_tool_state_specs -q
```

This snapshot is based on Gorilla commit
`6ea5797` from <https://github.com/ShishirPatil/gorilla> and includes the local
experiment changes. Local rollback directories, credentials, and runtime caches
are excluded. The current sampler has not yet been redesigned for multiple
independent monitored entities.
