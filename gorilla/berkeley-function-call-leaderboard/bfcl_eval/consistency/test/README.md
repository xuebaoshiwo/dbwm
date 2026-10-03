# World-model long-horizon consistency evaluation

Run commands from `gorilla/berkeley-function-call-leaderboard`.

The runner is domain independent. A JSON configuration selects the task text,
complete tool schema (JSON or JSONL), examples, and a `module:function` checker
with interface `check_trace(steps, **options) -> {"status": ...}`. Only the
optional fixtures and configuration under `domains/` know about TradingBot.

## Evaluation modes

- `endpoint`: one independent test per monitored field. Resolve
  `planning.target_chains[field].read_step` from the original symbolic plan
  against stable grounded `step_id` values. Include *all* earlier calls and real
  observations, predict the entire final observation once, and solve the prefix
  including that prediction. Do not include future calls, future observations,
  or predictions from the other fields. A mutating call can be the final read.
  Two fields sharing a final read still get independent requests.
- `rollout`: start with an empty history and predict every observation in order.
  Each subsequent request includes only previous WM observations. The checker
  evaluates the full predicted trace. Each trajectory is sequential internally;
  separate trajectories run concurrently. Invalid JSON/API failure ends that
  trajectory without substituting real observations.

Both modes send five inputs: domain task description, the sequence's agent query,
the full schema, three examples for the current tool, and the call/observation
history plus the current call. Each accepted sequence must have a non-empty
top-level `query` string; it is sent as `agent_query` on every prediction and
saved in the case record. The WM is asked to produce plausible observations that
support completion of the query's task while remaining consistent with history
and tool rules, without prescribing a state-tracking or deduction procedure. The query's
intended actions are not treated as completed actions. This is a generation
instruction; the score still measures trace consistency, not task completion.
Actual initial state, replay snapshots, branch labels, target
metadata, future calls, and reference answer are never sent. Examples are
independent fixtures, not selected from evaluation sequences. Their JSON stores
provenance states for reproducibility, but prompts omit that provenance.

## Examples and validation

```powershell
python -m bfcl_eval.consistency.test.generate_examples --config bfcl_eval/consistency/test/domains/trading_bot_hard.json
python -m unittest bfcl_eval.consistency.test.test_pipeline -v
```

There are 24 tools and 72 backend-executed examples. Fixtures select three
representative common situations, including normal and error behavior, without
claiming a measured frequency ranking. Some tools have fewer than three
distinct behaviors; the fixed clock has only one possible observation. Those
tools use three scenarios with repeated behavior rather than fabricated outputs.

## First experiment: DeepSeek Flash with thinking, seven threads

```powershell
python -m bfcl_eval.consistency.test.run `
  --config bfcl_eval/consistency/test/domains/trading_bot_hard.json `
  --input-dir bfcl_eval/consistency/data_v2/trading_bot_hard/grounded_20261001_124001_32k `
  --symbolic-dir bfcl_eval/consistency/data_v2/trading_bot_hard/raw_tool_chain_linked_20260930 `
  --output-dir bfcl_eval/consistency/test/results/deepseek_flash_endpoint_thinking_20261002 `
  --mode endpoint --model deepseek-flash --thinking enabled --workers 7
```

Use `--dry-run` to enumerate tests without model calls. This batch contains
45 source sequences and 90 endpoint tests. Use `--mode rollout` and a new output
directory for the second mode. `--limit` caps independent tests, not sequences.

On 2026-10-02 the configured gateway rejected `deepseek-v4-flash` with HTTP 503
`model_not_found`; its model list exposed `deepseek-flash`. The user approved
using that gateway name. Results preserve the actual request/response model ID;
the gateway did not provide independent confirmation of the underlying version.
The stopped original attempt is retained separately and is not a scored run.

The default client reuses `utils.model_client` and requests
`thinking={"type":"enabled"}`, temperature 0, max_tokens 32768 and a 300-second
request timeout. It stores the complete returned response, including reasoning,
finish reason and usage. Truncation and invalid JSON are separate outcomes;
neither triggers answer repair or resampling. Transport errors permit one retry
by default (`--transport-retries`).

For later local vLLM tests, add `--base-url http://localhost:8000/v1 --model NAME
--thinking omit`; put a required API key in `WM_API_KEY` or select another env
variable with `--api-key-env`. Model-specific reasoning configuration remains
the serving endpoint's responsibility when thinking is omitted. No local model
is launched by this runner.

## Checking and results

Model requests use a thread pool. Each check executes in a separate subprocess
to isolate Z3's global context. The default 120-second checker wall timeout
returns `unknown`, never `unsat`; adjust it with `--checker-timeout`.

The checker sees **only** tool names, arguments and observations. It infers an
existential initial state; it never receives the generated initial state.
Ground-truth prefixes are checked as controls. An UNSAT reference is reported
as `invalid_reference` and no prediction is requested. Other reference statuses
are preserved for coverage diagnosis. Solver policies are explicit in the
domain configuration: for TradingBot, static tools are ignored and transaction
timestamps are symbolic, matching `HARD_SOLVER.md`.

`summary.json` reports status counts, decision coverage, SAT/(SAT+UNSAT), exact
reference matches (descriptive only), token usage and groups by monitored field,
requested minimum length and requested minimum writes. Those requested groups
do not imply actual write counts; each case preserves its original `monitor`,
including lifecycle shortfalls. The main score checks the *entire* final
observation and preceding history, not just the named monitored field.

Each `cases/<id>/case.json` saves exact prompts, raw responses, predictions,
control check and predicted-trace check. Solver input/result files include
witness states or conflicts for inspection. `prompt_inputs.json` freezes the
shared prompt inputs. A prediction can be SAT without exactly matching the
reference because unobserved initial values and random timestamps can differ.
Invalid JSON, truncation, API errors, invalid solver input and solver unknown
remain separate from both SAT and UNSAT.

Writes are atomic. To resume an interrupted run, repeat the exact command with
`--resume`; completed cases are skipped, and saved predictions are reused for
unfinished cases. Resume rejects changed settings, code, prompts or dataset.
Use a fresh output directory for another stochastic repeat or changed settings.
Atomic replacements retry transient Windows file-sharing errors. A runner
source snapshot preserves the implementation used for a run; the completed
2026-10-02 experiment predates this persistence-only retry improvement. Its one
file-lock interruption was recovered from the saved prediction without a new
model request before the improvement was applied.

## Disabling thinking through this gateway

The 2026-10-02 Chat Completions run with `thinking=disabled` still received
nonempty reasoning in 89/90 responses (the gateway reported Anthropic-origin
usage for those responses). That run is retained as a failed control experiment,
not a valid no-thinking score. `--api-format anthropic` uses the native
`/messages` endpoint, moves the same system text to its `system` field, and
passes `thinking: {type: disabled}` directly. The original provider response is
retained under `response.provider_response`; normalized usage includes input
cache tokens in the input total.

To reproduce the valid no-thinking experiment, use the same endpoint-mode command
above with a fresh output directory and replace/add:

```powershell
--thinking disabled --api-format anthropic
```

The client flags `thinking_not_disabled` if a disabled request returns reasoning
text, positive reasoning-token counts, or native thinking/redacted-thinking
blocks. Such responses are not scored as no-thinking observations. Absence of
these signals verifies the exposed response, not hidden provider internals.
Because the gateway requires different wire protocols to honor this control,
the thinking/no-thinking comparison records that transport difference as well.

Later in the full-rollout experiment, the native Messages gateway exposed only
text blocks but its nested `usage.billing_usage.openai_usage` reported positive
reasoning tokens for all 635 responses. That attempted no-thinking rollout is
invalid as a control, despite the absence of visible reasoning. The client now
checks reasoning-token metadata recursively, including raw provider responses.
The earlier 90-case native endpoint experiment was re-audited and had no such
positive nested counts.

The verified rollout retry uses Chat Completions with additional provider
controls. These controls are saved in the manifest and cannot replace core
experiment settings or prompts:

```powershell
--thinking disabled --api-format openai --request-options bfcl_eval/consistency/test/request_options/deepseek_no_thinking.json
```

Seven real-prompt probes first verified zero exposed/nested reasoning. Every
subsequent response is also checked; a contradictory response is retained as
`thinking_not_disabled` instead of silently counting as a no-thinking test.
