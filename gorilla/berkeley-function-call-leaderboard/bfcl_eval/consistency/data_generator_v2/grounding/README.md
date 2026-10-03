# Generic symbolic trajectory grounding

`ground_and_replay.py` turns a symbolic trajectory into an executable case with
an LLM and a real backend. It is domain-neutral: the only required inputs are
the symbolic JSON, a tool document, and a Python backend module.

The model is called in two grounding phases. First it assigns concrete values
to all numbered placeholders. The script fills bound arguments and identity
requirements from those values. The second call fills unrestricted arguments and
creates a realistic initial state. The backend then executes the chain. A third
call audits the selected branches. If replay or the audit fails, at most
`--max-debug` repair calls are made using the requested repair priority.

The backend module must expose tool methods with the names used in the tool
document and one scenario loader named `_load_scenario`, `load_scenario`, or
`set_state`. A class can be selected explicitly with `--backend-class`; when it
is omitted, the loader discovers the only class defining one of those methods.

Example from the repository root:

```powershell
python -m bfcl_eval.consistency.data_generator_v2.grounding.ground_and_replay `
  --input-dir bfcl_eval/consistency/data_v2/trading_bot_hard/raw_tool_chain_linked_20260930 `
  --tool-doc bfcl_eval/data/multi_turn_func_doc/trading_bot_hard.json `
  --backend bfcl_eval/eval_checker/multi_turn_eval/func_source_code/trading_bot_hard.py `
  --output-dir bfcl_eval/consistency/data_v2/trading_bot_hard/grounded_20260930 `
  --max-debug 1
```

The output preserves the source path, concrete placeholder values, a clean
`tool_chain` with symbolic binding metadata, the generated `initial_state`, the
replay trace, and the branch audit. Rejected cases retain their trace and repair
diagnostics.

Model requests use a 300-second HTTP connect/read timeout by default, including
memory calls. Override it with `--request-timeout SECONDS`. This is the client's
network timeout per request, not a deadline for an entire trajectory.

To continue a stopped run, reuse its input and output paths and add `--resume`.
Accepted output files are preserved and skipped. All failed/rejected cases are
regenerated first, followed by unfinished cases. The previous manifest and
non-accepted output files are copied to `_run_history/<timestamp>/` before the
new attempt; current counters describe the latest outcomes without double
counting earlier failures. Request settings must be passed again, for example:

```powershell
# Append these to the original command, keeping the same --output-dir.
--resume --max-tokens 32768 --request-timeout 300
```

Stop any existing process for that output directory before resuming. Manifests
and case files are replaced atomically. Exceptions also produce a small
`status: failed` case file; `current_source` in the running manifest identifies
the trajectory being processed.

## Diversity memory (enabled by default)

The CLI shares one persistent pool across runs at
`bfcl_eval/consistency/diversity_memory.json`. It is not tied to a domain:
features use tool/parameter paths and initial-state paths. Backend source,
tool-document content and selected class determine the retrieval scope, so
different domains or backend versions do not share concrete value frequencies.

The pool contains:

- `records`: structured features extracted in Python from accepted cases only:
  tool/branch counts, parameter values, numeric ranges and orders of magnitude,
  container sizes, state values, and bindings. Arrays and homogeneous maps of
  records use wildcard paths. Profiles cap depth, paths and sampled children;
  reported top-value frequencies describe these stored samples, not an exact
  census of arbitrarily large states.
- `summary`: **one rolling natural-language summary for the entire pool**,
  at most 100 characters by default. No per-case model summaries are stored.

Before grounding, retrieval selects up to two similar records, one additional
recent record and one random record, plus relevant parameter frequencies. A
short model call reads this bounded context and the current chain outline and
produces an **avoidance-only guide of at most 50 characters**. The same guide
is given to placeholder assignment, state/argument generation and debug. It is
a soft preference and cannot override bindings or branch/execution constraints.
No model call is needed when the current scope has no accepted history.

After a case passes replay and the existing branch audit, Python records its
features and one short model call updates the single summary. Duplicate
concrete cases do not add records or trigger summary calls. Failed candidates
are never added. This does not import old output folders automatically.

Cost and failure bounds:

- At most **two extra short calls per accepted case**: guide and summary update.
  A rejected case uses at most the guide call; a cold start skips that call.
- Each call requests `max_tokens=2048` and thinking disabled. This is a hard
  request budget, not a target output length. The current gateway still spent
  tokens on reasoning in live tests: summary generation succeeded, but guide
  generation returned empty content even in a separate 4096-token diagnostic.
  The default remains capped at 2048; successful live guide generation with
  this gateway has not been verified. The failure is recorded as a warning
  and grounding continues without a guide.
  `--memory-max-tokens` controls this budget independently of grounding calls.
  `--memory-model` optionally selects a cheaper model supported by the same
  client; without it, memory uses the grounding model. There is no automatic
  model switch or budget increase.
  The guide's JSON input
  is capped at 8,000 characters; summary features at 3,500 characters. Backend
  source, full documentation, replay traces and the entire pool are not sent.
- Limits count Unicode characters, including punctuation, Latin letters and
  digits. Complete clauses exceeding the limit are dropped, never retried.
  Guide clauses must start with `避免`; explicit positive suggestions are
  filtered. Semantic relevance is still a model judgment, not a proof.
- Memory errors do not invalidate an executable case: warnings appear under
  `diversity` in its output. A failed summary call retains the previous summary
  and still stores the structured features. Writes use an atomic replacement
  and an exclusive lock; a busy pool is reported rather than overwritten.

Options (append to the command above):

```powershell
# Completely disable memory reads, writes, and extra model calls.
--no-memory

# Use a stricter global summary limit (the guide always remains <= 50).
--memory-summary-chars 50

# Bound each memory call, including reasoning if the provider enables it.
--memory-max-tokens 2048

# Optional: a provider-supported model dedicated to these short calls.
--memory-model YOUR_SHORT_TEXT_MODEL

# Explicitly choose a shared pool and reproducible sampling seed.
--memory-path path/to/shared_pool.json --memory-seed 42
```

Each output's `diversity` field records the guide, retrieved record IDs and the
memory update result. The manifest records whether memory was enabled and its
path/limits/seed. Programmatic callers pass a `DiversityMemory` instance to
`ground_one(memory=...)`; `memory=None` disables memory for that call.

Offline tests:

```powershell
python -m unittest bfcl_eval.consistency.data_generator_v2.grounding.test_diversity_memory -v
python -m unittest bfcl_eval.consistency.data_generator_v2.grounding.test_resume -v
```

## Agent-facing user queries

After grounding, `generate_queries.py` uses DeepSeek V4 Pro to add a natural
user request to each accepted case **in its existing JSON file**. The request
is addressed to the task-executing agent. The existing concrete `tool_chain`
is its ground-truth reference; it is never changed by query generation.

The writer receives ordered concrete calls, schemas for the tools used, and
references to objects created during the task. Future-only generated IDs are
replaced by descriptive references in its input. Replay results are supplied
only to the reference-aware reviewer, not the writer or intent-brief generator.
All model inputs omit backend code, full initial state, state snapshots and
sampling roles. The prompt forbids leaking future results or generated IDs. All
calls, including incidental calls and repeated observations, must have a
user-facing purpose. The prompt requests conversational prose rather than
numbered instructions or a call-by-call checklist.

A deterministic check also rejects literal task-generated IDs identified by
result bindings and placeholder values, unless the same value is independently
supplied as a user input. Broader answer leakage remains a semantic review task.

Each draft first receives a blind model review that sees only the query and
tool schemas and reconstructs individual requested calls with supporting query
quotes. Python counts these calls; counts must match the reference exactly.
This catches vague wording that fails to request repeated calls and avoids
relying on the model's aggregate counting. A separate review then sees the
reference and checks naturalness, coverage, arguments, timing and answer leakage. This review must quote
actual query text for every reference step. A failed draft gets bounded revisions; if none pass,
the original file remains unchanged. This is an LLM semantic review, **not**
proof that an agent will reproduce exactly one sequence. Independent lookups
may admit equivalent orderings; arbitrary sampled sequences can also be hard
to express naturally and faithfully at the same time.

```powershell
python -m bfcl_eval.consistency.data_generator_v2.grounding.generate_queries `
  --input-dir PATH_TO_GROUNDED_CASES `
  --tool-doc PATH_TO_TOOL_DOCUMENT `
  --model deepseek-v4-pro --language English --workers 4
```

Use `--trajectory PATH` instead of `--input-dir` for one case. The tool document
uses the same JSON/JSONL formats as grounding. No domain-specific tool names,
state fields or backend classes are built into this script.

- `query`: the agent-facing user message; English by default, configurable with
  `--language`, for example `--language Chinese`.
- `query_generation`: model, language, prompt version, generation timestamp,
  attempt count, original-trajectory hash and both successful model reviews.
  This metadata is not part of the agent's user message.
- Existing queries are skipped, so rerunning resumes a batch. `--overwrite`
  explicitly regenerates them. A hash mismatch flags a stale generated query.
  `--overwrite --revise-existing` uses the existing prose as a draft for minimal
  corrections; future IDs are redacted before the draft reaches the writer.
  Existing queries are also checked for literal future IDs before being skipped.
- Archived `_run_history` cases, hidden directories, manifests and unaccepted
  cases are excluded. No separate query dataset or manifest is created.
- Writes use atomic file replacement, preserve every other JSON field, and
  refuse to overwrite a case observed to change during generation. Run only
  one writer for a given case at a time.
- `--max-revisions 2` allows up to three drafts, each with up to two reviews.
  Drafts use `--temperature 0.7` for varied wording; reviews use temperature 0
  and request thinking mode for call reconstruction and temporal reasoning.
  For difficult long trajectories, `--intent-first` adds one call to extract a
  goal-grouped intent brief. The writer then uses that brief and schemas instead
  of the raw chronological trace; both reviews still check the original reference.
  `--max-tokens 16384` and `--request-timeout 300` bound each request. The model
  gateway may count reasoning against the token budget even with thinking
  disabled. `--workers` bounds concurrent cases (default 1).

Offline tests (include a non-trading domain):

```powershell
python -m unittest bfcl_eval.consistency.data_generator_v2.grounding.test_generate_queries -v
```
