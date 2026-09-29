# Backward symbolic sampling

The goal is to test long-horizon consistency when a world model acts as the
environment for a tool-calling agent. For each monitored state field, the
trajectory obtains an early observation, applies multiple writes, and later
reads it again. The consistency test concerns all observations returned along
the sequence and whether they agree with the intervening state transitions.

This script generates only a symbolic tool-call plan. It does not fill tool
arguments, create initial state, produce observations, or execute the backend.
Structural source fixing does not prove that the selected success branches can
run together: entity bindings, guards, and especially order lifecycles may be
incompatible. Downstream grounding must instantiate and execute each plan,
rejecting and resampling those that cannot run.

Every eligible success branch remains a distinct sampling node. Tools are sampled
uniformly, then their eligible success branches, then eligible mutations or
return fields.
No configured writer sequence is used.

## Optional lifecycle rules

The catalog may point to a `lifecycle_rules` JSON file. `--lifecycle-rules PATH`
overrides it; `--no-lifecycle-rules` disables it and restores the original
unconstrained sampler. The version-1 file maps target names to allowed initial
states and `(tool, success branch, from, to)` transitions. Branches that write a
constrained target are eligible only when selected for that target and when the
transition fits the current reverse-time state frontier. Other targets keep the
original minimum-write requirement. An optional positive `max_calls` limits a
branch's uses on the target entity, including state-preserving transitions. A
configured branch that writes several constrained
targets must have a valid transition for each one.

The rules constrain one symbolic entity per target chain. They are domain data,
not backend facts: TradingBotHard deliberately samples cancellation after
activation even though its backend also permits cancellation while Pending.
The downstream grounder must bind all writes in a target chain to the same
entity, and instantiate distinct entities for any other calls it adds later.

For constrained targets, `min-writes` is a goal rather than an acceptance gate.
The sampler prefers candidates that preserve the longest reachable lifecycle
path, but returns a shorter valid chain when the available transitions are
exhausted. `planning.target_chains` records `min_write_count`, `write_count`,
`write_shortfall`, and `shortfall_reason`. Unconstrained targets retain the
existing minimum-write requirement. Branch guards and backend feasibility still
require downstream grounding and execution.

Run from `gorilla/berkeley-function-call-leaderboard`:

```powershell
python -m bfcl_eval.consistency.data_generator_v2.generate_backward_trajectories `
  --targets balance transaction_history orders `
  --min-writes 3 `
  --dependency-max-writes 1 --min-length 20 --max-length 60 `
  --count 10 --seed 42 --output-dir bfcl_eval/consistency/trajectories_v2/backward_demo
```

`--min-writes` is a required minimum write-call count for each selected target.
`--target-min-writes NAME=N` overrides individual targets. The legacy
`--max-writes` / `--target-max-writes` flags and Python keyword names remain
accepted, but now also mean minimum counts, not upper bounds or exact counts.
One shared call counts once for every target whose monitoring interval contains
it; exceeding that target's minimum is allowed. Necessary
steps may exceed `--min-length`, but never `--max-length`. A shortfall is filled
in two stages: finite read-only support anchors from already used branches,
then unrelated read-only success branches.
The output directory must be empty.

The existing catalog supplies target paths, the specification directory, and
reader refinements only. Its writer sequences, global state assumptions and
initial Pending-order requirement are deliberately ignored. The history reader
refinement is retained: omit both date bounds and require `long_context=false`.
All branch and earlier-branch conditions remain in the output for grounding.

The new sampler hardcodes `long_context=false`, independently of the catalog.
Success branches requiring `long_context=true` and branches mutating the mode
are excluded from the node catalog, covering anchors, writers, readers and
distractors. Standard `otherwise` branches remain available. Every selected
step includes `$.state_before.long_context == false` in `additional_conditions`;
cases and the manifest record this in `state_conditions`. Grounding must pass
`long_context=False` to backend scenario loading, not just add it to the scenario
dictionary. Original enabled-mode guards may still appear in the list of earlier
conditions to avoid; those branches are never selected.

## Construction

- Work backward on one global frontier. Randomly choose a target's final reader,
  an outstanding target writer/anchor, or a depth-one dependency writer/anchor.
  Chains may interleave, including their initial anchors.
- A return source can fix a target only when it covers the target (same path or
  ancestor), is exact/conditional, and its co-sources are fixed. Same-branch
  returns are solved to a fixed point before adding external anchors.
- A writer selects one mutation whose target covers the selected field. Only
  that mutation's persistent write sources need same-call fixing or an earlier
  fixing chain. A fixing node only adds dependencies for its selected return
  field. Other returns may help fix these sources, but neither unrelated
  returns nor other mutations cause additional anchors. This remains true even
  when a side effect writes another monitored target. All actual mutations and
  branch conditions are retained for grounding.
- Depth-one sources may have 0..`dependency-max-writes` sampled writes before
  their anchor. If there is no eligible writer, no optional write is inserted.
  At depth two and deeper, immediately use a return field with a single
  reversible state source; no writer chain is sampled at that depth.
- Shared writes count for all affected active target chains, even if one has
  already reached its minimum. Writes outside a target's anchor/read interval
  do not count for that target and do not block other chains. A writer exposing its own pre-state
  may also be the initial anchor; `roles` records both duties.
- A missing fixer, exhausted length budget or inability to reach minimum write counts
  rejects the structural attempt. `--attempts` controls retries.
- Before padding, run a chronological knowledge pass. Earlier observations stay
  fixed through writes whose sources are fixed; forward propagation does not
  require mutations to be invertible. Source-fixed child writes also preserve a
  known parent's full symbolic value. Remove read-only intermediate anchors
  when their fields are already fixed, reusing the earlier observation for
  dependency and initial-target duties. Preserve all final readers. Recompute
  monitoring intervals and write counts, allowing the widened intervals to
  include additional writes. Check final maximum length after anchor reuse,
  then fill only the remaining minimum-length shortfall with distractors.
- If the core chains are shorter than `min-length`, first sample each eligible
  same-branch alternative return field at most once. Also allow direct reversible
  source observations in the selected target dictionaries. These support anchors
  are read-only, have no selected mutation, and do not change target write counts.
  The same `(tool, branch)` may be sampled provisionally for another return
  field, but the normal forward anchor-reuse pass runs again after support
  anchors are inserted. Because one invocation returns all declared fields,
  redundant observations are removed before unrelated distractors are added.
  `core_length`, `support_anchor_count`, `necessary_length`, and
  `distractor_count` report these stages.
- Apply every retained side effect to symbolic knowledge. If its sources are
  already known or fixed by the same call, propagate its value. Otherwise mark
  its target unknown without adding an anchor. An unknown child makes its full
  parent unknown, but leaves unaffected siblings known. If a later selected
  source needs an unknown value, it must have its own legitimate fixing point;
  the forward audit rejects attempts that cannot prove this. Post-write returns
  can restore knowledge for later calls but cannot stand in for pre-write values.

## Output and limitations

Each step preserves branch conditions, all earlier branches to avoid, role(s),
dependency depth, return specifications, mutation specifications and write-source
fixing requirements. `planning.target_chains` indexes initial anchors, target
writes and final reads, including required minimum and actual write counts.
`planning.target_min_writes` records thresholds and `write_count_mode` is
`at_least_minimum`. `planning.writer_candidates_by_target` also lists graph
candidates and missing reversible write-source fixers. No arguments,
observations, initial states, model calls
or backend executions are generated here.

`planning.redundant_anchors_removed` records removed provisional anchors and
their reused fixing steps. Propagated write-source evidence records its
`anchor_step` and intervening `mutation_steps`; grounding must preserve the
entity bindings along that entire chain, including reused dependency anchors.
Additional `observation_steps` record later observations used to restore parts
of a known parent. `anchor_phase` distinguishes pre-write and post-write fixing;
an initial post-write anchor does not count its own call as an intervening write.

`selected_mutation_index` is a zero-based index into the step's complete
`mutations` list. Only its state sources appear in `write_source_requirements`;
reader/anchor steps have no selected mutation. Selected return fields and their
missing co-sources are recorded in `selected_return_field` and `fix_dependencies`.
`dependency_scope` in planning and the manifest is
`selected_mutation_and_return_field`.

`mutation_knowledge` reports source fixing and post-state knowledge for every
actual mutation. `planning.unfixed_side_effects` lists effects whose sources
were not fixed; it is distinct from unresolved selected dependencies. Actual
`write_steps` and `write_count` still include shared side effects, while
`source_fixed_write_steps` and `fully_source_fixed` report whether those writes'
state inputs were fixed. Minimum counts constrain actual writes, not this subset.

State knowledge and partial invalidation live in `backward_state_knowledge.py`;
branch sampling and dependency selection remain in `generate_backward_trajectories.py`.

The script checks structural source fixing, write counts and the fixed disabled
long-context mode, not general branch-condition satisfiability. In particular,
repeated activation/execution of the same symbolic
order can be structurally sampled but fail grounding. The downstream model must
instantiate arguments/state, verify the backend and reject/resample infeasible
chains. Success of this script is not proof of backend executability.

Two important consequences of the existing TradingBotHard specifications:

- Funding/withdrawal writes both balance and history. Additional execution can
  raise balance writes above its requested minimum; this no longer causes
  rejection. Existing order lifecycles still limit executable status writes,
  regardless of the structural minimum; their guards require grounding.
- `place_order` is a graph candidate, but requires the complete pre-write orders
  mapping and order counter. There is no reversible whole-mapping reader in the
  supplied success-branch specifications, so the strict source-fixing rule rejects
  it. Changing the random seed does not remove that missing dependency.

Verify with:

```powershell
python -m unittest tests.test_backward_trajectory_sampling -v
```
