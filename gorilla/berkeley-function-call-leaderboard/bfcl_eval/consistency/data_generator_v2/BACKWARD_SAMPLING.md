# Backward symbolic sampling

For the user's intended behavior, accepted decisions and remaining gaps, see
[DESIGN_CONTEXT.md](DESIGN_CONTEXT.md). This document describes the current
implementation; it does not imply that every requested extension is complete.

The goal is to test long-horizon consistency when a world model acts as the
environment for a tool-calling agent. For each monitored state field, the
trajectory obtains an early observation, applies multiple writes, and later
reads it again. The consistency test concerns all observations returned along
the sequence and whether they agree with the intervening state transitions.

This script generates only a symbolic tool-call plan. It does not fill tool
arguments, create initial state, produce observations, or execute the backend.
Structural source fixing does not prove that the selected success branches can
run together. Configured lifecycle rules filter transitions during sampling;
other guards and concrete entity keys still require downstream grounding and backend
execution, rejecting and resampling plans that cannot run.

Every eligible success branch remains a distinct sampling node. Tools are sampled
uniformly, then their eligible success branches, then eligible mutations or
return fields.
No configured writer sequence is used.

## Optional lifecycle rules

The catalog may point to a `lifecycle_rules` JSON file. `--lifecycle-rules PATH`
overrides it; `--no-lifecycle-rules` disables lifecycle filtering. The version-1
file maps target names to allowed initial
states and `(tool, success branch, from, to)` transitions. The target path locates
the lifecycle field; it does not restrict enforcement to selected monitoring
targets. Each bound path has its own reverse-time state frontier and branch
counts. Every retained mutation must fit all affected lifecycles, including
side effects. An optional positive `max_calls` limits a branch's uses per
instance, including state-preserving transitions.

The sampling strategy is unchanged from the version before numbered
placeholders. After choosing a transition backward, move only that instance's
frontier to its `from` state and increment only its branch count. Earlier
candidates must end at that frontier and stay within its call limits. Thus,
selecting `execute_order` for `{order_id_1}` allows `activate_order` earlier
for that same instance; `{order_id_2}` keeps its own frontier and counts.
The sampler retains its lifecycle capacity preference, random candidate
selection, dependency fixing, anchor reuse and length padding. It does not
sample an unordered set of calls and then sort them.

The rules are domain data: TradingBotHard deliberately samples cancellation
after activation even though its backend also permits cancellation while
Pending. Lifecycle constraints filter sampled calls; they do not insert missing
predecessors or require a complete history from creation. An unmonitored order
first used by `execute_order` can start Open. Its first transition determines
the required initial state for grounding; later transitions must connect and
respect branch call limits. Explicit `initial_states` constrain the start of a
monitored lifecycle chain, independently of this order filtering.

## Numbered placeholders

Any path placeholder denotes an identity slot. A selected monitoring chain
receives numbered identities automatically: `{target_order_id}` becomes
`{order_id_1}`, and an independent selection becomes `{order_id_2}`. Repeat
`--targets orders orders` to select both; output labels are `orders_1` and
`orders_2`. Repeating a path without placeholders is rejected.

Tool aliases are aligned by the selected path's index positions. Within a call,
the naming conventions `args.order_id`, `result.order_id`, `target_order_id`
and `order_id` identify the same slot. Other slot names remain separate: an
order ID and `order.symbol` never share an identity. Nested placeholders are
bound separately. Specifications must name distinct slots distinctly;
undocumented semantic aliases cannot be inferred from arbitrary names.

Binding substitutes the complete branch, prior conditions, return sources,
observation relations, mutation sources and reversible rules before dependency
planning. Unbound slots in a new call receive fresh identities. A dependency
fixer inherits the identity of the path it fixes. An explicit numbered path can
reuse an existing identity. No entity-type or field registry is used.

Template placeholders remain wildcards when finding candidates. Numbered
placeholders compare as identities: `{order_id_1}` does not cover
`{order_id_2}`. A parent observation still covers its children. Grounding assigns
concrete backend keys to these symbolic identities and preserves their bindings.

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

`--min-writes` is a required minimum for unconstrained targets and a best-effort
goal for targets with lifecycle rules.
`--target-min-writes NAME=N` overrides individual targets. The legacy
`--max-writes` / `--target-max-writes` flags and Python keyword names remain
accepted, but now also mean minimum counts, not upper bounds or exact counts.
One shared call counts once for every target whose monitoring interval contains
it; exceeding that target's minimum is allowed. Necessary
steps may exceed `--min-length`, but never `--max-length`. A shortfall is filled
in two stages: finite read-only support anchors from already used branches,
then unrelated read-only success branches.
The output directory must be empty.

The catalog supplies target paths, the specification directory, reader
refinements and an optional lifecycle-rule path. Its legacy `writer_sequences`
and `state_conditions` do not drive sampling. The order target's descriptive
`identity_condition` is also not evaluated. The loaded lifecycle rule's
`initial_states` does constrain the monitored order; the current rule uses
`Pending`. The history reader
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

Specifications in version 1.2 keep computational `state_source_relations` and
add `state_observation_relations` to every return field. Both use `path`,
`transform`, `recoverability`, `given`, and `reason`. Observation paths refer to
post-call state; their inverses are independent and require only declared
`given` co-values. Versions 1.1, 1.2 and 1.3 are accepted by the sampler.

Schema 1.3 requires `target_identity_sources` on every mutation (empty when
unneeded). It describes only state-derived keys in that mutation's `target`:

```json
{
  "placeholder": "order.symbol",
  "path": "$.state_before.orders['{args.order_id}'].symbol",
  "logic": "The target holding key equals the selected order's pre-call symbol."
}
```

For a selected holdings writer, binding `{order.symbol}` to `{symbol_1}` also
binds this entry's bare `placeholder` name to `symbol_1`. The entry's path uses
that call's numbered order ID. The pre-call symbol joins the selected writer's
ordinary source-fixing needs, with the same dependency-depth and reuse rules.
It may be observed by the same call or by an earlier anchor. The output attaches
`target_identity_relations` to its `write_source_requirements`: grounding and
observation checking must enforce that the path's value at this call equals the
concrete key assigned to `symbol_1`. Merely allocating the placeholder is not
evidence of that equality. No concrete equality is evaluated by this planner.

Identity entries are separate from `value_from` and `state_source_relations`;
they do not claim that observing the written value reveals the key. Their paths
must be complete pre-call state paths with values equal to the keys. Computed
keys without such a state field need a future expression schema; do not claim a
false equality. Direct argument keys and generated keys exposed in this call's
result need no pre-call identity anchor. The existing `place_order` generated-ID
alias remains supported and its post-call anchor behavior is preserved.

Only selected mutation targets schedule these extra needs. Keys occurring only
in mutation value sources, return sources, observation paths or guards do not.
All side effects are still audited, without adding identity anchors for them:
an unfixed target identity marks its containing map unknown, since the actual
modified child is not established. Old 1.1/1.2 specifications retain their old
behavior and do not provide this new guarantee; regenerate or migrate them.

For example, a constant `place_order` status retains its original provenance
and adds an exact observation of the new order:

```json
{
  "path": "$.result.status",
  "source_kind": "constant",
  "value": "Pending",
  "logic": "The new order starts Pending.",
  "state_source_relations": [],
  "state_observation_relations": [{
    "path": "$.state_after.orders['{result.order_id}'].status",
    "transform": "copy",
    "recoverability": "exact",
    "given": [],
    "reason": "The returned value equals the stored post-call status."
  }]
}
```

`generate_tool_state_specs.py` generates version 1.3 for any supplied backend
and tool schema. Its prompt and validators contain no TradingBot-specific
observation logic. `migrate_trading_bot_hard_sources.py` is the separate,
backend-reviewed migration for the existing TradingBotHard files.

A return can establish a pre-call or post-call anchor, including a field set by
that same call. `fixes[].phase`, `anchor_phase`, and `read_phase` preserve this
distinction. A post-call initial anchor excludes its own call from monitored
writes; a post-call final observation includes its call when it writes the
target. Post-state never fixes pre-state unless a declared mutation inverse
justifies it. Unchanged regions can carry knowledge between both phases.

An anchor only fixes its selected return dependencies, even if the call writes
state. Its retained mutations are audited as side effects without requiring
their entire pre-state to be reconstructible. Thus `place_order` can anchor a
new order's post-state using the returned ID, while remaining ineligible as a
selected writer whose old full orders mapping must be fixed. Grounding must
bind that returned ID to later calls on the monitored order.

- Work backward on one global frontier. Randomly choose a target's final reader,
  an outstanding target writer/anchor, or a depth-one dependency writer/anchor.
  Chains may interleave, including their initial anchors.
- A return source can fix a target only when it covers the target (same path or
  ancestor), is exact/conditional, and its co-sources are fixed. Same-branch
  returns are solved to a fixed point before adding external anchors.
- A writer selects one mutation whose target is the selected field or one of
  its descendants. An `insert` mutation targeting an ancestor also counts as a
  writer because it initializes that descendant; other ancestor mutations do
  not. Only that mutation's persistent write sources need same-call fixing or
  an earlier fixing chain. A fixing node only adds dependencies for its
  selected return field. Other returns may help fix these sources, but neither
  unrelated returns nor other mutations directly cause dependency anchors.
  This remains true when a side effect writes another monitored target.
  Lifecycle filtering does not add calls or source dependencies. All actual
  mutations and branch conditions are retained.
- Depth-one sources may have 0..`dependency-max-writes` sampled writes before
  their anchor. If there is no eligible writer, no optional write is inserted.
  At depth two and deeper, immediately use a return field with a single
  reversible state source; no writer chain is sampled at that depth.
- Shared writes count for all affected active target chains, even if one has
  already reached its minimum. Writes outside a target's anchor/read interval
  do not count for that target and do not block other chains. A writer exposing its own pre-state
  may also be the initial anchor; `roles` records both duties.
- A missing fixer, exhausted length budget, an invalid lifecycle path, or
  inability to reach minimum writes for an unconstrained target rejects the
  structural attempt. A valid constrained chain with fewer writes is accepted.
  `--attempts` controls retries.
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
  All return relations of retained calls count as observed, not only their
  selected target return. A same-branch field already observed on that instance
  is not repeated to pad length. A target-dictionary observation keeps the
  selected target's instance and is inserted beside its final observation.
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
`planning.target_min_writes` records thresholds. `write_count_mode` is
`best_effort_lifecycle` when any selected target has a lifecycle rule and
`at_least_minimum` otherwise. In a mixed batch, unconstrained targets still
require their minimum; the mode does not relax them. `lifecycle_targets` lists
the constrained names. `planning.writer_candidates_by_target` also lists graph
candidates and missing reversible write-source fixers. No arguments,
observations, initial states, model calls
or backend executions are generated here.

`placeholder_bindings` maps each step's template aliases to numbered identities.
All retained step paths use those identities. `lifecycle_transitions` is keyed
by the bound lifecycle field path, and `planning.lifecycle_instances` records
each instance's rule, allowed initial states and chronological transition steps.
The latter includes instances that are not monitoring targets. For unmonitored
instances, the first sampled transition supplies the required initial state;
no predecessor calls are added to establish it.

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

The script checks structural source fixing, write counts, configured lifecycles
and the fixed disabled long-context mode. General branch-condition satisfiability
requires downstream grounding. With the default order rule active, repeated
activation/execution and execution followed by cancellation of the same bound
order are filtered, including when `orders` is not selected. Disabling the rule
removes those protections. Placeholder allocation does not infer lifecycle
rules or prove cross-target guard feasibility. The downstream model must instantiate
arguments/state, verify the backend and reject/resample infeasible chains.
Success of this script is not proof of backend executability.

Two important consequences of the existing TradingBotHard specifications:

- Funding/withdrawal writes both balance and history. Additional execution can
  raise balance writes above its requested minimum; this no longer causes
  rejection. Existing order lifecycles still limit executable status writes,
  regardless of the structural minimum; their guards require grounding.
- `place_order` is a selected-writer graph candidate, but requires the complete pre-write orders
  mapping and order counter. There is no reversible whole-mapping reader in the
  supplied success-branch specifications, so the strict source-fixing rule rejects
  that role. It is eligible as a post-call anchor for the new order's observed
  fields and counter. Changing the random seed does not remove the missing
  pre-write dependency when selecting it as a writer.
- A final observation may be a mutating call. `execute_order` and `cancel_order`
  can observe their resulting status. `place_order` is generically a post-state
  observation candidate, but the default order rule excludes it as the final
  observation: its predecessor is `Absent`, which cannot connect to the required
  monitored initial state `Pending`. With lifecycle rules disabled it can be
  selected as a final post-state observation.

Verify with:

```powershell
python -m unittest tests.test_backward_trajectory_sampling -v
```
