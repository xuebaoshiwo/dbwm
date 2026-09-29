# Design context and session handoff

Status recorded on 2026-09-29. Read this together with
[BACKWARD_SAMPLING.md](BACKWARD_SAMPLING.md) and the current implementation before
changing behavior. Keep this file current when accepted decisions or known gaps
change. A future session should treat the open issues below as unfinished work,
not as an instruction to implement them without a task from the user.

## Research goal

Generate long tool-call trajectories for testing a world model that simulates
an environment. Each monitored state field has an early observation that fixes
its value, intervening writes, and a later observation. The evaluation concerns
consistency across the returned observations and state transitions throughout
the trajectory, including interleaved chains and shared effects.

The immediate problem was a high rate of non-executable symbolic plans. Purely
random writer selection often violates an entity's lifecycle, repeats one-time
operations, or chooses mutually exclusive terminal operations on the same
entity. TradingBotHard orders exposed this problem; other domains must be able
to supply their own rules.

## Accepted requirements and decisions

- Preserve the existing global backward frontier, random task interleaving,
  and tool-then-success-branch-then-mutation/return sampling. Do not substitute
  fixed tool sequences. Candidate filtering and lifecycle capacity preference
  refine this strategy; randomness is applied to the remaining candidates.
- The user reaffirmed the pre-placeholder reverse-time lifecycle strategy.
  Preserve its candidate restriction, capacity preference, dependency fixing,
  anchor reuse, padding and write-count policy. Numbered placeholders change
  identity binding and lifecycle scope: each instance has its own frontier and
  call counts. Do not replace the sampler with independent random calls followed
  by rejection or post-sampling sorting.
- Keep domain lifecycle rules in readable external JSON. General parsing,
  transition validation and dependency analysis belong in reusable code.
- Preserve repeatable operations such as funding and withdrawal. A rule may
  restrict branch uses with `max_calls`; no default one-call limit applies to
  every writing tool.
- The user explicitly asked that an exhausted valid chain return normally even
  when it cannot reach the requested write count. Do not reject a valid
  lifecycle merely because its maximum capacity is below that count. The
  current implementation applies this relaxation only to constrained targets;
  the broader shortfall requirement remains incomplete (see known gaps).
- A return that reversibly determines post-call state is an observation and
  can anchor that state. Its computational provenance alone is insufficient to
  express this, so retain source relations and add post-state relations.
- Keep observation timing explicit. A post-call value cannot silently fix the
  pre-call value of a changed field. Count writes inside the observation interval.
- Keep selected dependency scope narrow: fix the chosen mutation's state
  sources or the chosen return's co-values. Retain and audit every actual side
  effect without recursively expanding all unrelated effects into anchors.
- The user requested clean, scoped changes and an easy rollback. Preserve
  existing work, generated batches and the snapshots listed below.
- The user rejected the entity-type registry implementation. Restore the
  preceding schema-1.2 sampler, then allocate identities from placeholders.
  Bind a chain's anchors, writes, sources and observations to the same numbered
  placeholder. Independent chains get different numbers. No entity or field
  registry is needed. Apply configured lifecycles to every affected bound
  instance, including dependencies and unmonitored effects.
- Lifecycle rules filter the sampled sequence. They must not automatically
  insert earlier lifecycle calls. A previously unused order can already be
  Open at its first execution; preserve that requirement for grounding.
  Explicit monitoring initial states and requested write counts remain
  separate constraints on the monitored chain.

The initial discussion also considered having a downstream model reorder,
delete or deduplicate illegal calls. The implemented solution adds sampling
constraints and leaves concrete grounding/execution downstream. No downstream
model repair pipeline has been implemented here. Any future repair that changes
calls must recompute dependencies, observation intervals and write counts.

## Current implementation

Schema 1.3 adds `target_identity_sources` to every mutation. Entries contain
`placeholder`, a complete pre-call state `path`, and `logic`; they assert that
the state value equals the dynamic target key. The TradingBotHard migration
adds this relation to the three `execute_order` holdings mutations and empty
lists elsewhere. No identity dependencies are added for keys occurring only in
value sources, return sources, observation relations or branch guards.

Only selected mutations add target identity paths to the existing dependency
frontier. Binding updates both the entry's bare placeholder name and its path.
The forward audit includes these requirements; unselected effects do not add
anchors and unresolved addresses conservatively invalidate their containing
map. Output `target_identity_relations` preserves the equality obligation for
grounding and observation checking at the write's pre-call state. The sampler
does not compute concrete keys or check numeric/string equality. Direct args
and returned generated IDs keep their existing behavior. Versions 1.1/1.2 are
still accepted with their original, weaker identity guarantees. The new generic
prompt and validator require 1.3, but have not been verified with a live model.

The active sampler is `generate_backward_trajectories.py`; domain dependency
JSON is generated separately by `generate_tool_state_specs.py`.

1. Load targets, schema-1.1/1.2/1.3 specifications and reader refinements; load any
   external lifecycle rules. Use success branches as distinct nodes.
2. Allocate target placeholders and bind candidate calls before dependency
   analysis. Repeated selections produce independent monitoring chains.
3. Choose final observations and work backward on one global frontier, mixing
   outstanding target writers/anchors and source dependency chains. Prefer
   the longest reachable monitored lifecycle,
   capped by the requested count. Lifecycle state and counts are per bound path.
4. Fix selected sources through reversible same-call returns or earlier anchors.
   Depth-one dependencies can have optional writes; depth two and beyond use
   immediate reversible singleton-source observations without writer chains.
5. Reverse the plan, propagate known state forward through source-fixed writes,
   invalidate unknown side effects and reuse redundant read-only anchors.
   Preserve mutating calls and final observations. Recompute target intervals.
6. Fill a minimum-length shortfall with read-only support observations, reuse
   anchors again, then add unrelated read-only distractors. Enforce maximum
   length and emit the symbolic plan plus grounding obligations.

Schema 1.2 adds `state_observation_relations` to every return field, with the
same relation keys as `state_source_relations`: `path`, `transform`,
`recoverability`, `given`, `reason`. Observation paths use `$.state_after` and
require their explicitly declared co-values. Original source provenance stays
intact. `exact` and `conditional` observations can establish anchors.

All existing TradingBotHard tool specifications were migrated to schema 1.3.
The generic generator's prompt and validators support this schema for supplied
backends; the separate migration script contains reviewed TradingBotHard facts.

## Order example and observation timing

The order target monitors one numbered order's status per selection. Repeat
`--targets orders orders` to monitor two independent orders. Its lifecycle rule
uses initial state `Pending` and these transitions:

```text
Absent --place_order--> Pending --activate_order--> Open
                                                   |--execute_order--> Completed
                                                   `--cancel_order---> Cancelled
```

Each configured branch has `max_calls=1`. Terminal states have no outgoing
transitions, so execution and cancellation cannot both apply to that order.
Cancellation after activation is the chosen sampling policy; the backend also
permits Pending cancellation.

`place_order` returns an order ID and Pending status, so it can anchor the new
order after creation. A symbolic example is:

```text
place_order [anchor after] -> activate_order [write]
    -> cancel_order [write] -> get_order_details [final observation]
```

For `min-writes=3`, this monitored interval has two writes and a reported
shortfall of one. Creation lies before the post-call anchor and does not count.
The completed-order variant can use `execute_order` as both a counted write
and its final post-call observation.

As a selected writer, `place_order` still requires the complete old orders
mapping and counter; the supplied specifications have no reversible reader
for that whole mapping. Its anchor role does not require reconstructing those
unrelated pre-state inputs. The downstream grounder must bind the returned ID
to later operations on that same order.

The generic sampler permits mutating final observations. The default order
rule excludes `place_order` as the final order observation because `Absent`
cannot connect to the monitored initial state `Pending`. Disabling the rule
allows that generic final-observation candidate.

## Extensibility and known gaps

Target registration is data-driven: add a named `path` under the catalog's
`targets`, then include its name in `--targets`. The CLI default still selects
`balance transaction_history orders`. The catalog's legacy `writer_sequences`
are ignored. Placeholders receive symbolic identities automatically; concrete
backend keys are supplied downstream. Alias binding uses selected path index
positions and the conventional `args.`, `result.` and `target_` prefixes.
Different slots must have distinct names in the specification.

The catalog includes these targets without a Python entity registry:

```json
{
  "watch_list": {"path": "$.state_before.watch_list"},
  "holdings": {"path": "$.state_before.holdings['{symbol}']"}
}
```

- `watch_list` has reader, add and remove specifications. Effective
  writes still require valid membership conditions when arguments are grounded.
- A whole-map `get_holdings` observation can fix one symbol's holding. Its
  writer is `execute_order`; repeated execution on the same order is invalid,
  so repeated holding changes require distinct orders. Parent observations
  cover children, but a child mutation does not count as a whole-parent write.
- Lifecycle state and call counts track every affected bound path independently.
  Holding writers can use fresh orders even when other orders are monitored.
  Their initial state is inferred from the first sampled transition; lifecycle
  filtering does not add activation or other missing predecessor calls.
- Without a lifecycle rule, a target still requires its minimum write count.
  Such a shortfall can retry and eventually raise `SamplingError`. The user's
  broader request to return any exhausted valid chain is not fully implemented.
- Rules apply even when their target is omitted from monitoring. A new domain
  still supplies its transition rules; placeholder syntax alone does not
  describe legal lifecycle states or arbitrary semantic aliases.
- `max_calls` is per `(tool, branch)` on the target entity. There is no separate
  tool-wide or named mutually exclusive group counter; the order state machine
  expresses exclusivity through its terminal states.
- The sampler currently hardcodes `long_context=false` and excludes enabled-mode
  branches and mode writers. This existing environment assumption must be
  considered when adapting the sampler to a different domain.
- Guards, authentication, balances, inventory, timestamps and concrete identity
  feasibility are grounding obligations. Symbolic success does not prove that
  the backend can execute the whole plan. No arguments, initial state, model
  observations or backend executions are produced by the sampler itself.

## Verification and rollback

Run from `gorilla/berkeley-function-call-leaderboard`:

```powershell
python -m unittest tests.test_backward_trajectory_sampling tests.test_generate_tool_state_specs tests.test_placeholder_trajectory_sampling tests.test_target_identity_sampling -q
```

The current implementation verification passed 93
tests. The local suites cover observation phases, dependency fixing, shared
writes, lifecycle rules, old schemas and generic non-trading post-state anchors.
Placeholder tests cover independent non-trading instances, nested slots,
aliases, isolated knowledge and lifecycles on unmonitored effects.
The lifecycle-filter correction adds tests for direct execution from Open and
rejection of repeated or mutually exclusive terminal calls on the same instance.
Examples under `D:/dbwm/placeholder_filter_smoke_20260928/` use the corrected
filter-only behavior. Earlier `placeholder_smoke_20260928` examples predate
this correction and may contain automatically inserted activation calls.
Existing TradingBotHard specifications were migrated locally; the generic
generator's schema/prompt validation was tested without a real model API call.

Local snapshots outside this source directory:

- `D:/dbwm/publish_dbwm_20260928/gorilla/berkeley-function-call-leaderboard/`:
  exact schema-1.2 baseline used to roll back the rejected entity-type version.
- `D:/dbwm/rollback_entity_types_20260928/`: the rejected implementation saved
  before restoration; do not restore it as the active design.
- `D:/dbwm/rollback_data_generator_v2_20260928/`: files before lifecycle rules,
  with restoration instructions in `ROLLBACK.md`. `--no-lifecycle-rules` disables
  filtering but does not undo later schema and post-state changes.
- `D:/dbwm/rollback_post_state_20260928/`: files before post-state observations,
  retaining the earlier lifecycle implementation; see its `ROLLBACK.md`.
- `D:/dbwm/rollback_post_state_20260928/verification/`: ten local sample plans
  from the post-state implementation. Files `backward_42_0003.json` and
  `backward_42_0007.json` illustrate creation anchors and terminal observations.

These snapshots are local to this workspace and do not travel with a repository
checkout. Review newer changes before restoration. The consistency directory
is currently untracked in Git, so a normal tracked-file rollback is insufficient.
