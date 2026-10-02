# Gap-bounded context workflow

The study supports two context modes. `full` is the clean baseline: after a
successful retrieval, the next Policy request can see the complete source
memory again, with each canonical source unit rendered at most once in that
request. `gap_bounded` is a diagnostic mode: retrieval is divided into phases.

Each Policy call returns one retrieval action and an information-gap
assessment:

```json
{
  "assessment": {
    "resolved_gaps": ["identify the town where the festival was held — Mary Town"],
    "missing_information": ["determine Mary Town's main industry"]
  }
}
```

`resolved_gaps` is cumulative state and `missing_information` is replaced on
each call. The assessment is a model working judgment, not evidence. The
runtime keeps the raw response in the trajectory and also keeps a normalized
cumulative assessment in episode state so an omitted older resolved item does
not reappear as a new gap.

In `gap_bounded` mode, a phase changes only after a valid retrieval executes,
adds new source text, and the assessment adds a resolved gap. The source window
for the next retrieval call is then cleared, while the original question,
assessment, action history, budgets, entity handles, reference registry, and
complete source artifact remain available to the runtime. Entity references
remain cumulative when they are still navigable; hidden or exhausted source
text is not silently treated as current evidence.

When `missing_information` becomes empty after a successful retrieval, the
runtime enters a separate answer stage. It sends all accumulated source memory,
closes retrieval tools, removes the assessment requirement, and exposes only
`finish(answer, evidence_refs)`. If the retrieval budget is exhausted first, a
single finish-only budget finalization may answer with the unresolved gaps
preserved in the artifact.

The phase index, source keys, transition reason, answer-stage flag, raw and
normalized assessments, and context audit are written to the episode artifact.
The renderer, protocol, and artifact contract hashes change when this workflow
changes, so an older run cannot be resumed under a different context contract.
