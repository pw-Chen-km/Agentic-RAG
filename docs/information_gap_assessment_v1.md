# Information-Gap Assessment v1

This document defines the assessment contract used by the information-gap retrieval
workflow. It is a model-generated aid for choosing the next operation. It is not source
text, an evidence citation, or a replacement for the visible source record.

## Contract

When assessment is enabled, the model provides an assessment together with its single
tool call in the same decision:

```json
{
  "assessment": {
    "missing_information": []
  },
  "action": {}
}
```

`missing_information` contains at most three non-empty strings. Each item describes a
specific fact or connection that is still needed to answer the question. An empty array
means that the model has no clearly identified information gap; it does not prove that
the answer is correct or that the visible evidence is sufficient.

The assessment is regenerated from the current observation on every decision. A previous
assessment is only the model's earlier judgment and must be reconsidered after new source
text or an operation result is shown. If a finish decision still lacks information, the
assessment may retain that gap.

When assessment is disabled, the tool schema does not contain an assessment field and the
decision is recorded as `not_requested`. A missing or unparseable assessment when it is
required is recorded as `unavailable`; it is not converted to an empty array.

## Evidence and action boundaries

Assessment text never creates evidence, references, entities, or source spans. Evaluators
use only source text that was actually visible to the policy. The action is still chosen
from the current tool registry; the assessment does not prescribe a tool, query, order,
or retrieval route.

The shared skill asks the model to identify a concrete information gap and compare the
scope and returned text unit of each available operation. It does not require a global
search, entity following, sentence result, passage result, or any other fixed sequence.

## Validation and versioning

The assessment schema rejects the historical `supported_facts` field and any unknown
assessment fields. `missing_information` must be an array of at most three strings. This
contract is identified as `information-gap-v1` in run manifests and artifact metadata.
The skill, decision schema, tool schema, renderer, and protocol hashes are part of the
resume manifest. A run using an older assessment contract or skill cannot be resumed as
an information-gap-v1 run.

## Diagnostics

Artifacts may record the assessment value, decision turn, parse status, and the action
selected in that turn. Analysis can report the number of missing-information items,
whether a gap is specific enough for review, changes after an operation or rejection,
and parse failures. These are workflow diagnostics and are not semantic answer or
evidence metrics.
