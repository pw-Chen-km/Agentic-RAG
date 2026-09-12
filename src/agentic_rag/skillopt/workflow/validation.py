"""Pure validation gate; execution and metrics collection stay injectable."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping

@dataclass(frozen=True)
class ValidationMetrics:
    accuracy: float
    target_tokens: int
    policy_calls: int
    evaluated_questions: int
    def __post_init__(self):
        if not 0 <= self.accuracy <= 1 or self.target_tokens < 0 or self.policy_calls < 0 or self.evaluated_questions <= 0: raise ValueError('invalid validation metrics')

@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str
    accuracy_delta: float
    token_gain: float | None
    call_gain: float | None

def evaluate_candidate(current: ValidationMetrics, candidate: ValidationMetrics, *, tolerance_pp: float=.02, min_efficiency_gain: float=.05) -> GateDecision:
    if current.evaluated_questions != candidate.evaluated_questions: return GateDecision(False,'incomplete_or_mismatched_evaluation',candidate.accuracy-current.accuracy,None,None)
    if current.target_tokens <= 0 or current.policy_calls <= 0 or candidate.target_tokens <= 0 or candidate.policy_calls <= 0: return GateDecision(False,'missing_nonzero_cost_measurement',candidate.accuracy-current.accuracy,None,None)
    tg=None if current.target_tokens==0 else (current.target_tokens-candidate.target_tokens)/current.target_tokens
    cg=None if current.policy_calls==0 else (current.policy_calls-candidate.policy_calls)/current.policy_calls
    if tg is None or cg is None: return GateDecision(False,'missing_nonzero_cost_baseline',candidate.accuracy-current.accuracy,tg,cg)
    da=candidate.accuracy-current.accuracy
    accepted=(da>0 and tg>=-.05 and cg>=-.05) or (da>=-tolerance_pp and (tg>=min_efficiency_gain or cg>=min_efficiency_gain) and tg>=-.05 and cg>=-.05)
    return GateDecision(accepted,'accepted' if accepted else 'threshold_not_met',da,tg,cg)
