"""Structured contracts used by the Options runtime."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_rag.agent.models import AgentModel, AgentAction, Assessment


class OptionSelectorDecision(AgentModel):
    """The selector chooses only a sub-goal, never a primitive action."""

    option_id: str = Field(min_length=1)


class OptionPolicyDecision(AgentModel):
    """One selected option's observation-conditioned decision.

    COMPLETE and BLOCKED deliberately carry no action: the controller returns
    to the selector instead of executing a stale primitive action.  CONTINUE
    requires one primitive action.  This keeps option lifecycle transitions
    explicit while retaining one global Assessment per policy turn.
    """

    model_config = ConfigDict(extra="forbid")

    assessment: Assessment
    option_status: Literal["CONTINUE", "COMPLETE", "BLOCKED"]
    action: AgentAction | None = None

    @model_validator(mode="after")
    def status_action_consistency(self) -> "OptionPolicyDecision":
        if self.option_status == "CONTINUE" and self.action is None:
            raise ValueError("CONTINUE requires one primitive action")
        if self.option_status in {"COMPLETE", "BLOCKED"} and self.action is not None:
            raise ValueError(f"{self.option_status} must not include an action")
        return self


class OptionEvent(AgentModel):
    """Separate lifecycle audit; it is not mixed into primitive observations."""

    step: int = Field(ge=0)
    policy_attempt: int = Field(ge=0)
    event: Literal[
        "selected", "continue", "continued", "complete", "completed",
        "blocked", "interrupted", "fallback",
    ]
    option_id: str
    status: Literal["CONTINUE", "COMPLETE", "BLOCKED"] | None = None
    option_status: Literal["CONTINUE", "COMPLETE", "BLOCKED"] | None = None
    action_type: str | None = None
    primitive_action: dict[str, Any] | None = None
    outcome: dict[str, Any] | str | None = None
    reason: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)


class OptionTrace(AgentModel):
    """Compact audit metadata written beside the existing primitive trajectory."""

    catalog_sha256: str
    selector_calls: int = Field(default=0, ge=0)
    policy_calls: int = Field(default=0, ge=0)
    events: list[OptionEvent] = Field(default_factory=list)
