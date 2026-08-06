"""Query-time agent configuration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import PydanticCustomError

from agentic_rag.agent.models import (
    DEFAULT_ENABLED_EXPANSIONS,
    ContextMode,
    ExpansionKind,
)


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PolicyConfig(ConfigModel):
    """OpenAI policy configuration.

    The historical class name remains the OpenAI configuration so existing
    ``PolicyConfig()`` callers keep working unchanged.
    """

    provider: Literal["openai"] = "openai"
    model: str = Field(default="gpt-5.6-terra", min_length=1)
    max_retries: int = Field(default=2, ge=0, le=10)


class AnswerConfig(ConfigModel):
    """OpenAI answer-generator configuration (backward compatible)."""

    provider: Literal["openai"] = "openai"
    model: str = Field(default="gpt-5.6-terra", min_length=1)
    max_retries: int = Field(default=2, ge=0, le=10)


OllamaThink = bool | Literal["low", "medium", "high"] | None


class OllamaConfig(ConfigModel):
    """Fields shared by Ollama policy and answer generation."""

    provider: Literal["ollama"] = "ollama"
    model: str = Field(min_length=1)
    host: str = Field(default="http://localhost:11434", min_length=1)
    temperature: float = Field(default=0.0, ge=0.0)
    think: OllamaThink = False
    timeout_seconds: float = Field(default=300.0, gt=0.0)
    keep_alive: str | int | float | None = None
    max_retries: int = Field(default=2, ge=0, le=10)
    num_ctx: int = Field(default=32_768, ge=2_048)

    @field_validator("model", "host")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("host")
    @classmethod
    def validate_ollama_host(cls, value: str) -> str:
        invalid_message = (
            "Ollama host must be a valid HTTP(S) URL with a hostname"
        )
        try:
            parsed = urlparse(value)
            hostname = (parsed.hostname or "").casefold().rstrip(".")
            # Accessing ``port`` performs urllib's numeric/range validation.
            parsed.port
        except ValueError as exc:
            raise PydanticCustomError(
                "ollama_host_invalid",
                invalid_message,
            ) from exc
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or not hostname
            or any(character.isspace() for character in value)
            or parsed.username is not None
            or parsed.password is not None
            or bool(parsed.query)
            or bool(parsed.fragment)
        ):
            raise PydanticCustomError(
                "ollama_host_invalid",
                invalid_message,
            )
        if hostname == "ollama.com" or hostname.endswith(".ollama.com"):
            raise PydanticCustomError(
                "ollama_cloud_unsupported",
                "Ollama Cloud does not support the structured output "
                "required by the Agentic RAG provider",
            )
        return value.rstrip("/")

    @field_validator("keep_alive", mode="before")
    @classmethod
    def validate_keep_alive(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("keep_alive must not be a boolean")
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError("keep_alive must be finite")
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("keep_alive must not be blank")
        return value


class OllamaPolicyConfig(OllamaConfig):
    """Ollama chat policy configuration."""


class OllamaAnswerConfig(OllamaConfig):
    """Ollama chat answer-generator configuration."""


PolicyProviderConfig = Annotated[
    PolicyConfig | OllamaPolicyConfig,
    Field(discriminator="provider"),
]
AnswerProviderConfig = Annotated[
    AnswerConfig | OllamaAnswerConfig,
    Field(discriminator="provider"),
]


class AgentConfig(ConfigModel):
    workflow_mode: Literal[
        "legacy",
        "single_agent_v2",
        "single_agent_v2_compact",
        "single_agent_v2_2",
        "single_agent_v3",
        "single_agent_v3_action_catalog",
        "single_agent_v3_typed_refs",
        "single_agent_v3_2",
    ] = "legacy"
    max_steps: int = Field(default=10, ge=1)
    max_policy_attempts: int | None = Field(default=None, ge=1)
    max_consecutive_invalid_attempts: int = Field(default=2, ge=1, le=10)
    max_retrieved_tokens: int = Field(default=12_000, ge=1)
    v3_include_last_assessment: bool = True
    v3_include_latest_event: bool = True
    v3_include_attempted_actions: bool = True
    v3_include_budget: bool = True
    context_mode: ContextMode = ContextMode.COMPACT_EVIDENCE
    enabled_expansions: tuple[ExpansionKind, ...] = DEFAULT_ENABLED_EXPANSIONS
    policy: PolicyProviderConfig = Field(default_factory=PolicyConfig)
    answer: AnswerProviderConfig = Field(default_factory=AnswerConfig)

    @field_validator("policy", "answer", mode="before")
    @classmethod
    def omitted_provider_remains_openai(
        cls, value: Any
    ) -> Any:
        """Preserve legacy mappings that predate provider selection."""

        if isinstance(value, dict) and "provider" not in value:
            return {"provider": "openai", **value}
        return value

    @field_validator("enabled_expansions")
    @classmethod
    def expansions_must_be_unique(
        cls, value: tuple[ExpansionKind, ...]
    ) -> tuple[ExpansionKind, ...]:
        if len(value) != len(set(value)):
            raise ValueError("enabled_expansions must not contain duplicates")
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("agent config must be a YAML mapping")

        raw: dict[str, Any]
        if "agent" in loaded:
            allowed_root_keys = {"agent", "policy", "answer"}
            unknown_root_keys = sorted(set(loaded) - allowed_root_keys)
            if unknown_root_keys:
                raise ValueError(
                    "unknown root key(s) in agent config: "
                    + ", ".join(unknown_root_keys)
                )
            agent_section = loaded["agent"]
            if not isinstance(agent_section, dict):
                raise ValueError("agent must be a YAML mapping")
            raw = dict(agent_section)
            for section in ("policy", "answer"):
                if section in loaded and section in raw:
                    raise ValueError(
                        f"{section} must be configured either under agent "
                        "or at the YAML root, not both"
                    )
                if section in loaded:
                    raw[section] = loaded[section]
        else:
            raw = dict(loaded)
        return cls.model_validate(raw)

    def effective_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
