"""Configuration for the single semantic-memory agent workflow."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import PydanticCustomError

from agentic_rag.agent.models import DEFAULT_ENABLED_EXPANSIONS, ExpansionKind


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


OllamaThink = bool | Literal["low", "medium", "high"] | None


class OllamaPolicyConfig(ConfigModel):
    provider: Literal["ollama"] = "ollama"
    model: str = Field(default="qwen3.5:4b", min_length=1)
    host: str = Field(default="http://localhost:11434", min_length=1)
    temperature: float = Field(default=0.0, ge=0.0)
    think: OllamaThink = False
    timeout_seconds: float = Field(default=600.0, gt=0.0)
    keep_alive: str | int | float | None = None
    max_retries: int = Field(default=2, ge=0, le=10)
    num_ctx: int = Field(default=32_768, ge=2_048)
    max_output_tokens: int = Field(default=2_048, ge=1)

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
        message = "Ollama host must be a valid HTTP(S) URL with a hostname"
        try:
            parsed = urlparse(value)
            hostname = (parsed.hostname or "").casefold().rstrip(".")
            parsed.port
        except ValueError as exc:
            raise PydanticCustomError("ollama_host_invalid", message) from exc
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or not hostname
            or any(character.isspace() for character in value)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise PydanticCustomError("ollama_host_invalid", message)
        if hostname == "ollama.com" or hostname.endswith(".ollama.com"):
            raise PydanticCustomError(
                "ollama_cloud_unsupported",
                "Ollama Cloud does not support the required structured output",
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


class OpenAICompatiblePolicyConfig(ConfigModel):
    provider: Literal["openai_compatible"] = "openai_compatible"
    model: str = Field(min_length=1)
    base_url: str = Field(default="http://localhost:8000/v1", min_length=1)
    api_key: str = "EMPTY"
    temperature: float = Field(default=0.0, ge=0.0)
    timeout_seconds: float = Field(default=600.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0, le=10)
    num_ctx: int = Field(default=32_768, ge=2_048)
    max_output_tokens: int = Field(default=2_048, ge=1)

    @field_validator("model", "base_url")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value.rstrip("/") if value.startswith("http") else value


PolicyProviderConfig = Annotated[
    OllamaPolicyConfig | OpenAICompatiblePolicyConfig,
    Field(discriminator="provider"),
]


class AgentConfig(ConfigModel):
    """One configuration surface; no architecture selector exists."""

    max_steps: int = Field(default=15, ge=1)
    max_policy_attempts: int = Field(default=15, ge=1)
    max_retrieved_tokens: int = Field(default=12_000, ge=1)
    episode_timeout_seconds: float = Field(default=3_600.0, gt=0.0)
    enabled_expansions: tuple[ExpansionKind, ...] = DEFAULT_ENABLED_EXPANSIONS
    show_available_action_options: bool = True
    use_state_conditioned_schema: bool = True
    require_evidence_assessment: bool = True
    interface: str | None = None
    policy: PolicyProviderConfig = Field(default_factory=OllamaPolicyConfig)

    @field_validator("policy", mode="before")
    @classmethod
    def omitted_provider_is_ollama(cls, value: Any) -> Any:
        if isinstance(value, dict) and "provider" not in value:
            return {"provider": "ollama", **value}
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
        if "agent" not in loaded:
            return cls.model_validate(loaded)
        unknown = sorted(set(loaded) - {"agent", "policy"})
        if unknown:
            raise ValueError("unknown root key(s) in agent config: " + ", ".join(unknown))
        agent = loaded["agent"]
        if not isinstance(agent, dict):
            raise ValueError("agent must be a YAML mapping")
        raw = dict(agent)
        if "policy" in loaded:
            if "policy" in raw:
                raise ValueError("policy must be configured at one YAML level only")
            raw["policy"] = loaded["policy"]
        return cls.model_validate(raw)

    def effective_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# A concise alias for callers that prefer the provider-specific name.
PolicyConfig = OllamaPolicyConfig
