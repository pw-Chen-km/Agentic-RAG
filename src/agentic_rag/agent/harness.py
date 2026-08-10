"""Composition root for one canonical Agentic RAG run."""

from __future__ import annotations

from pathlib import Path

from agentic_rag.agent.artifacts import ArtifactWriter
from agentic_rag.agent.config import AgentConfig, OllamaPolicyConfig, OpenAIPolicyConfig
from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.controller import BUDGET_FINALIZE_INSTRUCTION, AgentController
from agentic_rag.agent.evidence import EvidenceResolver
from agentic_rag.agent.expansion import ExpansionEngine
from agentic_rag.agent.models import EpisodeResult, Message
from agentic_rag.agent.policy import PolicyClient
from agentic_rag.agent.router import ActionRouter
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.agent.validator import DecisionValidator
from agentic_rag.substrate.embedding import EmbeddingBackend
from agentic_rag.substrate.retrieval import Retriever
from agentic_rag.substrate.storage import Substrate


class AgentHarness:
    def __init__(
        self,
        *,
        substrate: Substrate,
        config: AgentConfig,
        skill: SkillDocument,
        policy: PolicyClient,
        output_root: str | Path,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> None:
        self.substrate = substrate
        self.config = config
        self.skill = skill
        self.policy = policy
        self.output_root = Path(output_root)
        retriever = Retriever(substrate, embedding_backend=embedding_backend)
        expansion = ExpansionEngine(substrate, embedding_backend=embedding_backend)
        evidence = EvidenceResolver(substrate)
        self.context_builder = PolicyContextBuilder(
            substrate,
            config.enabled_expansions,
            show_available_action_options=config.show_available_action_options,
        )
        self.controller = AgentController(
            policy=policy,
            context_builder=self.context_builder,
            validator=DecisionValidator(substrate, config.enabled_expansions),
            router=ActionRouter(substrate, retriever, expansion),
            evidence_resolver=evidence,
            skill=skill,
            max_steps=config.max_steps,
            max_policy_attempts=config.max_policy_attempts,
            max_retrieved_tokens=config.max_retrieved_tokens,
        )
        self.artifact_writer = ArtifactWriter(self.output_root)

    @classmethod
    def from_config(
        cls,
        substrate_path: str | Path,
        config: AgentConfig | str | Path,
        skill_file: str | Path,
        output_root: str | Path,
        *,
        policy: PolicyClient | None = None,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> "AgentHarness":
        resolved_config = config if isinstance(config, AgentConfig) else AgentConfig.from_yaml(config)
        substrate = Substrate.open(substrate_path)
        return cls(
            substrate=substrate,
            config=resolved_config,
            skill=SkillDocument.load(skill_file),
            policy=policy or _policy_from_config(resolved_config),
            output_root=output_root,
            embedding_backend=embedding_backend,
        )

    @classmethod
    def from_skill_content(
        cls,
        substrate_path: str | Path,
        config: AgentConfig | str | Path,
        skill_content: str,
        output_root: str | Path,
        *,
        skill_source_path: str | None = None,
        policy: PolicyClient | None = None,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> "AgentHarness":
        resolved_config = config if isinstance(config, AgentConfig) else AgentConfig.from_yaml(config)
        substrate = Substrate.open(substrate_path)
        return cls(
            substrate=substrate,
            config=resolved_config,
            skill=SkillDocument.from_text(skill_content, source_path=skill_source_path),
            policy=policy or _policy_from_config(resolved_config),
            output_root=output_root,
            embedding_backend=embedding_backend,
        )

    def run(
        self,
        question: str,
        scope_id: str,
        *,
        episode_id: str | None = None,
    ) -> EpisodeResult:
        initial_messages = list(self.controller.initial_messages(question, scope_id))
        result = self.controller.run_episode(question, scope_id, episode_id=episode_id)
        destination = self.artifact_writer.path_for_episode(result.episode_id)
        result.artifact_dir = destination.as_posix()
        written = self.artifact_writer.write_episode(
            episode_id=result.episode_id,
            episode=result,
            trajectory=result.trajectory,
            target_system_prompt=_first_message_for_role(initial_messages, "system"),
            target_user_prompt=_messages_for_role(initial_messages, "user"),
            skill_content=self.skill.content,
            effective_config=self.effective_config(),
        )
        result.artifact_dir = written.as_posix()
        return result

    def build_io_trace(self, result: EpisodeResult) -> dict:
        prior_steps = []
        policy_calls: list[dict] = []
        for step in result.trajectory:
            built = self.context_builder.build(
                result.query,
                self.skill,
                step.state_before,
                prior_steps,
                scope_id=result.scope_id,
            )
            messages = list(built.messages)
            if step.observation is not None and step.observation.metadata.get("budget_finalize") is True:
                messages.append(Message(role="user", content=BUDGET_FINALIZE_INSTRUCTION))
            policy_calls.append(
                {
                    "step": step.step,
                    "policy_attempt": step.policy_attempt,
                    "input": [message.model_dump(mode="json") for message in messages],
                    "raw_policy_decision": (
                        step.decision.model_dump(mode="json") if step.decision is not None else None
                    ),
                    "resolved_decision": (
                        step.resolved_decision.model_dump(mode="json")
                        if step.resolved_decision is not None
                        else None
                    ),
                    "context_reference_map": (
                        step.context_reference_map.model_dump(mode="json")
                        if step.context_reference_map is not None
                        else None
                    ),
                    "validation_status": step.validation_status.value,
                    "validation_error": step.validation_error,
                    "observation": (
                        step.observation.model_dump(mode="json")
                        if step.observation is not None
                        else None
                    ),
                    "agent_visible_observation": step.agent_visible_observation,
                    "usage": step.usage.model_dump(mode="json"),
                }
            )
            prior_steps.append(step)
        return {
            "trace_format": "agentic-rag-episode-v1",
            "episode_id": result.episode_id,
            "target_input": {
                "question": result.query,
                "scope_id": result.scope_id,
                "skill_sha256": self.skill.sha256,
            },
            "policy_calls": policy_calls,
            "termination_reason": result.termination_reason.value,
            "total_usage": result.usage.model_dump(mode="json"),
        }

    def effective_config(self) -> dict:
        manifest = self.substrate.manifest
        return {
            "architecture": "semantic_memory_typed_refs_compact",
            "agent": self.config.effective_dict(),
            "policy_context": {
                "node_reference_scheme": "episode_local_typed_refs_with_frozen_visibility",
                "show_available_action_options": (
                    self.config.show_available_action_options
                ),
                "visible_sections": [
                    "question",
                    "action_protocol",
                    *(
                        ["available_action_options"]
                        if self.config.show_available_action_options
                        else []
                    ),
                    "skill",
                    "last_assessment",
                    "semantic_memory",
                    "attempted_actions",
                    "remaining_budget",
                ],
                "budget_representation": "compact_text",
                "stable_node_ids_visible_to_policy": False,
                "selection_semantics": "automatic_semantic_memory",
            },
            "runtime_components": {
                "policy_client": type(self.policy).__name__,
                "state_manager": "EpisodeStateManager",
                "controller_role": "stateless_loop_orchestrator",
            },
            "skill": {"sha256": self.skill.sha256, "source_path": self.skill.source_path},
            "substrate": {
                "root": self.substrate.root.as_posix(),
                "corpus_id": manifest.corpus_id,
                "split": manifest.split,
                "constructor_version": manifest.constructor_version,
                "schema_version": manifest.schema_version,
                "embedding_model": manifest.embedding_model.model_dump(mode="json"),
            },
        }


def _messages_for_role(messages: list[Message], role: str) -> str:
    return "\n\n".join(message.content for message in messages if message.role == role)


def _first_message_for_role(messages: list[Message], role: str) -> str:
    return next((message.content for message in messages if message.role == role), "")


def _policy_from_config(config: AgentConfig) -> PolicyClient:
    if isinstance(config.policy, OpenAIPolicyConfig):
        from agentic_rag.agent.providers.openai import OpenAIResponsesPolicy

        return OpenAIResponsesPolicy(
            model=config.policy.model,
            max_retries=config.policy.max_retries,
            enabled_expansions=config.enabled_expansions,
        )
    if isinstance(config.policy, OllamaPolicyConfig):
        from agentic_rag.agent.providers.ollama import OllamaChatPolicy

        return OllamaChatPolicy(
            model=config.policy.model,
            host=config.policy.host,
            temperature=config.policy.temperature,
            think=config.policy.think,
            timeout_seconds=config.policy.timeout_seconds,
            keep_alive=config.policy.keep_alive,
            max_retries=config.policy.max_retries,
            num_ctx=config.policy.num_ctx,
            enabled_expansions=config.enabled_expansions,
        )
    raise TypeError(f"unsupported policy provider: {config.policy.provider}")
