"""Composition root for one query-time Agentic RAG run."""

from __future__ import annotations

from pathlib import Path

from agentic_rag.agent.answer import (
    AnswerGenerator,
    OpenAIResponsesAnswerGenerator,
)
from agentic_rag.agent.artifacts import ArtifactWriter
from agentic_rag.agent.config import (
    AgentConfig,
    AnswerConfig,
    OllamaAnswerConfig,
    OllamaPolicyConfig,
    PolicyConfig,
)
from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.controller import AgentController
from agentic_rag.agent.evidence import EvidenceResolver
from agentic_rag.agent.expansion import ExpansionEngine
from agentic_rag.agent.models import EpisodeResult, Message
from agentic_rag.agent.policy import (
    OpenAIResponsesPolicy,
    PolicyClient,
)
from agentic_rag.agent.router import ActionRouter
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.agent.validator import DecisionValidator
from agentic_rag.embedding import EmbeddingBackend
from agentic_rag.retrieval import Retriever
from agentic_rag.storage import Substrate


class AgentHarness:
    def __init__(
        self,
        *,
        substrate: Substrate,
        config: AgentConfig,
        skill: SkillDocument,
        policy: PolicyClient,
        answer_generator: AnswerGenerator,
        output_root: str | Path,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> None:
        self.substrate = substrate
        self.config = config
        self.skill = skill
        self.policy = policy
        self.answer_generator = answer_generator
        self.output_root = Path(output_root)
        retriever = Retriever(
            substrate, embedding_backend=embedding_backend
        )
        expansion_engine = ExpansionEngine(
            substrate, embedding_backend=embedding_backend
        )
        evidence_resolver = EvidenceResolver(substrate)
        self.context_builder = PolicyContextBuilder(
            config.enabled_expansions,
            context_mode=config.context_mode,
            evidence_resolver=evidence_resolver,
        )
        self.controller = AgentController(
            policy=policy,
            answer_generator=answer_generator,
            context_builder=self.context_builder,
            validator=DecisionValidator(
                substrate, config.enabled_expansions
            ),
            router=ActionRouter(substrate, retriever, expansion_engine),
            evidence_resolver=evidence_resolver,
            skill=skill,
            max_steps=config.max_steps,
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
        answer_generator: AnswerGenerator | None = None,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> "AgentHarness":
        resolved_config = (
            config
            if isinstance(config, AgentConfig)
            else AgentConfig.from_yaml(config)
        )
        substrate = Substrate.open(substrate_path)
        skill = SkillDocument.load(skill_file)
        resolved_policy = (
            policy
            if policy is not None
            else _policy_from_config(resolved_config)
        )
        resolved_answer = (
            answer_generator
            if answer_generator is not None
            else _answer_from_config(resolved_config)
        )
        return cls(
            substrate=substrate,
            config=resolved_config,
            skill=skill,
            policy=resolved_policy,
            answer_generator=resolved_answer,
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
        initial_messages = list(
            self.controller.initial_messages(question, scope_id)
        )
        result = self.controller.run_episode(
            question, scope_id, episode_id=episode_id
        )
        destination = self.output_root / result.episode_id
        result.artifact_dir = str(destination)
        written = self.artifact_writer.write_episode(
            episode_id=result.episode_id,
            episode=result,
            trajectory=result.trajectory,
            # Keep the fixed protocol separate from the trainable skill. The
            # latter is persisted only in skill.md so SkillOpt can replace it.
            target_system_prompt=_first_message_for_role(
                initial_messages, "system"
            ),
            target_user_prompt=_messages_for_role(
                initial_messages, "user"
            ),
            skill_content=self.skill.content,
            effective_config=self.effective_config(),
        )
        result.artifact_dir = str(written)
        return result

    def effective_config(self) -> dict:
        manifest = self.substrate.manifest
        return {
            "agent": self.config.effective_dict(),
            "policy_context": {
                "handle_summary_max_chars": 160,
                "selection_semantics": "full_set_replacement",
            },
            "runtime_components": {
                "policy_client": type(self.policy).__name__,
                "answer_generator": type(self.answer_generator).__name__,
            },
            "skill": {
                "sha256": self.skill.sha256,
                "source_path": self.skill.source_path,
            },
            "substrate": {
                "root": str(self.substrate.root),
                "corpus_id": manifest.corpus_id,
                "split": manifest.split,
                "constructor_version": manifest.constructor_version,
                "schema_version": manifest.schema_version,
                "embedding_model": manifest.embedding_model.model_dump(
                    mode="json"
                ),
            },
        }


def _messages_for_role(
    messages: list[Message], role: str
) -> str:
    return "\n\n".join(
        message.content for message in messages if message.role == role
    )


def _first_message_for_role(
    messages: list[Message], role: str
) -> str:
    return next(
        (
            message.content
            for message in messages
            if message.role == role
        ),
        "",
    )


def _policy_from_config(config: AgentConfig) -> PolicyClient:
    policy_config = config.policy
    if isinstance(policy_config, PolicyConfig):
        return OpenAIResponsesPolicy(
            model=policy_config.model,
            max_retries=policy_config.max_retries,
            enabled_expansions=config.enabled_expansions,
        )
    if isinstance(policy_config, OllamaPolicyConfig):
        # Imported only when selected so OpenAI-only callers do not eagerly
        # initialize the local Ollama provider package.
        from agentic_rag.agent.ollama import OllamaChatPolicy

        return OllamaChatPolicy(
            model=policy_config.model,
            host=policy_config.host,
            temperature=policy_config.temperature,
            think=policy_config.think,
            timeout_seconds=policy_config.timeout_seconds,
            keep_alive=policy_config.keep_alive,
            max_retries=policy_config.max_retries,
            num_ctx=policy_config.num_ctx,
            enabled_expansions=config.enabled_expansions,
        )
    raise TypeError(
        f"unsupported policy provider: {policy_config.provider}"
    )


def _answer_from_config(config: AgentConfig) -> AnswerGenerator:
    answer_config = config.answer
    if isinstance(answer_config, AnswerConfig):
        return OpenAIResponsesAnswerGenerator(
            model=answer_config.model,
            max_retries=answer_config.max_retries,
        )
    if isinstance(answer_config, OllamaAnswerConfig):
        from agentic_rag.agent.ollama import (
            OllamaChatAnswerGenerator,
        )

        return OllamaChatAnswerGenerator(
            model=answer_config.model,
            host=answer_config.host,
            temperature=answer_config.temperature,
            think=answer_config.think,
            timeout_seconds=answer_config.timeout_seconds,
            keep_alive=answer_config.keep_alive,
            max_retries=answer_config.max_retries,
            num_ctx=answer_config.num_ctx,
        )
    raise TypeError(
        f"unsupported answer provider: {answer_config.provider}"
    )
