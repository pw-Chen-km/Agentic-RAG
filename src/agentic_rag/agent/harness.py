"""Composition root for one query-time Agentic RAG run."""

from __future__ import annotations

from pathlib import Path

from agentic_rag.agent.answer import (
    AnswerGenerator,
    OpenAIResponsesAnswerGenerator,
    answer_provider_input,
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
from agentic_rag.agent.controller import (
    BUDGET_FINALIZE_INSTRUCTION,
    AgentController,
)
from agentic_rag.agent.evidence import EvidenceResolver
from agentic_rag.agent.expansion import ExpansionEngine
from agentic_rag.agent.models import EpisodeResult, Message, PolicyStagePhase
from agentic_rag.agent.policy import (
    OpenAIResponsesPolicy,
    PolicyClient,
)
from agentic_rag.agent.router import ActionRouter
from agentic_rag.agent.skill import ProgressiveSkillBundle, SkillDocument
from agentic_rag.agent.validator import DecisionValidator
from agentic_rag.embedding import EmbeddingBackend
from agentic_rag.retrieval import Retriever
from agentic_rag.storage import Substrate
from agentic_rag.benchmark_profiles import (
    AnswerMode,
    get_arag_dataset_profile,
)
from agentic_rag.errors import InputFormatError


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
        answer_mode: AnswerMode = AnswerMode.SHORT,
        skill_bundle: ProgressiveSkillBundle | None = None,
    ) -> None:
        self.substrate = substrate
        self.config = config
        self.skill = skill
        self.policy = policy
        self.answer_generator = answer_generator
        self.answer_mode = answer_mode
        self.skill_bundle = skill_bundle
        self.output_root = Path(output_root)
        single_agent_v2_compact = (
            config.workflow_mode == "single_agent_v2_compact"
        )
        single_agent_v2 = config.workflow_mode in {
            "single_agent_v2",
            "single_agent_v2_compact",
        }
        single_agent_v22 = config.workflow_mode == "single_agent_v2_2"
        single_agent_v3_action_catalog = (
            config.workflow_mode == "single_agent_v3_action_catalog"
        )
        single_agent_v32 = config.workflow_mode == "single_agent_v3_2"
        single_agent_v3_typed_refs = (
            config.workflow_mode
            in {"single_agent_v3_typed_refs", "single_agent_v3_2"}
        )
        single_agent_v3 = config.workflow_mode in {
            "single_agent_v3",
            "single_agent_v3_action_catalog",
        }
        if single_agent_v22 and skill_bundle is None:
            raise ValueError(
                "single_agent_v2_2 requires a progressive skill bundle"
            )
        if (
            skill_bundle is not None
            and skill.sha256 != skill_bundle.root.sha256
        ):
            raise ValueError(
                "skill must be the root document of the supplied progressive "
                "skill bundle"
            )
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
            # V2-2 inherits V2's accumulated-evidence state projection. Its
            # actual two-stage prompts are built by V22ContextBuilder.
            single_agent_v2=(single_agent_v2 or single_agent_v22),
            single_agent_v2_compact=single_agent_v2_compact,
            single_agent_v3=single_agent_v3,
            single_agent_v3_action_catalog=(
                single_agent_v3_action_catalog
            ),
            single_agent_v3_typed_refs=single_agent_v3_typed_refs,
            include_v3_last_assessment=(
                config.v3_include_last_assessment
            ),
            include_v3_latest_event=(
                config.v3_include_latest_event and not single_agent_v32
            ),
            include_v3_attempted_actions=(
                config.v3_include_attempted_actions
            ),
            include_v3_budget=(
                True if single_agent_v32 else config.v3_include_budget
            ),
            compact_v3_budget=single_agent_v32,
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
            skill_bundle=skill_bundle,
            max_steps=config.max_steps,
            max_policy_attempts=config.max_policy_attempts,
            max_consecutive_invalid_attempts=(
                config.max_consecutive_invalid_attempts
            ),
            max_retrieved_tokens=config.max_retrieved_tokens,
            single_agent_v2=single_agent_v2,
            single_agent_v22=single_agent_v22,
            single_agent_v3=single_agent_v3,
            single_agent_v3_typed_refs=single_agent_v3_typed_refs,
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
        answer_mode = _answer_mode_for_substrate(substrate)
        skill_bundle = None
        if resolved_config.workflow_mode == "single_agent_v2_2":
            skill_bundle = ProgressiveSkillBundle.load(skill_file)
            skill = skill_bundle.root
        else:
            skill = SkillDocument.load(skill_file)
        resolved_policy = (
            policy
            if policy is not None
            else _policy_from_config(resolved_config)
        )
        resolved_answer = (
            answer_generator
            if answer_generator is not None
            else _answer_from_config(resolved_config, answer_mode=answer_mode)
        )
        return cls(
            substrate=substrate,
            config=resolved_config,
            skill=skill,
            policy=resolved_policy,
            answer_generator=resolved_answer,
            output_root=output_root,
            embedding_backend=embedding_backend,
            answer_mode=answer_mode,
            skill_bundle=skill_bundle,
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
        answer_generator: AnswerGenerator | None = None,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> "AgentHarness":
        """Compose a harness from an in-memory SkillOpt candidate skill.

        SkillOpt supplies candidate skills as Markdown strings during rollout.
        Keeping this as a separate constructor preserves the existing
        file-backed public interface while preventing adapters from creating
        temporary skill files solely to call the harness.
        """

        resolved_config = (
            config
            if isinstance(config, AgentConfig)
            else AgentConfig.from_yaml(config)
        )
        if resolved_config.workflow_mode == "single_agent_v2_2":
            raise ValueError(
                "single_agent_v2_2 requires a file-backed progressive "
                "skill bundle; in-memory SkillOpt candidates are not "
                "supported"
            )
        substrate = Substrate.open(substrate_path)
        answer_mode = _answer_mode_for_substrate(substrate)
        skill = SkillDocument.from_text(
            skill_content,
            source_path=skill_source_path,
        )
        resolved_policy = (
            policy
            if policy is not None
            else _policy_from_config(resolved_config)
        )
        resolved_answer = (
            answer_generator
            if answer_generator is not None
            else _answer_from_config(resolved_config, answer_mode=answer_mode)
        )
        return cls(
            substrate=substrate,
            config=resolved_config,
            skill=skill,
            policy=resolved_policy,
            answer_generator=resolved_answer,
            output_root=output_root,
            embedding_backend=embedding_backend,
            answer_mode=answer_mode,
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
        destination = self.artifact_writer.path_for_episode(result.episode_id)
        result.artifact_dir = destination.as_posix()
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
            skill_bundle=(
                self.skill_bundle.manifest()
                if self.skill_bundle is not None
                else None
            ),
        )
        result.artifact_dir = written.as_posix()
        return result

    def build_io_trace(self, result: EpisodeResult) -> dict:
        """Reconstruct the exact logical target inputs and outputs.

        The trace intentionally contains no HTTP headers or credentials.  It
        records the provider messages assembled by the context builder, the
        parsed structured decision, the environment observation, and the
        evidence-only answer call.  SkillOpt rollout code can persist this
        beside the immutable six-file episode artifact set.
        """

        prior_steps = []
        policy_calls: list[dict] = []
        for step in result.trajectory:
            if step.policy_stages:
                raw_observation = (
                    step.observation.model_dump(mode="json")
                    if step.observation is not None
                    else None
                )
                call_stages = [
                    stage
                    for stage in step.policy_stages
                    if stage.usage.policy_calls > 0
                ]
                final_stage_index = len(call_stages) - 1
                for stage_index, stage in enumerate(call_stages):
                    selection_output = (
                        stage.output
                        if stage.phase
                        == PolicyStagePhase.ACTION_SELECTION
                        else None
                    )
                    policy_calls.append(
                        {
                            "step": step.step,
                            "policy_attempt": step.policy_attempt,
                            "cycle": step.policy_attempt,
                            "stage": stage.phase.value,
                            "action_type": (
                                stage.action_type.value
                                if stage.action_type is not None
                                else (
                                    selection_output.get("action_type")
                                    if selection_output is not None
                                    else None
                                )
                            ),
                            "input": [
                                message.model_dump(mode="json")
                                for message in stage.messages
                            ],
                            "output": stage.output,
                            "pending_selected_evidence": (
                                selection_output.get(
                                    "selected_evidence_refs", []
                                )
                                if selection_output is not None
                                else None
                            ),
                            "stage_error": stage.error,
                            "disclosed_skill_paths": (
                                stage.disclosed_skill_paths
                            ),
                            "disclosed_skill_hashes": (
                                stage.disclosed_skill_hashes
                            ),
                            "validator": (
                                {
                                    "status": step.validation_status.value,
                                    "error": step.validation_error,
                                }
                                if stage_index == final_stage_index
                                else None
                            ),
                            "final_handle_action": (
                                (
                                    step.repaired_decision or step.decision
                                ).action.model_dump(mode="json")
                                if stage_index == final_stage_index
                                and (
                                    step.repaired_decision is not None
                                    or step.decision is not None
                                )
                                else None
                            ),
                            "final_stable_action": (
                                step.resolved_decision.action.model_dump(
                                    mode="json"
                                )
                                if stage_index == final_stage_index
                                and step.resolved_decision is not None
                                else None
                            ),
                            "observation": (
                                raw_observation
                                if stage_index == final_stage_index
                                else None
                            ),
                            "agent_visible_observation": (
                                step.agent_visible_observation
                                if stage_index == final_stage_index
                                else None
                            ),
                            "usage": stage.usage.model_dump(mode="json"),
                        }
                    )
                prior_steps.append(step)
                continue
            built = self.context_builder.build(
                result.query,
                self.skill,
                step.state_before,
                prior_steps,
                scope_id=result.scope_id,
            )
            policy_messages = list(built.messages)
            if (
                step.observation is not None
                and step.observation.metadata.get("budget_finalize") is True
            ):
                policy_messages.append(
                    Message(
                        role="user",
                        content=BUDGET_FINALIZE_INSTRUCTION,
                    )
                )
            raw_decision = (
                step.decision.model_dump(mode="json")
                if step.decision is not None
                else None
            )
            repaired_model = getattr(step, "repaired_decision", None)
            repaired_decision = (
                repaired_model.model_dump(mode="json")
                if repaired_model is not None
                else None
            )
            resolved_model = getattr(step, "resolved_decision", None)
            resolved_decision = (
                resolved_model.model_dump(mode="json")
                if resolved_model is not None
                else None
            )
            raw_observation = (
                step.observation.model_dump(mode="json")
                if step.observation is not None
                else None
            )
            policy_calls.append(
                {
                    "step": step.step,
                    "policy_attempt": step.policy_attempt,
                    "input": [
                        message.model_dump(mode="json")
                        for message in policy_messages
                    ],
                    # ``output`` remains as a compatibility alias for older
                    # trace readers.  It is the parsed LLM output and therefore
                    # contains only episode-local handles such as S1/C1/E1.
                    "output": raw_decision,
                    "raw_policy_decision": raw_decision,
                    "raw_handle_action": (
                        raw_decision.get("action")
                        if raw_decision is not None
                        else None
                    ),
                    "repaired_policy_decision": repaired_decision,
                    "repaired_handle_action": (
                        repaired_decision.get("action")
                        if repaired_decision is not None
                        else None
                    ),
                    "repair_code": step.repair_code,
                    # The Controller resolves every handle before validation
                    # and execution.  Persist that exact resolution beside the
                    # raw output so replay and audit never have to infer it.
                    "resolved_decision": resolved_decision,
                    "resolved_stable_action": (
                        resolved_decision.get("action")
                        if resolved_decision is not None
                        else None
                    ),
                    "context_reference_map": (
                        step.context_reference_map.model_dump(mode="json")
                        if step.context_reference_map is not None
                        else None
                    ),
                    "validation_status": step.validation_status.value,
                    "validation_error": step.validation_error,
                    # Raw stable-ID environment result for deterministic audit.
                    "observation": raw_observation,
                    "raw_observation": raw_observation,
                    # Exact handle-safe semantic feedback persisted to
                    # conversation.json and shown to the Policy next turn.
                    "agent_visible_observation": (
                        step.agent_visible_observation
                    ),
                    # FINISH StepRecord usage also includes the separate
                    # answer call.  Keep the Policy trace stage-pure so the
                    # same tokens are not attributed to both calls.
                    "usage": {
                        "policy_calls": step.usage.policy_calls,
                        "input_tokens": step.usage.policy_input_tokens,
                        "output_tokens": step.usage.policy_output_tokens,
                        "reasoning_tokens": (
                            step.usage.policy_reasoning_tokens
                        ),
                    },
                    "retrieved_tokens": step.usage.retrieved_tokens,
                }
            )
            prior_steps.append(step)

        answer_usage = {
            "answer_calls": result.usage.answer_calls,
            "answer_input_tokens": result.usage.answer_input_tokens,
            "answer_output_tokens": result.usage.answer_output_tokens,
            "answer_reasoning_tokens": result.usage.answer_reasoning_tokens,
        }
        answer_call = None
        if (
            result.answer is not None
            and result.resolved_evidence
            and result.usage.answer_calls > 0
        ):
            answer_call = {
                "input": answer_provider_input(
                    result.query,
                    result.resolved_evidence,
                    answer_mode=self.answer_mode,
                ),
                "output": {"answer": result.answer},
                "usage": answer_usage,
            }

        v22_metrics = (
            _v22_trace_metrics(result)
            if self.config.workflow_mode == "single_agent_v2_2"
            else None
        )
        return {
            "trace_format": (
                "agentic-rag-logical-io-v3"
                if v22_metrics is not None
                else "agentic-rag-logical-io-v2"
            ),
            "episode_id": result.episode_id,
            "target_input": {
                "question": result.query,
                "scope_id": result.scope_id,
                "skill_sha256": self.skill.sha256,
                **(
                    {"skill_bundle_sha256": self.skill_bundle.sha256}
                    if self.skill_bundle is not None
                    else {}
                ),
            },
            "policy_calls": policy_calls,
            "answer_generation": answer_call,
            "termination_reason": result.termination_reason.value,
            "total_usage": result.usage.model_dump(mode="json"),
            **({"v2_2_metrics": v22_metrics} if v22_metrics else {}),
        }

    def effective_config(self) -> dict:
        manifest = self.substrate.manifest
        return {
            "agent": self.config.effective_dict(),
            "policy_context": {
                "handle_summary_max_chars": 160,
                "node_reference_scheme": (
                    "episode_local_typed_refs_with_frozen_visibility_v1"
                    if self.config.workflow_mode
                    in {"single_agent_v3_typed_refs", "single_agent_v3_2"}
                    else (
                        "context_local_memory_and_citation_indices_v1"
                        if self.config.workflow_mode
                        in {
                            "single_agent_v3",
                            "single_agent_v3_action_catalog",
                        }
                        else "episode_local_typed_handles_v1"
                    )
                ),
                "observation_contract": "minimal_policy_observation_v1",
                **(
                    {
                        "latest_event_visible": (
                            self.context_builder.include_v3_latest_event
                        ),
                        "budget_representation": (
                            "compact_text_v1"
                            if self.context_builder.compact_v3_budget
                            else (
                                "structured_v1"
                                if self.context_builder.include_v3_budget
                                else "omitted"
                            )
                        ),
                    }
                    if self.context_builder.semantic_memory_v3
                    else {}
                ),
                "selection_semantics": (
                    "automatic_semantic_memory"
                    if self.config.workflow_mode
                    in {
                        "single_agent_v3",
                        "single_agent_v3_action_catalog",
                        "single_agent_v3_typed_refs",
                        "single_agent_v3_2",
                    }
                    else (
                        "controller_accumulated"
                        if self.config.workflow_mode
                        in {
                            "single_agent_v2",
                            "single_agent_v2_compact",
                            "single_agent_v2_2",
                        }
                        else "full_set_replacement"
                    )
                ),
                "stable_node_ids_visible_to_policy": False,
            },
            "runtime_components": {
                "policy_client": type(self.policy).__name__,
                "answer_generator": type(self.answer_generator).__name__,
                "state_manager": "EpisodeStateManager",
                "controller_role": "stateless_loop_orchestrator",
            },
            "answer_contract": {"mode": self.answer_mode.value},
            "skill": {
                "sha256": self.skill.sha256,
                "source_path": self.skill.source_path,
            },
            **(
                {"skill_bundle": self.skill_bundle.manifest()}
                if self.skill_bundle is not None
                else {}
            ),
            "substrate": {
                "root": self.substrate.root.as_posix(),
                "corpus_id": manifest.corpus_id,
                "split": manifest.split,
                "constructor_version": manifest.constructor_version,
                "schema_version": manifest.schema_version,
                "embedding_model": manifest.embedding_model.model_dump(
                    mode="json"
                ),
            },
        }


def _v22_trace_metrics(result: EpisodeResult) -> dict[str, int]:
    """Summarize progressive-cycle outcomes without reinterpreting actions."""

    initial_invalid = 0
    repair_attempts = 0
    repair_success = 0
    selection_invalid = 0
    unresolved_invalid = 0
    unavailable = 0
    invalid_indices: list[int] = []
    policy_calls = 0

    for index, step in enumerate(result.trajectory):
        policy_calls += sum(
            stage.usage.policy_calls for stage in step.policy_stages
        )
        selection_stage = next(
            (
                stage
                for stage in step.policy_stages
                if stage.phase == PolicyStagePhase.ACTION_SELECTION
            ),
            None,
        )
        draft_stage = next(
            (
                stage
                for stage in step.policy_stages
                if stage.phase == PolicyStagePhase.ACTION_DRAFT
            ),
            None,
        )
        repair_stage = next(
            (
                stage
                for stage in step.policy_stages
                if stage.phase == PolicyStagePhase.ACTION_REPAIR
            ),
            None,
        )
        if selection_stage is not None and selection_stage.error is not None:
            selection_invalid += 1
        if draft_stage is not None and draft_stage.error is not None:
            initial_invalid += 1
        if repair_stage is not None:
            repair_attempts += 1
            if (
                repair_stage.error is None
                and step.validation_status.value == "valid"
            ):
                repair_success += 1
        if step.validation_status.value == "invalid":
            invalid_indices.append(index)
            if repair_stage is not None or draft_stage is not None:
                unresolved_invalid += 1
            # An invalid draft with no recovery call means the catalog had no
            # legal choice for that action family in the frozen state.
            if (
                draft_stage is not None
                and draft_stage.error is not None
                and repair_stage is None
            ):
                unavailable += 1

    reselections = sum(
        1 for index in invalid_indices if index + 1 < len(result.trajectory)
    )
    return {
        "action_cycles": len(result.trajectory),
        "policy_calls": policy_calls,
        "initial_invalid": initial_invalid,
        "repair_attempts": repair_attempts,
        "repair_success": repair_success,
        "selection_invalid": selection_invalid,
        "reselections": reselections,
        "unresolved_invalid": unresolved_invalid,
        "unavailable": unavailable,
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
            direct_answer=(
                config.workflow_mode
                in {
                    "single_agent_v2",
                    "single_agent_v2_compact",
                    "single_agent_v2_2",
                    "single_agent_v3",
                    "single_agent_v3_action_catalog",
                    "single_agent_v3_typed_refs",
                    "single_agent_v3_2",
                }
            ),
            semantic_memory_v3=(
                config.workflow_mode
                in {"single_agent_v3", "single_agent_v3_action_catalog"}
            ),
            semantic_memory_v31=(
                config.workflow_mode
                in {"single_agent_v3_typed_refs", "single_agent_v3_2"}
            ),
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
            direct_answer=(
                config.workflow_mode
                in {
                    "single_agent_v2",
                    "single_agent_v2_compact",
                    "single_agent_v2_2",
                    "single_agent_v3",
                    "single_agent_v3_action_catalog",
                    "single_agent_v3_typed_refs",
                    "single_agent_v3_2",
                }
            ),
            semantic_memory_v3=(
                config.workflow_mode
                in {"single_agent_v3", "single_agent_v3_action_catalog"}
            ),
            semantic_memory_v31=(
                config.workflow_mode
                in {"single_agent_v3_typed_refs", "single_agent_v3_2"}
            ),
        )
    raise TypeError(
        f"unsupported policy provider: {policy_config.provider}"
    )


def _answer_from_config(
    config: AgentConfig,
    *,
    answer_mode: AnswerMode = AnswerMode.SHORT,
) -> AnswerGenerator:
    answer_config = config.answer
    if isinstance(answer_config, AnswerConfig):
        return OpenAIResponsesAnswerGenerator(
            model=answer_config.model,
            max_retries=answer_config.max_retries,
            answer_mode=answer_mode,
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
            answer_mode=answer_mode,
        )
    raise TypeError(
        f"unsupported answer provider: {answer_config.provider}"
    )


def _answer_mode_for_substrate(substrate: Substrate) -> AnswerMode:
    try:
        return get_arag_dataset_profile(substrate.manifest.dataset).answer_mode
    except InputFormatError:
        return AnswerMode.SHORT
