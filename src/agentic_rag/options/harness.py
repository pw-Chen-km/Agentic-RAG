"""Composition root for the independent Options v1 runtime."""

from __future__ import annotations

from pathlib import Path

from agentic_rag.agent.artifacts import ArtifactWriter
from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.evidence import EvidenceResolver
from agentic_rag.agent.expansion import ExpansionEngine
from agentic_rag.agent.harness import _policy_from_config
from agentic_rag.agent.policy import PolicyClient
from agentic_rag.agent.router import ActionRouter
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.agent.validator import DecisionValidator
from agentic_rag.options.catalog import OptionCatalog
from agentic_rag.options.context import OptionContextBuilder
from agentic_rag.options.controller import OptionsAgentController
from agentic_rag.substrate.embedding import EmbeddingBackend
from agentic_rag.substrate.retrieval import Retriever
from agentic_rag.substrate.storage import Substrate


class OptionsAgentHarness:
    """Run Options v1 while leaving the legacy AgentHarness untouched."""

    def __init__(
        self, *, substrate: Substrate, config: AgentConfig, skill: SkillDocument,
        catalog: OptionCatalog, policy: PolicyClient, output_root: str | Path,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> None:
        self.substrate = substrate
        self.config = config
        self.skill = skill
        self.catalog = catalog
        self.policy = policy
        self.output_root = Path(output_root)
        retriever = Retriever(substrate, embedding_backend=embedding_backend)
        expansion = ExpansionEngine(substrate, embedding_backend=embedding_backend)
        self.base_context_builder = PolicyContextBuilder(
            substrate, config.enabled_expansions,
            show_available_action_options=config.show_available_action_options,
            use_state_conditioned_schema=config.use_state_conditioned_schema,
        )
        self.context_builder = OptionContextBuilder(substrate, self.base_context_builder, catalog)
        self.controller = OptionsAgentController(
            policy=policy, context_builder=self.context_builder,
            base_context_builder=self.base_context_builder,
            validator=DecisionValidator(substrate, config.enabled_expansions),
            router=ActionRouter(substrate, retriever, expansion),
            evidence_resolver=EvidenceResolver(substrate), skill=skill,
            catalog=catalog, max_steps=config.max_steps,
            max_policy_attempts=config.max_policy_attempts,
            max_retrieved_tokens=config.max_retrieved_tokens,
        )
        self.artifact_writer = ArtifactWriter(self.output_root)

    @classmethod
    def from_config(
        cls, substrate_path: str | Path, config: AgentConfig | str | Path,
        skill_file: str | Path | None, option_file: str | Path, output_root: str | Path,
        *, policy: PolicyClient | None = None,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> "OptionsAgentHarness":
        resolved = config if isinstance(config, AgentConfig) else AgentConfig.from_yaml(config)
        substrate = Substrate.open(substrate_path)
        catalog = OptionCatalog.load(option_file)
        catalog_skill = (
            SkillDocument.load(skill_file)
            if skill_file is not None
            else SkillDocument.from_text(catalog.render_markdown(), source_path=str(option_file))
        )
        return cls(
            substrate=substrate, config=resolved,
            skill=catalog_skill, catalog=catalog,
            policy=policy or _policy_from_config(resolved), output_root=output_root,
            embedding_backend=embedding_backend,
        )

    def run(self, question: str, scope_id: str, *, episode_id: str | None = None):
        initial = list(self.controller.initial_messages(question, scope_id))
        result = self.controller.run_episode(question, scope_id, episode_id=episode_id)
        destination = self.artifact_writer.path_for_episode(result.episode_id)
        result.artifact_dir = destination.as_posix()
        written = self.artifact_writer.write_episode(
            episode_id=result.episode_id, episode=result,
            trajectory=result.trajectory,
            target_system_prompt=next((m.content for m in initial if m.role == "system"), ""),
            target_user_prompt="\n\n".join(m.content for m in initial if m.role == "user"),
            # ``self.skill`` is already the catalog-derived runtime document
            # when the harness is constructed from an option file.  Do not
            # append the same Markdown a second time to the audit artifact.
            skill_content=self.skill.content,
            effective_config={
                "architecture": "agentic-rag-options-v1",
                "options_catalog_sha256": self.catalog.sha256(),
                "options_catalog_source": self.catalog.source_path,
                "agent": self.config.effective_dict(),
            },
        )
        result.artifact_dir = written.as_posix()
        return result
