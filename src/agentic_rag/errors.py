"""Domain errors with stable error codes for API and CLI callers."""


class AgenticRAGError(Exception):
    code = "agentic_rag_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InputFormatError(AgenticRAGError):
    code = "input_format_error"


class BuildError(AgenticRAGError):
    code = "build_error"


class SubstrateNotFoundError(AgenticRAGError):
    code = "substrate_not_found"


class NodeNotFoundError(AgenticRAGError):
    code = "node_not_found"


class ScopeNotFoundError(AgenticRAGError):
    code = "scope_not_found"


class InvalidRetrievalPairError(AgenticRAGError):
    code = "invalid_retrieval_pair"


class IndexNotFoundError(AgenticRAGError):
    code = "index_not_found"


class ValidationFailedError(AgenticRAGError):
    code = "validation_failed"


class AgentConfigurationError(AgenticRAGError):
    code = "agent_configuration_error"


class InvalidDecisionError(AgenticRAGError):
    code = "invalid_decision"


class DisabledExpansionError(InvalidDecisionError):
    code = "disabled_expansion"


class EvidenceEligibilityError(InvalidDecisionError):
    code = "evidence_not_eligible"


class RetrievedTokenBudgetError(AgenticRAGError):
    code = "retrieved_token_budget_exceeded"


class PolicyRuntimeError(AgenticRAGError):
    code = "policy_runtime_error"


class AnswerGenerationError(AgenticRAGError):
    code = "answer_generation_error"


class ArtifactWriteError(AgenticRAGError):
    code = "artifact_write_error"
