from agentic_rag.agent.expansion import ExpansionEngine
from agentic_rag.agent.interface import get_interface_contract
from agentic_rag.agent.models import EpisodeState, ObservationStatus, SearchAction, SearchMethod, SearchTarget
from agentic_rag.agent.router import ActionRouter
from agentic_rag.substrate.ranking import RankingService
from agentic_rag.substrate.retrieval import Retriever
from agentic_rag.substrate.storage import Substrate


def test_router_uses_projector_visibility_delta(built_substrate, fake_embedder) -> None:
    substrate = Substrate.open(built_substrate)
    ranking = RankingService(substrate, embedding_backend=fake_embedder)
    router = ActionRouter(
        substrate,
        Retriever(substrate, embedding_backend=fake_embedder, ranking_service=ranking),
        ExpansionEngine(substrate, embedding_backend=fake_embedder, ranking_service=ranking),
        get_interface_contract("C2"),
    )
    observation = router.execute(
        SearchAction(query="Marie Curie", method=SearchMethod.DENSE, target=SearchTarget.CHUNK),
        EpisodeState.initial(), question="Where was Marie Curie born?",
        scope_id="q1", action_id="projection-authority",
    )
    assert observation.status is ObservationStatus.OK
    _, expected, _ = router.projector.project(observation.results, action_type="SEARCH")
    assert observation.metadata["visibility_delta"] == expected
    assert expected["visible_entity_ids"]
    assert expected["eligible_sentence_ids"]
