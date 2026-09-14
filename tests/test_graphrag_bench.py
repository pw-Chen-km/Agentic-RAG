from agentic_rag.evaluation.graphrag_bench import GraphRAGSemanticEvaluator, rouge_l


class FakeJudge:
    def complete(self, prompt):
        if "Split the text" in prompt:
            text = "Paris is the capital of France."
            return {"statements": [text]}, {"calls": 1}
        if "Classify each answer" in prompt:
            return {"items": [{"verdict": "TP", "reason": "match"}]}, {"calls": 1}
        if "Rate context" in prompt:
            return {"score": 2}, {"calls": 1}
        return {"items": [{"attributed": 1, "supported": 1}]}, {"calls": 1}

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts], {"calls": 1}


def test_rouge_l_uses_lcs_and_empty_is_zero():
    assert rouge_l("a b c", "a c") > 0
    assert rouge_l("", "a") == 0


def test_semantic_evaluator_formula_and_applicability():
    result = GraphRAGSemanticEvaluator(FakeJudge()).evaluate(
        question={"question": "capital?", "answer": "Paris is the capital of France.", "question_type": "Fact Retrieval"},
        answer="Paris is the capital of France.",
        contexts=["Paris is the capital of France."],
        evidence=["Paris is the capital of France."],
    )
    assert result["metrics"]["answer_correctness"] == 1.0
    assert result["metrics"]["rouge_l"] == 1.0
    assert result["metrics"]["coverage"] is None
    assert result["details"]["coverage"]["status"] == "not_applicable"
