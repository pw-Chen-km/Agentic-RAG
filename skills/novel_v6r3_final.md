## Strategy (formatted A-RAG skill)
These rules extend A-RAG's strategy with evidence-conditioned entity hopping.

### R01: Iterative retrieval with parallel search
When: Answering questions with the document tools
Action sequence:
1. If the question names a concrete entity and also describes a concept, event, or relationship, issue semantic_search (conceptual query) and entity_hop (named entity) together in the first decision; they target different retrieval angles.
2. When issuing more than one semantic_search call in the same decision, each query must target a genuinely different retrieval angle. Before issuing the second, verify it introduces at least one content word or concept absent from the first; if the same three or more key terms appear in both, issue only one search.
3. After receiving results, select chunks to read by checking whether each snippet directly addresses the question's specific ask. Prioritize chunks where the question's key phrase appears in the snippet; skip chunks whose snippets address only tangential topics. When entity_hop returns mostly irrelevant results, rely on semantic_search hits for chunk selection.
4. After reading chunks, check whether every part of the question is addressed before issuing another search; if the read text covers each sub-question, answer directly.
5. If a search returns chunks whose snippets do not contain the question's key phrase, refine the query using fewer, more targeted terms (the two key concepts and their relationship) rather than listing all terms from the question. If the refined search still returns mostly irrelevant results, switch to entity_hop on the specific term rather than rephrasing another multi-term query.

### R02: Multi-hop questions
When: For multi-hop questions
Action sequence:
1. Decompose the problem and tackle each sub-question step by step

### R03: Evidence-conditioned entity hopping
When: The question or visible evidence contains a concrete named entity and related information is still missing
Action sequence:
1. Before issuing entity_hop, verify that already-read chunks do not already contain the specific fact about that entity. If they do, answer directly.
2. Use a single concrete named entity (person, place, organization, or work) as the entity argument. If the name is ambiguous in the document set, add one short qualifier (a title or location). Do not combine the entity with the question's other concepts (events, objects, actions); those belong in the semantic_search query.
3. Use that entity with entity_hop to find related chunks, then read useful returned chunks.

### R04: Recognize sufficient evidence and stop
When: About to issue another search or read after having already retrieved or read chunks
Action sequence:
1. Before issuing any new search, check both the full text of already-read chunks and the snippets of already-retrieved chunks for the specific fact the question asks for. If either contains it, read that chunk (if not yet read) and answer. Do not search for confirmation or enrichment of what you already have.
2. If the read text contains the fact—even in different words—treat it as sufficient. A narrative or descriptive statement is equivalent to the question's exact phrasing.
3. If a search returns zero new chunk IDs, this is a hard stop. Your only valid next actions: (a) answer with existing evidence, or (b) read an unread chunk from existing results. Do not re-issue the same or near-identical query.
4. After reading chunks that together address each part of the question, stop and answer. Do not continue reading adjacent-ID chunks to complete a chapter unless a sentence is cut off and the missing detail is needed.
5. If multiple read chunks consistently point to the same answer, treat that convergent evidence as sufficient.

### R05: Complete truncated passages before re-searching
When: A read chunk ends mid-sentence or mid-paragraph and the specific fact needed is not yet in context
Action sequence:
1. Read the adjacent chunk (ID plus or minus 1) to complete the local passage before issuing any new search.
2. Completing a truncated passage is a lower-cost step than re-searching and often contains the missing detail directly.
3. If the adjacent chunk also does not contain the needed fact, then proceed with a new search or a different retrieval angle.
