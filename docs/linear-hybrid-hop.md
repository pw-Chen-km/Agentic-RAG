# LinearRAG-style graph hopping

The existing A-RAG retrieval baseline remains `direct_bm25` with
`question_or_visible_evidence`. The optional `linear_hop` action adds a
sentence/entity graph over the same chunk corpus. It is an additional action,
not a replacement for the baseline.

Build the graph from the existing corpus-only artifacts:

```bash
python scripts/build_linear_graph.py \
  --chunks data/novel/chunks.json \
  --sentence-index data/novel/index/sentence_index.pkl \
  --entity-index data/novel/entityhop/entity_linked.json \
  --output data/novel/linear_trigraph.json
```

The builder never reads questions, answers or gold evidence. It records
sentence-to-entity links, entity-to-sentence postings, entity co-occurrence and
within-chunk adjacent-sentence links. Sentence embeddings are reused from the
existing semantic index; no model is downloaded by the builder.

Enable the action in a separate experiment arm:

```yaml
linear_hop:
  enabled: true
  graph_index_file: data/novel/linear_trigraph.json
  sentence_index_file: data/novel/index/sentence_index.pkl
  entity_source: question_or_visible_evidence
  top_entities: 3
  max_hops: 2
  top_k: 5
  max_top_k: 10
```

The defaults are `max_hops: 2` and `top_k: 5`. The agent supplies an entity
and may leave both parameters at their defaults. The runtime uses the original
question from `AgentContext` to rank candidate bridge sentences, propagates a
bounded path score to newly activated entities, removes cycles and duplicate
chunks, and returns only snippets, bridge metadata and Chunk IDs. The agent can
then call `read_chunk` to retrieve full text.

The runtime graph action is named `linear_hop`; the existing `entity_hop`
action remains the Direct-BM25 control. A graph arm changes the public tool
menu and therefore needs its own recorded contract and independent rollout.
