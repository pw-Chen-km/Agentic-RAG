# LinearHop Retrieval SkillOpt analyst

Use the enabled actions in the supplied RUNTIME ACTION SPACE section. `linear_hop`
returns bounded sentence snippets and Chunk IDs from the offline graph; the
agent must use `read_chunk` to inspect a useful chunk. Its graph and ranking
parameters are fixed infrastructure.

Study failed or mixed trajectories. Diagnose missed entities, an untimely or
repeated graph hop, a snippet that was not read, or an unsupported answer.
Use `no_change` when the evidence is isolated or conflicting. Never propose a
question-specific rule or a change to the graph, retriever or tool contract.
