# Agentic RAG interface study retrieval policy

Start with one `DENSE -> CHUNK` search using the original question. Use only
references shown in the current policy state. When entity annotations are
available, an entity may be expanded only through the relation listed by the
active interface condition. Read a chunk before using it as chunk evidence.

Do not infer or quote hidden source text. Finish only when the answer is
supported by complete visible sentences or a chunk that was read in full.
