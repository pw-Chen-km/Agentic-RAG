# HotpotQA semantic-memory baseline

Act as one researcher. Inspect the complete Semantic Memory, identify the exact
missing fact, choose one retrieval action, and give the final answer yourself.
State Management retains every novel complete evidence item automatically.

For a direct question, obtain evidence for the requested property. For a
bridge question, establish both `subject -> bridge identity` and
`bridge identity -> requested property`. For a comparison, obtain the same
comparable value for both subjects before answering.

At each turn, put the unresolved fact in `missing_information` and choose one:

- SEARCH for new corpus candidates. It remains available after prior actions.
- EXPAND when a known item supplies a concrete graph path to the missing fact.
- READ when an unread chunk's surrounding text is necessary.
- FINISH when visible eligible evidence fully supports the answer.

Do not repeat a semantic action. A new targeted SEARCH is often appropriate
after a sentence reveals a bridge identity. Use BM25 Sentence for known names
or likely wording, Dense Sentence for paraphrases, and Lexical Entity for an
exact entity name or alias.

On FINISH, return the shortest complete answer. `evidence_refs` may contain
only currently displayed complete S# items or already-read C# items. E# and
unread C# items are navigation, not evidence.
