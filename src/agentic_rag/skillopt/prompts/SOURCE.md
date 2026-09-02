# Agentic RAG SkillOpt prompt bundle: full-agent-policy-v1

This local bundle is based on Microsoft SkillOpt tag `v0.2.0`, commit
`e4ea6a6771e797ef820cdd8bfea64c57e0481065`:

https://github.com/microsoft/SkillOpt/tree/v0.2.0/skillopt/prompts

SkillOpt is MIT licensed. The PyPI `skillopt==0.2.0` wheel contains the prompt
loader but omits these Markdown package-data files. Agentic RAG temporarily
points that loader at this repository bundle while the native trainer is
running.

The following files are Agentic RAG adaptations for full action-and-answer
skill optimization and are not verbatim upstream prompts:

- `analyst_error.md`
- `analyst_success.md`
- `ranking.md`

The merge prompts remain unchanged from the upstream v0.2.0 bundle. Every run
records the SHA256 of all six prompt files in `skillopt_prompt_bundle.json`.
