# A-RAG + SkillOpt: Novel Reflection48k

Latest reusable code from the completed Novel Initial / Optimized / Original
A-RAG experiment. `v6r3` is an experiment revision label, not another model.
This public branch is a deliberately source-only, portable export. Generic runtime
modules, selected current runners, prompts and Skills are copied without changes;
old machine-specific deployment scripts, historical configs, raw records,
datasets/indexes, weights, virtualenvs and private paths are not published.
Their original versions and experimental lineage belong to the private data backup.

## Get this version

```bash
git clone --branch novel-reflection48k-v6r3 --single-branch https://github.com/pw-Chen-km/Agentic-RAG.git
cd Agentic-RAG
# Later, while on this branch:
git pull --ff-only
```

Cloning or pulling does not install packages, fetch models or run inference.
On a suitable future Linux machine, create a Python 3.10+ environment and install
the package with `python -m pip install -e '.[full,dev]'` as needed. This explicit
optional installation can be large; do not run it on a space-constrained laptop.
Baseline runtime was Ubuntu 22.04, Python 3.10 and Ollama 0.35.0. The captured
package inventory is in the private backup; `pyproject.toml` is not a lockfile.

## Contents and entry points

- `src/arag`: Agent, tools, exact Ollama context/output guards.
- `src/arag/tools/linear_hop.py` and `scripts/build_linear_graph.py`: optional
  LinearRAG-style sentence/entity graph hopping; it reuses the existing
  sentence index and returns snippets before `read_chunk`.
- `src/arag_skillopt`: Phase1, Meta, gates, sealed test and cache lineage checks.
- `prompts/`, `skills/`: runtime/Optimizer prompts and initial/final retrieval rules.
- `configs/novel_reflection48k.example.yaml`: derived portable config; only paths
  were changed from the original frozen run config. Use it explicitly with `--config`.
- `scripts/run_workflow.py`, `run_phase1.py`, `run_final_test.py`: guarded runners.
- `experiment_extensions/run_original_test.py`, `compare_three_way.py`: Original
  arm and three-way comparison entry points; use `--help` for required paths.
- `tests/`: selected synthetic offline unit tests with their fixture dependencies;
  not the complete historical deployment test suite. No tests/inference are auto-run.
- `backup_tools/`: read-only private backup integrity helpers.
- `provenance/published-source.json`: exact copied-file hashes and export policy.

The example retains Agent context 65536 / answer 4096 / input 60928 / safety 512,
no truncation/shift, two workers, seed 42, Optimizer medium / output 8192 /
Reflection and Merge input 48000. Judge retains its separate original settings;
known timeout and grading-quality limitations have not been fixed by this export.
The sealed final Skill is a small JSON/Markdown retrieval policy, not model weights.

## Experiment boundaries

Novel used the project's fixed 402 TRAIN / 402 validation / 1206 test split.
History spans multiple revisions, cached rollouts and previously analyzed test
data: it is not a clean all-48k rerun or fresh blind test. Initial-vs-Optimized
isolates Skill changes; Original-vs-new-framework is a whole-system comparison.
Unknown evaluations must remain unknown, not zeros. Novel evidence recall is N/A.
Test results must not feed Optimizer, gates or Skill selection. Hotpot remains
deferred and requires new human authorization; no auto-start queue is included.

See [restoration notes](docs/RESTORE_AND_REUSE.md) and [attribution](NOTICE.md).
