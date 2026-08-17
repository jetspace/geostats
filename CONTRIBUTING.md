# Contributing to Jetspace

Patches belong to [Jetspace Research](https://github.com/orgs/jetspace/repositories).
Read [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) first.

This repository is a research program: convex geometry, geometric
probability, information geometry, and stochastic analysis on manifolds.
Keep that line of argument intact.

## Standards

- Mathematics in `$...$` and `$$...$$` only.
- Cite [1]-[5] from [`REFERENCES.md`](REFERENCES.md). Do not present those
  theorems as original, and do not redistribute the source texts.
- No em dashes.
- Keep changes scoped, reviewable, and checked against an identity.
- Add or update tests when a kernel or identity changes.
- Keep [`notebooks/program.ipynb`](notebooks/program.ipynb) in sync with
  `notebooks/_emit.py`. Do not hand-edit the notebook if you plan to emit.
- Patches are under the Jetspace Research License 1.0.

## Workflow

1. Branch from `main`.
2. Make focused commits with a clear reason.
3. Before a pull request:
   - `pip install -e ".[dev,notebooks]"`
   - `python notebooks/_emit.py` if the board changed
   - `pytest -q`
4. Open a pull request with the problem, the proposed change, and the
   identity or test that checks it.

Commercial use is a licensing question, not a pull request. Cite as
Jetspace Research. See [`CITATION.cff`](CITATION.cff).
