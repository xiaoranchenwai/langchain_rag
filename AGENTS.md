# Repository Guidelines

## Project Structure & Module Organization

This repository is currently a flat collection of Python RAG prototypes at the repo root. `langchain_rag.py` is the simplest baseline. The `rag_base*.py` files are experiment variants for different retrieval strategies, including parent-child retrieval, BM25/ensemble retrieval, reranking, agent-based flows, and Excel-oriented loaders. There is no `src/` package yet, so keep related logic grouped by feature and prefer adding new reusable code in small helper modules rather than making existing scripts longer.

## Build, Test, and Development Commands

There is no committed build system or lockfile, so contributors currently run scripts directly.

- `python langchain_rag.py`: run the basic LangChain RAG example.
- `python rag_base_index_EnsembleRetriever.py`: run an indexed retrieval variant.
- `python -m py_compile *.py`: quick syntax validation for all root scripts.
- `pytest`: preferred test runner once `tests/` is added.

If you add dependency or task tooling, document it in the same change.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and follow PEP 8. Functions and variables should use `snake_case`; classes should use `PascalCase`; constants should use `UPPER_SNAKE_CASE`. Keep one retrieval experiment per file and use descriptive suffixes such as `_agent`, `_rerank`, or `_7b` to reflect behavior. Prefer explicit helper functions over top-level side effects when touching existing scripts.

## Testing Guidelines

There is no formal test suite in the repo yet, so new work should start adding one under `tests/` with files named `test_<module>.py`. Use `pytest` for unit and integration coverage, and target at least 80% coverage for any new module you introduce. Until that exists, validate changes with `python -m py_compile *.py` and at least one manual run against a representative document set.

## Commit & Pull Request Guidelines

The current history is minimal and uses short descriptive subjects, for example `知识库问答优化记录`. Keep commits focused and readable, one concern per commit. Pull requests should explain which retrieval variant changed, any new model endpoint or document path assumptions, manual verification steps, and sample output when answer quality changes.

## Security & Configuration Tips

Do not commit API keys, internal service URLs, or machine-specific absolute paths. Several existing scripts contain environment-specific endpoints and local paths; treat those as legacy and move new configuration into environment variables or clearly documented constants before merging.
