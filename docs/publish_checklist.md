# cuvis-ai-dinomaly plugin publish checklist

This checklist prepares `cuvis-ai-dinomaly` as an official plugin release for:

- Repo: `https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly.git`
- Initial tag: `v0.1.0`

## 1) Pre-publish validation

From repo root (`/home/dev/anish/cuvis-ai-dinomaly`):

```bash
uv sync --extra dev
uv run pytest tests -q
uv run pytest tests --cov=cuvis_ai_dinomaly --cov-report=term-missing --cov-fail-under=0
uv run python -c "from cuvis_ai_core.utils.node_registry import NodeRegistry; r=NodeRegistry(); r.load_plugins('examples/plugins.yaml'); print(sorted(r.list_plugins().keys()))"
```

Expected:

- test suite passes
- coverage report is generated
- plugin loads from local manifest and exposes Dinomaly nodes

## 2) Confirm release metadata

- `pyproject.toml` has:
  - `project.name = "cuvis-ai-dinomaly"`
  - `project.version = "0.1.0"`
  - stable public node paths under `cuvis_ai_dinomaly.node.*`
- `.gitignore` excludes:
  - `diagnostics/`
  - `outputs/`
  - `presentation_docs/`

## 3) Create GitHub repository and push

If remote does not exist yet:

```bash
git init
git remote add origin git@github.com:cubert-hyperspectral/cuvis-ai-dinomaly.git
git add .
git commit -m "Prepare cuvis-ai-dinomaly v0.1.0 plugin release"
git branch -M main
git push -u origin main
```

If remote already exists, skip `git init` and `remote add`.

## 4) Tag release

```bash
git tag -a v0.1.0 -m "cuvis-ai-dinomaly v0.1.0"
git push origin v0.1.0
```

## 5) Validate repo+tag plugin install flow

From `cuvis-ai` repo root:

```bash
uv run python -c "from cuvis_ai_core.utils.node_registry import NodeRegistry; r=NodeRegistry(); r.load_plugins('configs/plugins/dinomaly.yaml'); print(sorted(r.list_plugins().keys()))"
```

Expected:

- `DinomalyDetector`
- `DinomalyTrainLossBridge`

## 6) cuvis-ai integration files

Already prepared in `cuvis-ai`:

- `configs/plugins/dinomaly.yaml` (repo + tag manifest)
- `configs/plugins/registry.yaml` (added `dinomaly` entry)

## Notes on local development vs release installs

- `pyproject.toml` keeps local `tool.uv.sources` for internal dev convenience.
- Tagged GitHub plugin installs rely on standard `project.dependencies`.
