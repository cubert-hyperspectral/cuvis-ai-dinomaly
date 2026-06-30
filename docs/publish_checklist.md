# cuvis-ai-dinomaly publish checklist

Pre-release checklist for any `vX.Y.Z` tag. Run all steps from the repo root.

## 1. Pre-publish validation

```bash
# Install all dev deps from public indexes (no local-path overrides)
uv sync --no-sources --extra dev

# Fast test suite — must be green before tagging
uv run --no-sources --extra dev pytest tests/ -m "not slow" -v --tb=short

# Manifest smoke test — confirm NodeRegistry can load the plugin (3 nodes)
uv run --no-sources --extra dev python -c "
from cuvis_ai_core.utils.node_registry import NodeRegistry
r = NodeRegistry()
r.register_plugin('examples/plugins.yaml')
print(sorted(r.list_plugins()))
"
# Expected: ['dinomaly']
```

## 2. Confirm release metadata

- `pyproject.toml`:
  - `project.version = "X.Y.Z"` matches the intended tag
  - `project.name = "cuvis-ai-dinomaly"`
  - `project.license = "Apache-2.0"`
- `CHANGELOG.md`: `## X.Y.Z - YYYY-MM-DD` section exists and is complete (no stale "Unreleased" content)
- `uv.lock`: generated with `--no-sources` so CI can use `--locked`

## 3. Run compatibility audit (skill §8)

Check that every plugin runtime dep that also appears in `cuvis-ai-core`'s `uv.lock` satisfies the plugin's specifier. See `docs/compatibility_audit.md` for the procedure and last recorded results. Re-run if any runtime dep version changed since the last audit.

## 4. Build and validate wheel

```bash
uv build --no-sources
uv run --no-sources --with twine twine check dist/*
```

Verify the wheel name starts with `cuvis_ai_dinomaly-X.Y.Z-`.

## 5. Tag the release

```bash
git tag -a vX.Y.Z -m "cuvis-ai-dinomaly vX.Y.Z"
git push origin vX.Y.Z
```

The `release.yml` workflow fires on the tag push and:
- Validates tests + lint
- Runs security scanning
- Builds the wheel and checks tag == package version
- Creates a GitHub release with the CHANGELOG section as release notes

## 6. Validate git-tag manifest install

After the tag is pushed, confirm the plugin loads from the git-tag manifest (not just local path):

```bash
# From the cuvis-ai repo (or any clean venv with cuvis-ai-core installed)
uv run python -c "
from cuvis_ai_core.utils.node_registry import NodeRegistry
r = NodeRegistry()
r.register_plugin('configs/plugins/dinomaly.yaml')  # uses repo + tag
print(sorted(r.list_plugins()))
"
```

## 7. Update central registry (cuvis-ai repo)

After the git-tag manifest install is verified, update or open a PR against `cuvis-ai`:

- `configs/plugins/registry.yaml` — bump/add the `dinomaly:` tag entry to `vX.Y.Z`
- `configs/plugins/dinomaly.yaml` — bump `tag:` to `vX.Y.Z`
