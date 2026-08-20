# cuvis-ai-dinomaly publish checklist

Pre-release checklist for any `vX.Y.Z` tag. Run all steps from the repo root.

## 1. Pre-publish validation

```bash
# Install all dev deps from public indexes (no local-path overrides)
uv sync --no-sources --extra dev

# Fast test suite — must be green before tagging
uv run --no-sources --extra dev pytest tests/ -m "not slow" -v --tb=short

# Manifest smoke test — confirm NodeRegistry can load the plugin (5 nodes)
uv run --no-sources --extra dev python -c "
from cuvis_ai_core.utils.node_registry import NodeRegistry
r = NodeRegistry()
r.register_plugin('configs/plugins/dinomaly.yaml')
print(sorted(r.list_plugins()))
"
# Expected: ['dinomaly']
```

## 2. Confirm release metadata

- `pyproject.toml`:
  - version is setuptools-scm dynamic: the tag IS the version (no `project.version` to edit)
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

Releases are manual — no workflow fires on the tag push (the old `release.yml` was removed in
0.1.5; the plugin is distributed via git tags referenced from cuvis-ai plugin manifests). After
pushing the tag, create the GitHub release yourself with the matching CHANGELOG section as notes:

```bash
gh release create vX.Y.Z --title vX.Y.Z --notes-file <extracted CHANGELOG section> --verify-tag
```

## 6. Validate git-tag manifest install

After the tag is pushed, confirm the plugin loads from the git-tag manifest (not just local path):

```bash
# Run from a checkout of the cuvis-ai repo (or any clean venv with cuvis-ai-core installed).
# Note: this loads the CUVIS-AI repo's configs/plugins/dinomaly.yaml (repo + tag form), which is
# a different file from this repo's local-path configs/plugins/dinomaly.yaml.
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
