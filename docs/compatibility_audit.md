# Compatibility audit

Per the cuvis-ai plugin skill §8, every plugin runtime dependency that also appears in `cuvis-ai-core`'s `uv.lock` must be satisfiable by core's locked version. Mismatches produce in-process upgrade attempts at `LoadPlugins` time which can break a consumer's venv (e.g. Windows native-extension `Access is denied` failures with Pillow / OpenCV / lxml).

This file records the result for each pre-tag run.

## 2026-05-08 — pre-`v0.1.4` audit

- **Plugin commit audited:** `96afb2d` (branch `ci/add-workflow`, tip of [PR #1](https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly/pull/1)).
- **Target core versions:** `cuvis-ai-core` `v0.1.0` (declared floor) and `v0.5.2` (latest released).
- **Lock snapshots:**
  - https://raw.githubusercontent.com/cubert-hyperspectral/cuvis-ai-core/v0.1.0/uv.lock
  - https://raw.githubusercontent.com/cubert-hyperspectral/cuvis-ai-core/v0.5.2/uv.lock
- **Procedure:** ad-hoc Python script using `tomllib` + `packaging.specifiers.SpecifierSet.contains(version, prereleases=True)`; PEP 503 name normalisation. Host packages (`cuvis-ai-core`, `cuvis-ai-schemas`) excluded.

### Per-dep verdict

| dep | plugin spec | core 0.1.0 lock | core 0.5.2 lock | verdict |
|---|---|---|---|---|
| `anomalib` | `==2.1.0` | not in lock | not in lock | OK |
| `kornia` | `==0.6.12` | not in lock | not in lock | OK |
| `numpy` | `>=1.20.0` | `2.4.1` | `2.4.1` | OK |
| `opencv-python` | `>=4.8.0` | not in lock | not in lock | OK |
| `tqdm` | `>=4.66.0` | `4.67.1` | `4.67.1` | OK |
| `defusedxml` | `>=0.7.1` | `0.7.1` | `0.7.1` | OK |
| `open-clip-torch` | `>=2.24.0` | not in lock | not in lock | OK |
| `requests` | `>=2.31.0` | `2.32.5` | `2.32.5` | OK |

**Overall: PASS** against both target core versions. No accommodations needed.

### Notes

- `anomalib`, `kornia`, `opencv-python`, and `open-clip-torch` are not transitive deps of either core release. They are installed fresh by the plugin loader, so there is no version-conflict risk regardless of the plugin's pin.
- `numpy`, `tqdm`, `defusedxml`, and `requests` *are* in both core locks; the locked versions satisfy the plugin's lower-bound specifiers in every case.
- Both core releases use the same locked versions for the four shared deps — no drift to worry about between them.
