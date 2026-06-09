# Proposal: `DinomalyDetector(fast_inference=…)` — promote the validated speedup recipe to first-class API

**Status:** approved, deferred until the bedding-all6 PR lands
**Author:** Anish Raj
**Date:** 2026-06-09
**Tracking:** ALL-5732 (parent), follow-up subtask filed separately
**Sequencing:** lands on top of the bedding-all6 PR — branch from `main` once that PR merges
**Estimated size:** ~80 LoC detector changes, ~150 LoC tests, ~30 LoC docstrings, ~50 LoC docs page

## Why

Empirical validation across four pilots (`comparisons/experiments/all6_*`) shows
that **TF32 + bf16 autocast + `torch.compile(mode="reduce-overhead")`** delivers a
**3.6×–8.4× lossless speedup** on DinomalyDetector inference, with score-map
correlation ≥0.9996 vs the fp32 baseline and full-val metrics within ±0.003 on
every measured config. The recipe currently lives only as a benchmark script
(`examples/bedding_dinomaly/benchmark_inference_speedups.py`) — every downstream
user has to copy-paste it. Promoting it to a first-class kwarg makes the speedup
discoverable, default-safe (off), and unit-testable.

### Measured speedup

| Pilot | fp32 (ms/frame) | Fastest (TF32 + bf16 + `torch.compile`) | Speedup | Score corr. vs fp32 |
|---|---:|---:|---:|---:|
| ALL6 high-res 672² | 184.8 | 50.8 | 3.64× | 0.99977 |
| ALL6 aspect 434×1036 | 163.7 | 42.7 | 3.83× | 0.99974 |
| ALL6 aspect 504×1204 | 184.8 | 50.8 | 3.64× | 0.99977 |
| **ALL6 aspect 1260×2996** | **3099.7** | **371.1** | **8.35×** | **0.99967** |

Speedup grows with resolution because `torch.compile` overhead amortises better
over GPU-bound work. Full-val metric drift (verified by
`verify_fast_inference_metrics.py`):

- pixel AUROC: +0.00004
- image AUROC (filename): +0.003
- Dice (raw): +0.0005

Bit-exact reproducibility is **not** preserved (bf16 is lossy by design); metric
drift is well below the noise floor of run-to-run variance.

## API surface

```python
class DinomalyDetector(Node):
    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        # … existing kwargs unchanged …
        fast_inference: bool = False,
        use_tf32: bool | None = None,
        autocast_dtype: torch.dtype | None = None,
        compile_mode: str | None = None,
    ) -> None:
        ...

    def warmup(self, sample_input: torch.Tensor | None = None) -> None:
        """Trigger torch.compile up-front so the first real inference doesn't pay the
        10-30s compile cost. No-op if compile_mode is None or compile already done.

        If sample_input is None, a zero tensor of the configured (B=1, H, W, C)
        shape is fabricated from self.image_size / self.input_channels.
        """
```

### Convenience flag resolves to the validated recipe

```python
if fast_inference:
    self._use_tf32       = True if use_tf32 is None else use_tf32
    self._autocast_dtype = torch.bfloat16 if autocast_dtype is None else autocast_dtype
    self._compile_mode   = "reduce-overhead" if compile_mode is None else compile_mode
else:
    # Power-user knobs still honoured without the umbrella flag.
    self._use_tf32       = bool(use_tf32) if use_tf32 is not None else False
    self._autocast_dtype = autocast_dtype  # may be None or any torch dtype
    self._compile_mode   = compile_mode    # may be None or any compile-mode string

self._compiled = False  # lazy
```

### Forward path (guarded)

```python
def forward(self, rgb_image, context):
    if self._use_tf32:
        torch.set_float32_matmul_precision("high")  # idempotent, no-op if already set

    # Compile ONLY on inference, ONLY once, ONLY if requested
    if (self._compile_mode
        and not self._compiled
        and context.stage in (ExecutionStage.INFERENCE, ExecutionStage.VALIDATION)):
        self.dinomaly_model = torch.compile(self.dinomaly_model, mode=self._compile_mode)
        self._compiled = True
        logger.info("DinomalyDetector: torch.compile applied (mode={})", self._compile_mode)

    ctx = (torch.autocast(device_type="cuda", dtype=self._autocast_dtype)
           if self._autocast_dtype is not None else contextlib.nullcontext())
    with ctx:
        ...  # existing forward body
```

### `warmup()` implementation

```python
def warmup(self, sample_input: torch.Tensor | None = None) -> None:
    """Pre-compile the model so the first inference call doesn't pay compile cost."""
    if not self._compile_mode or self._compiled:
        return
    if sample_input is None:
        h, w = self.image_size if isinstance(self.image_size, tuple) else (self.image_size,) * 2
        sample_input = torch.zeros(1, h, w, self.input_channels,
                                    device=next(self.dinomaly_model.parameters()).device)
    # Use a synthetic INFERENCE context so the guard above fires
    from cuvis_ai_schemas.execution import Context
    from cuvis_ai_schemas.enums import ExecutionStage
    ctx = Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0, global_step=0)
    with torch.inference_mode():
        self.forward(sample_input, context=ctx)
    logger.info("DinomalyDetector: warmup complete (compile cost amortised)")
```

Discovery use: call `detector.warmup()` once after `pipeline.load()` and before
the first real batch — predictable latency from the first inference onward.

## Backward compatibility

| Invariant | Verified by |
|---|---|
| New kwargs default to `False`/`None` → existing call sites byte-identical | type-level (default values) |
| Default forward path = zero new ops (no compile, no autocast, no TF32 mutation) | `test_default_kwargs_are_off` |
| Saved pipeline YAML without these keys → `load_pipeline()` works | `test_yaml_roundtrip_legacy` |
| Saved pipeline YAML with these keys → newer cuvis-ai-dinomaly loads cleanly | `test_yaml_roundtrip_with_new_kwargs` |
| Training path untouched — compile gated on `INFERENCE`/`VALIDATION` only | `test_compile_only_at_inference` |
| Existing 71/71 pytest pass unchanged | full `tests/` run in CI |

**Process-wide side effect:** `torch.set_float32_matmul_precision("high")` mutates
global state. Documented in the docstring as "this affects all torch matmuls in
the process, not just this detector". For most use cases this is what users want;
for paranoid users, `use_tf32=False, fast_inference=True` skips just that step.

## Edge cases to test (and the corresponding test names)

| Risk | Test |
|---|---|
| `torch.compile` + monkey-patched `get_encoder_decoder_outputs` (rectangular input) silently caches the original method via FX tracing | `test_rectangular_input_under_compile` — build rect detector + `fast_inference=True`, forward on `(1, 28, 56, 3)`, assert scores shape `(1, 28, 56, 1)` |
| bf16 autocast on pre-Ampere GPU (T4 / V100) crashes | `test_bf16_unsupported_raises` — mock `torch.cuda.is_bf16_supported()=False`, expect informative `RuntimeError` at `__init__` |
| Variable input shape triggers recompile (silent perf regression) | `test_compile_warns_on_shape_change` — first forward at one shape, second at another → log a warning |
| Compile applied during training step would break gradients | `test_compile_only_at_inference` — forward at `ExecutionStage.TRAINING` → `_compiled` stays False; INFERENCE → True after one call |
| Lazy compile applied twice would re-wrap and double-compile | `test_compile_is_idempotent` |
| `warmup()` called when `compile_mode=None` should no-op silently | `test_warmup_noop_without_compile` |
| `warmup()` should make first real forward cheap | `test_warmup_eliminates_first_call_cost` (timing assertion; tolerant) |
| `fast_inference=True, autocast_dtype=torch.float16` honours the override (fp16, not bf16) | `test_explicit_knobs_override_umbrella` |
| Saved pipeline + new kwargs round-trip preserves values | `test_yaml_roundtrip_with_new_kwargs` |
| Metric drift on the saved 1260×2996 pipeline ≤ ±0.005 | `test_metric_drift_under_fp32_baseline` — load saved pipeline + 5 val frames, compare `roc_auc_score` pixel + image |

## Open design questions (answer before implementation)

1. **TF32 unset on detector deletion?** Probably no — `torch.set_float32_matmul_precision` doesn't expose a "previous value", and other models in the pipeline might depend on the change. Document as one-way.
2. **`fast_inference` during training — silent downgrade or hard error?**
   - Lean toward: log a one-time `WARNING` at first training forward saying "fast_inference is ignored during training; autocast + TF32 + compile only active at INFERENCE/VALIDATION". Forward path proceeds at fp32 — no surprise behaviour.
3. **Expose a `compile_options: dict | None` kwarg for passing `dynamic=True` etc.?**
   - Defer to a follow-up if needed. Power users can subclass DinomalyDetector for now.

## Cross-cutting risks

- **`torch.compile` + `setattr` monkey-patch interaction** (the rectangular-input patch). FX tracing snapshots the method bound at trace time. If the patch is applied **after** compile, the patched method is invisible. Mitigation: `_rectangular_input_patch.py` is applied in `DinomalyDetector.__init__` before any forward; compile happens on first forward. So the order is correct — but `test_rectangular_input_under_compile` must verify this empirically, not just by construction.
- **`torch.compile` overhead during CI.** Compile takes 10–30 s. CI tests should either (a) mock the compile call and verify the wrapper is applied, or (b) skip the actually-compiled tests with `@pytest.mark.slow` and only check the structural behaviour by default. We already have `not slow and not gpu` in the default pytest selection — good.
- **bf16 numerical drift across drivers.** Different CUDA / cuDNN versions can produce slightly different bf16 outputs. The ≤±0.005 metric drift assertion gives 5× the empirically-observed worst case (0.001 image AUROC drift on 1260×2996). Should hold across the supported driver matrix.

## Documentation deliverables

1. **Docstring** on `DinomalyDetector.__init__` and `warmup()` — usage example, kwarg interactions, BC note.
2. **New page** `docs/inference_speedup.md` — when to use which knob, the validated recipe, the measured numbers from the four pilots, the BC contract.
3. **Mention in main README** — one line under "Performance" pointing at the new page.
4. **PR description** references this proposal doc + the bedding pilot inference reports as the empirical basis.

## Reproduce the validation numbers

```bash
cd /home/dev/anish/cuvis-ai-dinomaly

# Speedup ladder (4 configs × N frames, GPU clean)
.venv/bin/python /home/dev/anish/cuvis-ai-cookbook/examples/bedding_dinomaly/benchmark_inference_speedups.py \
    --pipeline-yaml <pilot>/trained_models/dinomaly_bedding_all6.yaml \
    --pipeline-pt   <pilot>/trained_models/dinomaly_bedding_all6.pt \
    --splits-csv    /mnt/data/bedding_dataset_npz/bedding_splits_npz.csv \
    --num-images 10 --warmup-images 2 \
    --out-md inference_speedup_benchmark.md

# Full-val metric verification (proves the recipe is lossless)
.venv/bin/python /home/dev/anish/cuvis-ai-cookbook/examples/bedding_dinomaly/verify_fast_inference_metrics.py \
    --pipeline-yaml <pilot>/trained_models/dinomaly_bedding_all6.yaml \
    --pipeline-pt   <pilot>/trained_models/dinomaly_bedding_all6.pt \
    --splits-csv    /mnt/data/bedding_dataset_npz/bedding_splits_npz.csv \
    --out-md fast_inference_verification.md
```

The benchmark + verify scripts will become the regression harness once the kwargs land — re-running them after the implementation should reproduce the same ms/frame and correlation numbers.
