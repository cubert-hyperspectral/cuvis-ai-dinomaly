"""Per-channel stats over all 193 bedding train NPZs.

Validates the EAD-published stats (in BEDDING_DATASET_REPORT.md §2) against
the actual training data we converted. Two parallel computations are run:

  - "ours"  = stats on cubes as stored in NPZ (divided by 10000, NO 0.55 factor)
              → these are the stats Dinomaly should use, since our converter
                drops the EAD-specific white-target factor.
  - "ead"   = stats with the 0.55 factor reapplied                             # reflectance = cube_npz * 0.55
              → directly comparable to EAD's published values.

Output: prints a comparison table + writes
`/mnt/data/bedding_dataset_npz/train_channel_stats.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NPZ_DIR = Path("/mnt/data/bedding_dataset_npz/train")
OUT = Path("/mnt/data/bedding_dataset_npz/train_channel_stats.json")
WAVELENGTHS = [450, 550, 625, 1050, 1200, 1450]
EAD_PUBLISHED_MEAN = np.array(
    [0.31257936, 0.43269954, 0.50889452, 0.64886759, 0.59585480, 0.39783356]
)
EAD_PUBLISHED_STD = np.array(
    [0.13495414, 0.17394394, 0.19844727, 0.21376508, 0.19457226, 0.13772688]
)


def main():
    files = sorted(NPZ_DIR.glob("*.npz"))
    print(f"Found {len(files)} train NPZs")
    C = 6
    n_total = 0
    s = np.zeros(C, dtype=np.float64)
    mn = np.full(C, np.inf, dtype=np.float64)
    mx = np.full(C, -np.inf, dtype=np.float64)
    # For percentiles, sample a fixed number of pixels per cube
    SAMPLE_PER_CUBE = 200_000
    sampled: list[np.ndarray] = []
    rng = np.random.default_rng(42)

    # PASS 1 — accumulate sum + min/max + percentile sample
    for i, f in enumerate(files, 1):
        with np.load(f) as z:
            cube = z["cube"]  # (1800, 4300, 6) float32, already / 10000
        flat = cube.reshape(-1, C).astype(np.float64, copy=False)
        n = flat.shape[0]
        s += flat.sum(axis=0)
        mn = np.minimum(mn, flat.min(axis=0))
        mx = np.maximum(mx, flat.max(axis=0))
        n_total += n
        idx = rng.choice(n, size=min(SAMPLE_PER_CUBE, n), replace=False)
        sampled.append(flat[idx].astype(np.float32))
        if i % 20 == 0 or i == len(files):
            print(f"  [PASS1 {i:>3}/{len(files)}] running mean={s/n_total}")

    mean = s / n_total

    # PASS 2 — accumulate sum of squared deviations (avoids catastrophic cancellation)
    ssd = np.zeros(C, dtype=np.float64)
    for i, f in enumerate(files, 1):
        with np.load(f) as z:
            cube = z["cube"]
        flat = cube.reshape(-1, C).astype(np.float64, copy=False)
        ssd += ((flat - mean) ** 2).sum(axis=0)
        if i % 40 == 0 or i == len(files):
            print(f"  [PASS2 {i:>3}/{len(files)}] running std={np.sqrt(ssd/n_total)}")

    var = ssd / n_total
    std = np.sqrt(var)

    sampled_all = np.concatenate(sampled, axis=0)
    p1 = np.percentile(sampled_all, 1, axis=0)
    p99 = np.percentile(sampled_all, 99, axis=0)
    p50 = np.percentile(sampled_all, 50, axis=0)

    # The "EAD scale" view: cube_npz * 0.55 to compare with EAD's published stats
    ead_mean = mean * 0.55
    ead_std = std * 0.55
    ead_p1 = p1 * 0.55
    ead_p99 = p99 * 0.55

    print("\n=== Per-channel stats: 'ours' (cube_npz, factor 1/10000 only) ===")
    print(f"{'wl':>5} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'p1':>10} {'p50':>10} {'p99':>10}")
    for i, wl in enumerate(WAVELENGTHS):
        print(f"{wl:>5} {mean[i]:>10.4f} {std[i]:>10.4f} {mn[i]:>10.4f} {mx[i]:>10.4f} "
              f"{p1[i]:>10.4f} {p50[i]:>10.4f} {p99[i]:>10.4f}")

    print("\n=== Comparison vs EAD published (apply 0.55 to ours) ===")
    print(f"{'wl':>5} {'mean_ours*.55':>14} {'mean_pub':>10} {'Δmean':>10} {'std_ours*.55':>14} {'std_pub':>10} {'Δstd':>10}")
    for i, wl in enumerate(WAVELENGTHS):
        print(f"{wl:>5} {ead_mean[i]:>14.5f} {EAD_PUBLISHED_MEAN[i]:>10.5f} {ead_mean[i]-EAD_PUBLISHED_MEAN[i]:>+10.5f} "
              f"{ead_std[i]:>14.5f} {EAD_PUBLISHED_STD[i]:>10.5f} {ead_std[i]-EAD_PUBLISHED_STD[i]:>+10.5f}")

    out = {
        "n_files": len(files),
        "n_pixels_total": int(n_total),
        "wavelengths": WAVELENGTHS,
        "ours_no_factor": {
            "mean": mean.tolist(), "std": std.tolist(),
            "min": mn.tolist(), "max": mx.tolist(),
            "p1": p1.tolist(), "p50": p50.tolist(), "p99": p99.tolist(),
        },
        "ead_scale_with_055": {
            "mean": ead_mean.tolist(), "std": ead_std.tolist(),
            "p1": ead_p1.tolist(), "p99": ead_p99.tolist(),
        },
        "ead_published": {
            "mean": EAD_PUBLISHED_MEAN.tolist(),
            "std": EAD_PUBLISHED_STD.tolist(),
        },
        "delta_vs_published": {
            "mean": (ead_mean - EAD_PUBLISHED_MEAN).tolist(),
            "std": (ead_std - EAD_PUBLISHED_STD).tolist(),
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
