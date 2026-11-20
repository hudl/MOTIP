"""Mean / std of the kit descriptors of a DETR cache built with ``build_detr_cache.py --kit`` -> ``<cache>/kit_stats.json``.

    python scripts/compute_kit_stats.py <cache dir> [--out kit_stats.json]

TEAM_CUE training standardises the descriptor with these (they are copied into the model as buffers and travel with the checkpoint).
Also prints the coverage (rows with no descriptor) and the per-dimension spread.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.kit_feature import KIT_DIM, KIT_VERSION  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.cache_dir, "**", "*.npz"), recursive=True))
    assert files, f"no .npz under {a.cache_dir}"
    n = 0
    s1 = np.zeros(KIT_DIM, np.float64)
    s2 = np.zeros(KIT_DIM, np.float64)
    n_rows = n_nan = 0
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            assert "kit" in z.files, f"{f} has no kit (build with --kit)"
            k = z["kit"].astype(np.float64)
        ok = np.isfinite(k).all(axis=1)
        n_rows += len(k)
        n_nan += int((~ok).sum())
        s1 += k[ok].sum(0)
        s2 += (k[ok] ** 2).sum(0)
        n += int(ok.sum())
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-8))
    out = {"version": KIT_VERSION, "dim": KIT_DIM, "n_rows": n_rows, "n_valid": n, "nan_frac": n_nan / max(n_rows, 1),
           "mean": mean.tolist(), "std": std.tolist(), "files": len(files)}
    path = a.out or os.path.join(a.cache_dir, "kit_stats.json")
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(f"{len(files)} sequences, {n_rows:,} rows, {n_nan:,} without descriptor ({100 * out['nan_frac']:.2f}%) -> {path}")
    print("mean", np.round(mean, 3).tolist())
    print("std ", np.round(std, 3).tolist())


if __name__ == "__main__":
    main()
