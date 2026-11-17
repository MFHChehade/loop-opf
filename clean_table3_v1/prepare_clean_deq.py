#!/usr/bin/env python3
"""Subset audited mixed data to clean rows for a clean-trained DEQ run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED = {
    "case300": ("c35eb18638cf585751f8d27e82cfb30c47e449d906d293a23756b97fbc6de50b", (1780, 244, 513)),
    "case1354pegase": ("1017ba4733ad623058a18d0847e61ed65b14abb7df64f39edc4a3bbc8719177e", (1721, 245, 476)),
    "case2869pegase": ("aaf605cde245af51b9d9616348cfd9953042c46181557f39dc0348ff6f89c343", (1731, 233, 505)),
    "case9241pegase": ("55c1fab63d17b9166f29b1f8d58288ce34ef490ee6a3b9229fef022c1de7e374", (1725, 248, 489)),
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=EXPECTED)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    expected_hash, counts = EXPECTED[args.case]
    actual_hash = sha256(args.source)
    if actual_hash != expected_hash:
        raise ValueError(f"source SHA mismatch: {actual_hash}")
    with np.load(args.source, allow_pickle=False) as z:
        if not {"split", "contingency_type", "pd_bus", "pg", "gs_bus", "pd_bus_includes_gs"}.issubset(z.files):
            raise ValueError("required source arrays missing")
        n = z["split"].size
        keep = z["contingency_type"] == 0
        if tuple(int(np.sum(keep & (z["split"] == i))) for i in range(3)) != counts:
            raise ValueError("clean split counts disagree with audited counts")
        if np.any(z["cont_line_idx"][keep] >= 0) or np.any(z["cont_gen_idx"][keep] >= 0):
            raise ValueError("clean rows contain outage metadata")
        arrays = {k: z[k][keep] if z[k].ndim and z[k].shape[0] == n else z[k] for k in z.files}
    if np.any(arrays["contingency_type"] != 0):
        raise AssertionError("nonclean row retained")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    with np.load(args.output, allow_pickle=False) as z:
        for k, source_arr in arrays.items():
            if not np.array_equal(z[k], source_arr):
                raise ValueError(f"round-trip mismatch: {k}")
    manifest = {
        "status": "PASS", "case": args.case, "source": str(args.source.resolve()),
        "source_sha256": actual_hash, "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output), "selection": "contingency_type == 0; preserve original split",
        "split_counts": dict(zip(("train", "validation", "test"), counts)),
        "output_rows": int(np.sum(keep)), "outage_indices": "all negative on retained rows",
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
