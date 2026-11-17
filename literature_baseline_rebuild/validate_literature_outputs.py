#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METHODS = ("deepopf", "dc3", "deeplde", "pdl", "canos")
CASES = ("case300", "case1354pegase", "case2869pegase", "case9241pegase")
PROTOCOLS = ("clean", "mixed")
PROTOCOL_VERSION = "literature_baselines_dc_v3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    ap.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    ap.add_argument("--require-full", action="store_true")
    ap.add_argument(
        "--grouped-smoke",
        action="store_true",
        help="read all five smoke results from each case/protocol/all directory",
    )
    args = ap.parse_args()
    missing = []
    checked = 0
    expected_total = len(args.cases) * len(args.protocols) * len(METHODS)
    for case in args.cases:
        for protocol in args.protocols:
            for method in METHODS:
                folder_name = "all" if args.grouped_smoke else method
                folder = args.root / case / protocol / folder_name
                paths = list(folder.glob(f"{method}_{protocol}_seed0.json"))
                if not paths:
                    missing.append(f"{case}/{protocol}/{method}")
                    continue
                d = json.loads(paths[0].read_text(encoding="utf-8"))
                assert d["case"] == case and d["method"] == method and d["protocol"] == protocol
                assert d["protocol_version"] == PROTOCOL_VERSION
                assert d["test_outage_identity_policy"] == "disjoint from training for mixed protocol"
                expected_labels = method in {"deepopf", "canos"}
                assert bool(d["training_uses_solver_dispatch_labels"]) == expected_labels
                if int(d["training_metadata"].get("optimizer_steps", 0)) <= 0:
                    raise ValueError(f"{paths[0]} records no optimizer updates")
                expected_equality = (
                    "differentiable all-generator headroom repair" if method == "pdl" else
                    "single available slack-priority generator analytic completion"
                )
                assert d["equality_handling"] == expected_equality
                stem = paths[0].stem
                if not (folder / f"{stem}.pt").is_file():
                    missing.append(f"{case}/{protocol}/{method}:checkpoint")
                    continue
                sample_path = folder / f"{stem}_per_sample.csv"
                if not sample_path.is_file():
                    missing.append(f"{case}/{protocol}/{method}:per_sample")
                    continue
                with sample_path.open(newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    required_columns = {
                        "source_index", "scenario", "cont_line_idx", "cont_gen_idx",
                        "cost_gap_raw", "cost_gap_abs", "rmse_MW",
                        "balance_abs_MW", "gen_bound_violation_MW",
                        "flow_violation_MW", "completion_gen_idx",
                    }
                    if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                        raise ValueError(f"{sample_path} missing per-sample columns")
                    sample_rows = list(reader)
                if args.require_full:
                    assert not d.get("smoke_limit", False)
                    selection = d.get("model_selection", {})
                    assert selection.get("split") == "validation"
                    assert math.isfinite(float(selection.get("best_score", float("nan"))))
                    for suffix in ("_training_state.pt", "_best_validation.pt"):
                        if not (folder / f"{stem}{suffix}").is_file():
                            missing.append(f"{case}/{protocol}/{method}:{suffix[1:]}")
                            continue
                if args.grouped_smoke:
                    assert d.get("smoke_limit", False)
                    assert d["split_counts"] == {"train": 84, "val": 12, "test": 24}
                m = d["metrics"]["all"]
                for key in (
                    "cost_gap_raw_mean", "cost_gap_abs_mean", "rmse_MW",
                    "viol_MW", "gen_bound_viol_MW", "bal_abs_MW",
                ):
                    if not math.isfinite(float(m[key])):
                        raise ValueError(f"{paths[0]} nonfinite {key}")
                if protocol == "mixed":
                    assert all(int(d["metrics"][name]["samples"]) > 0 for name in ("clean", "line", "generator"))
                else:
                    assert int(d["metrics"]["clean"]["samples"]) == int(m["samples"])
                    assert int(d["metrics"]["line"]["samples"]) == 0
                    assert int(d["metrics"]["generator"]["samples"]) == 0
                assert len(sample_rows) == int(m["samples"])
                checked += 1
    print(f"[validate-literature] checked={checked}/{expected_total} missing={len(missing)}")
    for item in missing:
        print(f"MISSING {item}")
    if missing:
        raise SystemExit(1)
    print("[validate-literature] PASS")


if __name__ == "__main__":
    main()
