#!/usr/bin/env python3
"""Post-hoc physical-unit audit for a completed strict-DC3 checkpoint.

This script does not train, select, or alter a model.  It restores the final
checkpoint written by the official DC3 loop, applies its *unchanged* upstream
test-time correction routine, then calls the common Fair4Net scorer used by
the DEQ/baseline experiments.  The output therefore uses MW and native
generator-cost units rather than DC3's internally scaled objective.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import torch
import torch.nn.functional as F

from common import UPSTREAM_COMMIT, import_upstream
from fair4net_adapter import Fair4NetDC3Data, Fair4NetNNSolver, load_fair4net_helpers


@torch.no_grad()
def completion_error_breakdown(prepared, prediction: torch.Tensor) -> dict:
    """Separate audit-only error of DC3's completed and direct outputs.

    Solver labels are read here only after training to report RMSE.  They are
    never provided to DC3's forward pass, objective, constraints, or updates.
    """
    idx = torch.as_tensor(prepared.idx["test"], device=prediction.device, dtype=torch.long)
    truth = prepared.pg[idx].to(dtype=prediction.dtype)
    complete = prepared.completion_gen[idx]
    rows = torch.arange(idx.numel(), device=prediction.device)
    completed_mask = F.one_hot(complete, num_classes=prediction.shape[1]).bool()
    direct_mask = ~completed_mask
    squared = (prediction - truth).square()

    upper = (prediction - prepared.pmax_eff[idx].to(prediction.dtype)).clamp_min(0.0)
    lower = (prepared.pmin[idx].to(prediction.dtype) - prediction).clamp_min(0.0)
    bound = torch.maximum(upper, lower)

    total_squared = squared.sum().clamp_min(1e-30)
    return {
        "completed_generator_rmse_MW": float(torch.sqrt(squared[completed_mask].mean())),
        "direct_generators_rmse_MW": float(torch.sqrt(squared[direct_mask].mean())),
        "completed_generator_squared_error_share": float(squared[completed_mask].sum() / total_squared),
        "completed_generator_bound_violation_mean_MW": float(bound[completed_mask].mean()),
        "completed_generator_bound_violation_max_MW": float(bound[completed_mask].max()),
        "direct_generators_bound_violation_mean_MW": float(bound[direct_mask].mean()),
        "direct_generators_bound_violation_max_MW": float(bound[direct_mask].max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--checkpoint", required=True, type=pathlib.Path)
    ap.add_argument("--protocol", required=True, choices=("clean", "mixed"))
    ap.add_argument("--case", default="case300")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--hidden-size", type=int, default=200,
                    help="hidden width used to train the checkpoint")
    ap.add_argument("--corr-test-max-steps", type=int, default=5,
                    help="DC3 test correction cap used by the checkpoint experiment")
    args = ap.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    method, _, defaults_module = import_upstream()
    os.environ["DEQ_CASE"] = args.case
    helpers = load_fair4net_helpers()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prepared = helpers.prepare(args.data, device, args.protocol)
    data = Fair4NetDC3Data(prepared, helpers)

    # Recreate the same upstream configuration used for training.  The only
    # model substitution remains the documented DC problem interface.
    config = defaults_module.method_default_args("acopf57")
    config["probType"] = "acopf57"
    config["hiddenSize"] = args.hidden_size
    config["corrTestMaxSteps"] = args.corr_test_max_steps
    method.NNSolver = Fair4NetNNSolver
    model = Fair4NetNNSolver(data, config).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # This is the authors' `grad_steps_all`: adaptive stopping up to their
    # corrTestMaxSteps=5.  It is deliberately not a new feasibility repair.
    with torch.no_grad():
        pred_raw = model(data.testX)
        pred_corrected, correction_steps = method.grad_steps_all(
            data, data.testX, pred_raw, config
        )

    # Exact same physical scorer used in the Fair4Net baseline reports.
    metrics, sample_rows = helpers.score_predictions(
        prepared, pred_corrected.float(), prepared.idx["test"]
    )
    raw_metrics, _ = helpers.score_predictions(
        prepared, pred_raw.float(), prepared.idx["test"]
    )

    payload = {
        "status": "PASS",
        "purpose": "posthoc physical-unit audit; no training or model selection",
        "upstream_commit": UPSTREAM_COMMIT,
        "dataset": str(args.data.resolve()),
        "protocol": args.protocol,
        "checkpoint": str(args.checkpoint.resolve()),
        "test_samples": int(len(sample_rows)),
        "test_correction_steps": int(correction_steps),
        "test_correction_max_steps": int(config["corrTestMaxSteps"]),
        "test_correction_tolerance": float(config["corrEps"]),
        "metrics_after_official_dc3_correction": metrics,
        "metrics_before_correction": raw_metrics,
        "completion_generator_breakdown_after_correction": completion_error_breakdown(
            prepared, pred_corrected.float()
        ),
        "metric_units": {
            "rmse_MW": "MW",
            "bal_abs_MW": "MW",
            "viol_MW": "MW; mean sample-worst branch overload",
            "gen_bound_viol_MW": "MW",
            "cost_gap_abs_mean": "native PGLib generator-cost objective units",
        },
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
