#!/usr/bin/env python3
"""Supervised DeepOPF-SC-architecture adaptation for preventive line K=32.

This predicts one preventive base dispatch for all declared line outages. It
uses the published DeepOPF-SC network architecture, but the balanced output,
all-contingency loss, and optional L1 feasibility projection are adaptations
to this audited IEEE-300 Phase A contract, not an upstream reproduction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deepopf_scdcopf_official.data import FeatureContract, load_exact_phase_a_dataset
from deepopf_scdcopf_official.metrics import evaluate_predictions
from deepopf_scdcopf_official.model import OfficialDeepOPF
from deepopf_scdcopf_official.physics import PreventiveLineSCPhysics
from deepopf_scdcopf_official.projection import L1FeasibilityProjector
from phase_a_scopf.common import sha256_file
from train_constraint_driven_scopf import predict, save_json, save_torch, test_predictions

PROTOCOL = "supervised_deepopf_sc_adaptation_preventive_line_k32_v1"
TOL_MW = 1e-3


def loss_value(model, physics, features, pd, target, label_scale, security_weight):
    predicted = predict(model, physics, features, pd)
    mse = ((predicted - target) / label_scale[None, :]).square().mean()
    flow = physics.all_flows(predicted, pd)
    overload = ((flow.abs() - physics.fmax[None, None, :]) /
                physics.fmax[None, None, :].clamp_min(1.0)).clamp_min(0.0)
    overload = overload * physics.flow_mask[None, :, :]
    security = overload.amax(dim=(1, 2)).square().mean()
    return mse + security_weight * security


@torch.no_grad()
def validation(model, physics, features, pd_all, target_all, ids, label_scale,
               weight, batch, device):
    model.eval()
    total = 0.0
    for begin in range(0, len(ids), batch):
        row = ids[begin:begin + batch]
        pd = pd_all[row].to(device)
        target = target_all[row].to(device)
        loss = loss_value(model, physics, features, pd, target, label_scale,
                          weight)
        total += float(loss.item()) * len(row)
    return total / len(ids)


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data_path = args.data.resolve()
    data = load_exact_phase_a_dataset(data_path)
    split = np.asarray(data["split"], dtype=np.int64)
    indices = {name: np.flatnonzero(split == value)
               for name, value in (("train", 0), ("val", 1), ("test", 2))}
    if args.smoke:
        indices = {name: ids[:count] for name, ids, count in
                   (("train", indices["train"], 70),
                    ("val", indices["val"], 10),
                    ("test", indices["test"], 20))}
    features = FeatureContract.fit(data["pd_bus"], split)
    physics = PreventiveLineSCPhysics(data, dtype=torch.float32).to(device)
    model = OfficialDeepOPF(len(features.buses), args.hidden, physics.ngen).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pd_all = torch.as_tensor(data["pd_bus"], dtype=torch.float32)
    target_all = torch.as_tensor(data["pg_scopf"], dtype=torch.float32)
    label_scale = target_all[indices["train"]].std(dim=0).clamp_min(1.0).to(device)
    best = math.inf
    best_epoch = 0
    best_state = None
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(indices["train"])
        epoch_losses = []
        for begin in range(0, len(order), args.batch_size):
            ids = order[begin:begin + args.batch_size]
            pd = pd_all[ids].to(device)
            target = target_all[ids].to(device)
            loss = loss_value(model, physics, features, pd, target, label_scale,
                              args.security_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite supervised SC-OPF loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().item()))
        val = validation(model, physics, features, pd_all, target_all,
                         indices["val"], label_scale, args.security_weight,
                         args.eval_batch_size, device)
        if not math.isfinite(val):
            raise FloatingPointError("nonfinite SC-OPF validation loss")
        history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)),
                        "validation_loss": val})
        if val < best:
            best = val
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            save_torch(args.output_dir / "selected.pt", {
                "protocol": PROTOCOL, "dataset_sha256": sha256_file(data_path),
                "security_set_digest": data["provenance"]["configuration"]["security_set_digest"],
                "model_state": best_state, "feature_contract": features.payload(),
                "hidden": args.hidden, "validation_loss": best, "epoch": best_epoch,
            })
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps({"epoch": epoch, "validation_loss": val,
                              "best_validation_loss": best}), flush=True)
    train_sec = time.perf_counter() - started
    model.load_state_dict(best_state)
    raw, inference_sec = test_predictions(model, physics, features, pd_all,
                                           indices["test"], args.eval_batch_size, device)
    selected = torch.load(args.output_dir / "selected.pt", map_location=device,
                          weights_only=False)
    reloaded = OfficialDeepOPF(len(features.buses), args.hidden, physics.ngen).to(device)
    reloaded.load_state_dict(selected["model_state"])
    replay, _ = test_predictions(reloaded, physics, features, pd_all,
                                  indices["test"], args.eval_batch_size, device)
    reload_max_abs_mw = float(np.max(np.abs(replay - raw)))
    if reload_max_abs_mw > 1e-3:
        raise ValueError("checkpoint reload changed predictions")
    test_pd = np.asarray(data["pd_bus"], dtype=np.float64)[indices["test"]]
    target = np.asarray(data["pg_scopf"], dtype=np.float64)[indices["test"]]
    scoring = PreventiveLineSCPhysics(data, dtype=torch.float64)
    raw_metrics = evaluate_predictions(scoring, raw, test_pd, target, TOL_MW)
    result = {
        "status": "PASS", "protocol": PROTOCOL,
        "method": "DeepOPF-SC official architecture, supervised Phase A adaptation",
        "not_exact_upstream": True,
        "training_uses_solver_dispatch_labels": True,
        "dataset_sha256": sha256_file(data_path),
        "security_set_digest": data["provenance"]["configuration"]["security_set_digest"],
        "split_counts": {k: len(v) for k, v in indices.items()},
        "secure_tolerance_mw": TOL_MW,
        "hyperparameters": {"epochs": args.epochs, "batch_size": args.batch_size,
                            "hidden": [4 * args.hidden, 2 * args.hidden, args.hidden],
                            "lr": args.lr, "security_weight": args.security_weight,
                            "seed": args.seed},
        "best_epoch": best_epoch, "best_validation_loss": best,
        "train_time_sec": train_sec,
        "raw_inference_total_sec": inference_sec,
        "raw_amortized_ms_per_sample": inference_sec / len(raw) * 1000,
        "checkpoint_reload_max_abs_mw": reload_max_abs_mw,
        "raw_metrics": raw_metrics, "history": history,
        "environment": {"host": platform.node(), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                        "torch": torch.__version__, "device": str(device)},
    }
    if args.project:
        projected, times, _ = L1FeasibilityProjector(scoring).project_batch(raw, test_pd)
        result["projected_metrics"] = evaluate_predictions(scoring, projected,
                                                              test_pd, target, TOL_MW)
        result["projection_total_sec"] = float(times.sum())
        result["projection_amortized_ms_per_sample"] = float(times.mean() * 1000)
    else:
        projected = np.empty((0, scoring.ngen), dtype=np.float64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "test_predictions.npz",
                        dataset_row_index=indices["test"], pd_bus=test_pd,
                        pg_ground_truth=target, pg_raw=raw, pg_projected=projected)
    save_json(args.output_dir / "results.json", result)
    print(json.dumps({"status": "PASS", "raw_metrics": raw_metrics,
                      "projected_metrics": result.get("projected_metrics")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--security-weight", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--project", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    run(args)


if __name__ == "__main__":
    main()
