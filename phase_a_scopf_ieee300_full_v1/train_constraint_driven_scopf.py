#!/usr/bin/env python3
"""Label-free constraint-driven baseline for the audited Phase A K=32 study.

This is a line-contingency adaptation inspired by Giraud et al., not an
exact reproduction of their CVXPYLayer base-case restoration. The sigmoid
network predicts bounded dispatch, and a differentiable scalar water-filling
projection enforces balance and generator limits. The training loss uses
generation cost and all declared base/outage thermal constraints, never
solver dispatch labels. Full K=32 L1 recovery is an optional, separately
timed postprocessing stage for comparison with DeepOPF.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from deepopf_scdcopf_official.data import FeatureContract, load_exact_phase_a_dataset
from deepopf_scdcopf_official.metrics import evaluate_predictions
from deepopf_scdcopf_official.model import OfficialDeepOPF
from deepopf_scdcopf_official.physics import PreventiveLineSCPhysics
from deepopf_scdcopf_official.projection import L1FeasibilityProjector
from phase_a_scopf.common import sha256_file

PROTOCOL = "constraint_driven_preventive_line_k32_phase_a_v1"
TOL = 1e-3


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def balance_project(raw: torch.Tensor, demand: torch.Tensor,
                    lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto box and one balance equality, batched."""
    if bool((demand < lo.sum() - 1e-3).any()) or bool((demand > hi.sum() + 1e-3).any()):
        raise ValueError("total demand outside generator aggregate bounds")
    left = (lo[None, :] - raw).amin(dim=1, keepdim=True)
    right = (hi[None, :] - raw).amax(dim=1, keepdim=True)
    # Bisection determines the active set. Final interpolation supplies the
    # correct gradient with respect to dispatch within that active set.
    for _ in range(48):
        mid = (left + right) * 0.5
        total = (raw + mid).clamp(lo[None, :], hi[None, :]).sum(dim=1, keepdim=True)
        left = torch.where(total < demand[:, None], mid, left)
        right = torch.where(total >= demand[:, None], mid, right)
    lam = ((left + right) * 0.5).detach()
    candidate = (raw + lam).clamp(lo[None, :], hi[None, :])
    free = ((candidate > lo[None, :] + 1e-5) &
            (candidate < hi[None, :] - 1e-5)).to(raw.dtype).detach()
    correction = ((demand[:, None] - candidate.sum(dim=1, keepdim=True)) /
                  free.sum(dim=1, keepdim=True).clamp_min(1.0))
    return candidate + free * correction


def predict(model, physics, features, pd):
    unit = model(features.normalize_torch(pd))
    raw = physics.pmin[None, :] + unit * (physics.pmax - physics.pmin)[None, :]
    demand = physics.effective_demand(pd).sum(dim=1)
    return balance_project(raw, demand, physics.pmin, physics.pmax)


def loss_parts(physics, pg, pd, security_weight):
    flow = physics.all_flows(pg, pd)
    normalized_overload = ((flow.abs() - physics.fmax[None, None, :]) /
                           physics.fmax[None, None, :].clamp_min(1.0)).clamp_min(0.0)
    normalized_overload = normalized_overload * physics.flow_mask[None, :, :]
    sample_worst = normalized_overload.amax(dim=(1, 2))
    cost = physics.objective(pg).mean() / 1e7
    security = sample_worst.square().mean()
    return cost + security_weight * security, cost, security


@torch.no_grad()
def evaluate_loss(model, physics, features, pd_all, indices, batch, weight, device):
    model.eval()
    total = 0.0
    for start in range(0, len(indices), batch):
        ids = indices[start:start + batch]
        pd = pd_all[ids].to(device)
        pg = predict(model, physics, features, pd)
        value, _, _ = loss_parts(physics, pg, pd, weight)
        total += float(value.item()) * len(ids)
    return total / len(indices)


@torch.no_grad()
def test_predictions(model, physics, features, pd_all, indices, batch, device):
    model.eval()
    warm = pd_all[indices[:min(batch, len(indices))]].to(device)
    predict(model, physics, features, warm)
    if device.type == "cuda": torch.cuda.synchronize(device)
    start = time.perf_counter()
    outputs = []
    for pos in range(0, len(indices), batch):
        pd = pd_all[indices[pos:pos + batch]].to(device)
        outputs.append(predict(model, physics, features, pd).cpu().numpy())
    if device.type == "cuda": torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return np.concatenate(outputs).astype(np.float64), elapsed


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
    indices = {name: np.flatnonzero(split == value) for name, value in
               (("train", 0), ("val", 1), ("test", 2))}
    if args.smoke:
        indices = {name: ids[:n] for name, ids, n in
                   (("train", indices["train"], 70),
                    ("val", indices["val"], 10),
                    ("test", indices["test"], 20))}
    features = FeatureContract.fit(data["pd_bus"], split)
    physics = PreventiveLineSCPhysics(data, dtype=torch.float32).to(device)
    model = OfficialDeepOPF(len(features.buses), args.hidden, physics.ngen).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pd_all = torch.as_tensor(data["pd_bus"], dtype=torch.float32)
    best = math.inf
    best_state = None
    best_epoch = None
    started = time.perf_counter()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(indices["train"])
        epoch_loss = []
        for pos in range(0, len(order), args.batch_size):
            ids = order[pos:pos + args.batch_size]
            pd = pd_all[ids].to(device)
            pg = predict(model, physics, features, pd)
            objective, _, _ = loss_parts(physics, pg, pd, args.security_weight)
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            epoch_loss.append(float(objective.detach().item()))
        val = evaluate_loss(model, physics, features, pd_all, indices["val"],
                            args.eval_batch_size, args.security_weight, device)
        if not math.isfinite(val): raise FloatingPointError("nonfinite validation loss")
        history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_loss)),
                        "validation_loss": val})
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_torch(args.output_dir / "best.pt", {
                "protocol": PROTOCOL, "dataset_sha256": sha256_file(data_path),
                "security_set_digest": data["provenance"]["configuration"]["security_set_digest"],
                "model_state": best_state, "hidden": args.hidden,
                "feature_contract": features.payload(), "selected_epoch": best_epoch,
                "validation_loss": best,
            })
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps({"epoch": epoch, "validation_loss": val,
                              "best_validation_loss": best}), flush=True)
    model.load_state_dict(best_state)
    raw, inference_sec = test_predictions(model, physics, features, pd_all,
                                           indices["test"], args.eval_batch_size, device)
    test_pd = np.asarray(data["pd_bus"], dtype=np.float64)[indices["test"]]
    target = np.asarray(data["pg_scopf"], dtype=np.float64)[indices["test"]]
    scoring = PreventiveLineSCPhysics(data, dtype=torch.float64)
    raw_metrics = evaluate_predictions(scoring, raw, test_pd, target, TOL)
    output = {"status": "PASS", "protocol": PROTOCOL,
              "method": "Giraud-inspired label-free fixed-K line adaptation; balanced bounded sigmoid net",
              "not_exact_upstream": "scalar balance projection replaces CVXPYLayer base-case restoration",
              "dataset_sha256": sha256_file(data_path),
              "security_set_digest": data["provenance"]["configuration"]["security_set_digest"],
              "split_counts": {k: len(v) for k, v in indices.items()},
              "training_uses_solver_dispatch_labels": False,
              "validation_uses_solver_dispatch_labels": False,
              "secure_tolerance_mw": TOL,
              "hyperparameters": {"epochs": args.epochs, "batch_size": args.batch_size,
                                  "hidden": [4*args.hidden, 2*args.hidden, args.hidden],
                                  "lr": args.lr, "security_weight": args.security_weight,
                                  "seed": args.seed},
              "best_epoch": best_epoch, "best_validation_loss": best,
              "train_time_sec": time.perf_counter() - started,
              "raw_inference_total_sec": inference_sec,
              "raw_amortized_ms_per_sample": inference_sec / len(raw) * 1000,
              "raw_metrics": raw_metrics, "history": history,
              "environment": {"host": platform.node(), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                              "torch": torch.__version__, "device": str(device)}}
    if args.project:
        projected, times, _ = L1FeasibilityProjector(scoring).project_batch(raw, test_pd)
        output["projected_metrics"] = evaluate_predictions(scoring, projected, test_pd, target, TOL)
        output["projection_total_sec"] = float(times.sum())
        output["projection_amortized_ms_per_sample"] = float(times.mean() * 1000)
    else:
        projected = np.empty((0, scoring.ngen), dtype=np.float64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "test_predictions.npz",
                        dataset_row_index=indices["test"], pd_bus=test_pd,
                        pg_ground_truth=target, pg_raw=raw, pg_projected=projected)
    save_json(args.output_dir / "results.json", output)
    print(json.dumps({"status": output["status"], "raw_metrics": raw_metrics,
                      "projected_metrics": output.get("projected_metrics")}), flush=True)


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
    run(parser.parse_args())


if __name__ == "__main__": main()
