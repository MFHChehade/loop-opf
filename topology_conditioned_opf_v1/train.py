#!/usr/bin/env python3
"""Train topology-conditioned scenario-wise N-1 DC-OPF baselines.

This is not the preventive K=32 SC-DCOPF task. Each row has one realized
clean, line-outage, or generator-outage topology and one corresponding OPF
dispatch. The two models receive explicit topology inputs for every row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fair4net_l1_study"))
import baseline_harness_fair4net as harness  # noqa: E402
import baselines_fair4net as existing  # noqa: E402
from models import AdmittanceConditionedNet, DynamicTopologyGNN  # noqa: E402

PROTOCOL = "topology_conditioned_scenario_n1_dcopf_v1"
FLOW_TOL_MW = 1e-3
RELOAD_TOL_MW = 2e-2  # Allow float32 CUDA index_add reduction-order jitter.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def validate_splits(data: dict, indices: dict, nbranch: int, ngen: int) -> dict:
    lines = data["cont_line"].detach().cpu().numpy()
    gens = data["cont_gen"].detach().cpu().numpy()
    if np.any((lines >= 0) & (gens >= 0)):
        raise ValueError("simultaneous line and generator outages are outside this protocol")
    if np.any((lines < -1) | (lines >= nbranch)) or np.any((gens < -1) | (gens >= ngen)):
        raise ValueError("outage index outside base internal network order")
    identities = {}
    for family, values in (("line", lines), ("generator", gens)):
        sets = {name: set(map(int, values[ids][values[ids] >= 0])) for name, ids in indices.items()}
        for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
            if sets[first] & sets[second]:
                raise ValueError(f"{family} outage identities overlap across {first}/{second}")
        identities[family] = {name: sorted(value) for name, value in sets.items()}
    return identities


def make_model(method: str, static: dict, device: torch.device) -> torch.nn.Module:
    susceptance = 1.0 / (np.asarray(static["branch_x"]) *
                          np.asarray(static["branch_tap"]))
    if not np.all(np.isfinite(susceptance)):
        raise ValueError("nonfinite branch susceptance")
    if method == "admittance_dense":
        return AdmittanceConditionedNet(static["nbus"], static["ngen"],
                                        torch.as_tensor(susceptance)).to(device)
    if method == "dynamic_gnn":
        f, t = harness.build_graph()
        if len(f) != static["nbranch"] or len(t) != static["nbranch"]:
            raise ValueError("graph branches do not match operator base branch rows")
        return DynamicTopologyGNN(
            static["nbus"], torch.as_tensor(static["bus_of_gen"]),
            torch.as_tensor(f), torch.as_tensor(t),
            torch.as_tensor(susceptance),
        ).to(device)
    raise ValueError(method)


def scenario_bounds(P: dict, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    upper = P["pmax_eff"][ids]
    lower = P["pmin"][ids].clone()
    lower = torch.where(upper <= 0, torch.zeros_like(lower), lower)
    if bool((lower > upper).any()):
        raise ValueError("scenario generator bounds are inconsistent")
    return lower, upper


def predict(model, P: dict, ids: torch.Tensor, normalization: dict) -> torch.Tensor:
    demand = P["d_eff"][ids]
    lower, upper = scenario_bounds(P, ids)
    capacity = upper
    dnorm = (demand - normalization["demand_mean"]) / normalization["demand_std"]
    cnorm = capacity / normalization["capacity_scale"]
    if isinstance(model, AdmittanceConditionedNet):
        unit = model(dnorm, cnorm, P["cont_line"][ids], P["line_features"][ids])
    else:
        unit = model(dnorm, cnorm, P["cont_line"][ids])
    raw = lower + unit * (upper - lower)
    return existing._feas(P, raw, ids) if bool((lower == P["pmin"][ids]).all()) else _repair(
        P, raw, ids, lower, upper
    )


def _repair(P, raw, ids, lower, upper):
    # The existing helper assumes fixed lower bounds. Generator-outage rows
    # require a zero lower bound for the unavailable unit.
    return harness.balance_repair(raw, P["d_eff"][ids], lower, upper)


def training_loss(P: dict, pred: torch.Tensor, ids: torch.Tensor,
                  label_scale: torch.Tensor, flow_weight: float) -> tuple[torch.Tensor, dict]:
    target = P["pg"][ids]
    mse = ((pred - target) / label_scale).square().mean()
    violation = P["op"].flow_violation_vec(pred, P["pd_bus"][ids],
                                             P["cont_line"][ids])
    fmax = P["op"].fmax[None, :].clamp_min(1.0)
    flow = (violation / fmax).amax(dim=1).square().mean()
    return mse + flow_weight * flow, {"normalized_dispatch_mse": mse, "flow_penalty": flow}


@torch.no_grad()
def validation_loss(model, P, indices, normalization, label_scale, flow_weight, batch_size):
    model.eval()
    total = 0.0
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start + batch_size]
        pred = predict(model, P, ids, normalization)
        loss, _ = training_loss(P, pred, ids, label_scale, flow_weight)
        total += float(loss.item()) * len(ids)
    return total / len(indices)


@torch.no_grad()
def evaluate(model, P, indices, normalization, batch_size):
    model.eval()
    predictions = []
    start = time.perf_counter()
    for begin in range(0, len(indices), batch_size):
        ids = indices[begin:begin + batch_size]
        predictions.append(predict(model, P, ids, normalization))
    if torch.device(P["device"]).type == "cuda":
        torch.cuda.synchronize(P["device"])
    elapsed = time.perf_counter() - start
    predicted = torch.cat(predictions, dim=0)
    return predicted, elapsed


def train_one(method: str, P: dict, args, normalization: dict, dataset_hash: str,
              identities: dict) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = make_model(method, P["static"], P["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_ids = torch.as_tensor(P["idx"]["train"], device=P["device"])
    val_ids = torch.as_tensor(P["idx"]["val"], device=P["device"])
    test_ids = torch.as_tensor(P["idx"]["test"], device=P["device"])
    label_scale = P["pg"][train_ids].std(dim=0).clamp_min(1.0)
    out_dir = args.out_dir / method
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "latest.pt"
    run_contract = {"protocol": PROTOCOL, "method": method,
                    "dataset_sha256": dataset_hash, "batch_size": args.batch_size,
                    "learning_rate": args.lr, "flow_weight": args.flow_weight,
                    "seed": args.seed}
    best = math.inf
    best_epoch = 0
    best_state = None
    history = []
    first_epoch = 1
    if args.resume and progress_path.is_file():
        progress = torch.load(progress_path, map_location=P["device"], weights_only=False)
        if progress.get("run_contract") != run_contract:
            raise ValueError("progress checkpoint does not match the run contract")
        model.load_state_dict(progress["model_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        best = float(progress["best_validation_loss"])
        best_epoch = int(progress["best_epoch"])
        best_state = progress["best_state"]
        history = progress["history"]
        first_epoch = int(progress["epoch"]) + 1
        torch.set_rng_state(progress["cpu_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state" in progress:
            torch.cuda.set_rng_state_all([state.cpu() for state in progress["cuda_rng_state"]])
        print(f"[resume] {method} from completed epoch {first_epoch - 1}", flush=True)
    if first_epoch > args.epochs:
        print(f"[resume] {method} already reached target epoch {args.epochs}", flush=True)
    started = time.perf_counter()
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        order = train_ids[torch.randperm(len(train_ids), device=P["device"])]
        losses = []
        for begin in range(0, len(order), args.batch_size):
            ids = order[begin:begin + args.batch_size]
            prediction = predict(model, P, ids, normalization)
            loss, _ = training_loss(P, prediction, ids, label_scale, args.flow_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite topology-conditioned loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        value = validation_loss(model, P, val_ids, normalization, label_scale,
                                args.flow_weight, args.batch_size)
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite validation loss")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation_loss": value})
        if value < best:
            best = value
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs:
            progress = {"run_contract": run_contract, "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_validation_loss": best, "best_epoch": best_epoch,
                        "best_state": best_state, "history": history,
                        "cpu_rng_state": torch.get_rng_state()}
            if torch.cuda.is_available():
                progress["cuda_rng_state"] = torch.cuda.get_rng_state_all()
            temporary = progress_path.with_suffix(".pt.tmp")
            torch.save(progress, temporary)
            temporary.replace(progress_path)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps({"method": method, "epoch": epoch,
                              "validation_loss": value, "best_validation_loss": best}), flush=True)
    train_sec = time.perf_counter() - started
    model.load_state_dict(best_state)
    predicted, inference_sec = evaluate(model, P, test_ids, normalization,
                                         args.batch_size)
    checkpoint = out_dir / "selected.pt"
    torch.save({"protocol": PROTOCOL, "method": method,
                "dataset_sha256": dataset_hash, "model_state": best_state,
                "normalization": {k: v.detach().cpu() for k, v in normalization.items()},
                "epoch": best_epoch, "validation_loss": best}, checkpoint)
    reloaded = make_model(method, P["static"], P["device"])
    payload = torch.load(checkpoint, map_location=P["device"], weights_only=False)
    if payload["dataset_sha256"] != dataset_hash or payload["method"] != method:
        raise ValueError("selected checkpoint provenance mismatch")
    reloaded.load_state_dict(payload["model_state"])
    replay, _ = evaluate(reloaded, P, test_ids, normalization, args.batch_size)
    reload_max_abs_MW = float((replay - predicted).abs().max())
    if reload_max_abs_MW > RELOAD_TOL_MW:
        raise ValueError(f"selected checkpoint reload changes predictions: {reload_max_abs_MW}")
    # Score and save the predictions produced by the persisted checkpoint.
    predicted = replay
    metrics = harness.score(P["op"], predicted, P["pd_bus"][test_ids],
                            P["pg"][test_ids], cont_line_idx=P["cont_line"][test_ids],
                            cont_gen_idx=P["cont_gen"][test_ids])
    np.savez_compressed(out_dir / "test_predictions.npz",
                        dataset_row_index=np.asarray(P["idx"]["test"], dtype=np.int64),
                        cont_line_idx=P["cont_line"][test_ids].cpu().numpy(),
                        cont_gen_idx=P["cont_gen"][test_ids].cpu().numpy(),
                        pg_pred=predicted.cpu().numpy(),
                        pg_label=P["pg"][test_ids].cpu().numpy())
    result = {"status": "PASS", "protocol": PROTOCOL, "method": method,
              "dataset_sha256": dataset_hash, "scenario": "realized single-contingency DC-OPF",
              "not_preventive_scopf": True,
              "input_contract": "effective PD+GS, generator availability, active admittances and LODF features or per-sample graph",
              "outage_identities": identities,
              "split_counts": {k: len(v) for k, v in P["idx"].items()},
              "secure_tolerance_mw": FLOW_TOL_MW,
              "train_time_sec": train_sec if first_epoch == 1 else None,
              "training_time_note": ("full invocation" if first_epoch == 1 else
                                     "resumed from a progress checkpoint; prior training time unavailable"),
              "batched_test_inference_sec": inference_sec,
              "batched_throughput_samples_per_sec": len(test_ids) / inference_sec,
              "best_epoch": best_epoch, "best_validation_loss": best,
              "checkpoint_reload_max_abs_MW": reload_max_abs_MW,
              "checkpoint_reload_tolerance_MW": RELOAD_TOL_MW,
              "epochs": args.epochs, "batch_size": args.batch_size,
              "flow_weight": args.flow_weight, "seed": args.seed,
              "metrics": metrics, "history": history,
              "checkpoint_sha256": sha256(checkpoint)}
    write_json(out_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["admittance_dense", "dynamic_gnn"],
                        choices=["admittance_dense", "dynamic_gnn"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--flow-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-smoke-split", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("DEQ_CASE"):
        raise RuntimeError("DEQ_CASE must select the exact network")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    audit_path = args.data.parent / "dataset_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"independent dataset audit missing: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("case") != os.environ["DEQ_CASE"]:
        raise ValueError("dataset audit is not a PASS for the selected network")
    P = existing.prep(*harness.build_operator(args.device), args.data, args.device)
    counts = {k: len(v) for k, v in P["idx"].items()}
    if audit.get("N") != sum(counts.values()):
        raise ValueError("dataset row count differs from independent audit")
    audited_shape = audit.get("shape", {})
    if any(int(audited_shape.get(key, -1)) != int(P["static"][key])
           for key in ("nbus", "ngen", "nbranch")):
        raise ValueError("dataset network shape differs from independent audit")
    if any(int(audit.get("split_counts", {}).get(str(i), -1)) != counts[name]
           for i, name in enumerate(("train", "val", "test"))):
        raise ValueError("dataset split counts differ from independent audit")
    if not args.allow_smoke_split and counts != {"train": 7000, "val": 1000, "test": 2000}:
        raise ValueError(f"production split must be 7000/1000/2000, got {counts}")
    identities = validate_splits(P, P["idx"], P["static"]["nbranch"], P["ngen"])
    train_ids = torch.as_tensor(P["idx"]["train"], device=P["device"])
    dtrain = P["d_eff"][train_ids]
    normalization = {
        "demand_mean": dtrain.mean(dim=0, keepdim=True),
        "demand_std": dtrain.std(dim=0, keepdim=True).clamp_min(1.0),
        "capacity_scale": P["pmax_eff"][train_ids].amax().clamp_min(1.0),
    }
    dataset_hash = sha256(args.data)
    results = {name: train_one(name, P, args, normalization, dataset_hash, identities)
               for name in args.methods}
    write_json(args.out_dir / "summary.json", {
        "protocol": PROTOCOL, "dataset_sha256": dataset_hash,
        "results": {k: {"metrics": v["metrics"], "checkpoint_sha256": v["checkpoint_sha256"]}
                    for k, v in results.items()},
    })


if __name__ == "__main__":
    main()
