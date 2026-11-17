#!/usr/bin/env python3
"""Resumable production training for preventive IEEE 300 SC-DCOPF.

This keeps the audited Phase-A physics and equilibrium solver from the smoke
implementation, but removes smoke-only assumptions: arbitrary deterministic
70/10/20 datasets, configurable epochs/batches, validation-only checkpoint
selection, epoch checkpoints, fresh-process reload, and artifact manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch

from phase_a_scopf.common import sha256_file
from train_phase_a_smoke import (
    DEQFixedPoint,
    LearnableSteps,
    MLP,
    PreventiveSCOperator,
    _evaluate_deq,
    _execution_device,
    _synchronize,
    _tensor_data,
    balance_repair,
    estimate_norm_k,
    load_audited_dataset,
    score,
)


PROTOCOL = "phase_a_preventive_sc_deq_production_v1"
SECURE_TOL_MW = 1e-3


def _atomic_json(value: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_torch_save(value: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for name, value in list(state.items()):
            if torch.is_tensor(value):
                state[name] = value.to(device)


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _deq_config(epochs: int, batch_size: int, seed: int) -> Dict[str, Any]:
    return {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": 1e-4,
        "forward_solver": "adaptive_picard_physics_gated",
        "forward_dtype": "float64",
        "forward_min_iter": 3000,
        "forward_max_iter": 5000,
        "forward_check_every": 25,
        "forward_pg_step_tol_mw": 1e-5,
        "forward_balance_tol_mw": SECURE_TOL_MW,
        "forward_bound_tol_mw": SECURE_TOL_MW,
        "forward_flow_tol_mw": SECURE_TOL_MW,
        "forward_dual_step_tol_mw": SECURE_TOL_MW,
        "validation_selected_s_init": -0.5,
        "s_max": 3.0,
        "forward_anderson": "rejected_by_validation_diagnostics",
        "backward_solver": "adaptive_picard_worst_sample",
        "backward_dtype": "float64",
        "backward_min_iter": 600,
        "backward_max_iter": 2500,
        "backward_check_every": 25,
        "backward_tol": 1e-6,
        "backward_anderson": "rejected_by_batched_validation_diagnostics",
        "w_cost": 1.0,
        "w_balance": 50.0,
        "w_flow_quadratic": 1e8,
        "w_flow_l1": 1e4,
        "seed": int(seed),
        "step_mode": "scalar",
    }


def _baseline_config(epochs: int, batch_size: int, seed: int) -> Dict[str, Any]:
    return {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": 1e-3,
        "seed": int(seed),
        "hidden": [256, 256],
        "selection": "minimum validation normalized MSE; test unused until final",
    }


def _validate_run_arguments(
    epochs: int,
    batch_size: int,
    baseline_batch_size: int,
    eval_batch_size: int,
    eval_every: int,
) -> None:
    values = {
        "epochs": epochs,
        "batch_size": batch_size,
        "baseline_batch_size": baseline_batch_size,
        "eval_batch_size": eval_batch_size,
        "eval_every": eval_every,
    }
    bad = {name: value for name, value in values.items() if value < 1}
    if bad:
        raise ValueError(f"all production integer controls must be positive: {bad}")


def _contract(
    data_path: Path,
    data: Dict[str, Any],
    deq_config: Dict[str, Any],
    baseline_config: Dict[str, Any],
    eval_every: int,
) -> Dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "dataset_sha256": sha256_file(data_path),
        "security_set_digest": data["provenance"]["configuration"][
            "security_set_digest"
        ],
        "deq_config": deq_config,
        "baseline_config": baseline_config,
        "eval_every": int(eval_every),
    }


def _require_contract(payload: Dict[str, Any], contract: Dict[str, Any]) -> None:
    if payload.get("contract") != contract:
        raise ValueError("checkpoint contract mismatch; refusing unsafe resume")


def _label_self_check(data: Dict[str, Any]) -> Dict[str, float]:
    operator = PreventiveSCOperator(data, dtype=torch.float64)
    pd_bus = torch.as_tensor(data["pd_bus"], dtype=torch.float64)
    target = torch.as_tensor(data["pg_scopf"], dtype=torch.float64)
    metrics = score(operator, target, pd_bus, target)
    if (
        metrics["balance_abs_max_mw"] > 1e-6
        or metrics["all_contingency_violation_abs_max_mw"] > 1e-4
        or metrics["dispatch_rmse_mw"] != 0.0
    ):
        raise ValueError(f"training operator does not reproduce audited labels: {metrics}")
    return metrics


def _deq_validation_key(
    metrics: Dict[str, float], convergence: Dict[str, Any]
) -> Tuple[float, float, float]:
    if (
        not convergence["all_batches_converged"]
        or metrics["balance_abs_max_mw"] > SECURE_TOL_MW
        or metrics["all_contingency_secure_rate_at_1e3_mw"] != 1.0
    ):
        return (math.inf, math.inf, math.inf)
    return (
        float(convergence["iterations_mean"]),
        float(convergence["relative_fixed_point_residual_abs_max"]),
        float(metrics["cost_relative_gap_abs_mean"]),
    )


def _train_deq(
    data_path: Path,
    data: Dict[str, Any],
    out_dir: Path,
    execution_device: torch.device,
    contract: Dict[str, Any],
    eval_every: int,
    eval_batch_size: int,
    resume: bool,
) -> Dict[str, Any]:
    config = contract["deq_config"]
    pd_bus = torch.as_tensor(data["pd_bus"], dtype=torch.float64)
    target = torch.as_tensor(data["pg_scopf"], dtype=torch.float64)
    split = data["split"].astype(np.int64)
    indices = {
        name: np.flatnonzero(split == value)
        for name, value in zip(("train", "val", "test"), (0, 1, 2))
    }
    scoring_operator = PreventiveSCOperator(data, dtype=torch.float64)
    progress_path = out_dir / "deq_progress.pt"
    final_path = out_dir / "deq_final.pt"

    static_operator = PreventiveSCOperator(data)
    norm_k = estimate_norm_k(static_operator, iterations=100)
    steps = LearnableSteps(
        norm_k,
        s_init=config["validation_selected_s_init"],
        s_max=config["s_max"],
    )
    operator = PreventiveSCOperator(data, steps=steps, dtype=torch.float64)
    operator.to(execution_device)
    deq = DEQFixedPoint(operator, config)
    optimizer = torch.optim.Adam(steps.parameters(), lr=config["learning_rate"])

    completed_epochs = 0
    history = []
    best_key = (math.inf, math.inf, math.inf)
    best_epoch = 0
    best_steps_state = None
    if resume and progress_path.is_file():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        _require_contract(progress, contract)
        if not math.isclose(float(progress["norm_k"]), norm_k, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("operator norm changed since the progress checkpoint")
        steps.load_state_dict(progress["steps_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        _move_optimizer_state(optimizer, execution_device)
        completed_epochs = int(progress["completed_epochs"])
        history = list(progress["history"])
        best_key = tuple(float(value) for value in progress["best_key"])
        best_epoch = int(progress["best_epoch"])
        best_steps_state = progress["best_steps_state"]
        print(json.dumps({"event": "deq_resume", "completed_epochs": completed_epochs}), flush=True)
    elif progress_path.exists():
        raise FileExistsError(
            f"{progress_path} exists; pass --resume or choose a new output directory"
        )

    if completed_epochs > config["epochs"]:
        raise ValueError("progress checkpoint is beyond requested epoch count")
    if execution_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(execution_device)

    for epoch_zero in range(completed_epochs, config["epochs"]):
        epoch = epoch_zero + 1
        order = np.random.default_rng(config["seed"] + epoch_zero).permutation(
            indices["train"]
        )
        loss_values = []
        forward_iterations = []
        backward_iterations = []
        forward_residuals = []
        backward_residuals = []
        gradient_maxima = []
        _synchronize(execution_device)
        started = time.perf_counter()
        for start in range(0, len(order), config["batch_size"]):
            batch = order[start : start + config["batch_size"]]
            deq.train()
            batch_pd_bus = pd_bus[batch].to(execution_device)
            z = deq(batch_pd_bus)
            prediction = z[:, : operator.ngen]
            cost = operator.objective(prediction) / operator.effective_demand(
                batch_pd_bus
            ).sum(dim=1).clamp_min(1e-9)
            balance = prediction.sum(dim=1) - operator.effective_demand(
                batch_pd_bus
            ).sum(dim=1)
            violation = operator.violation(prediction, batch_pd_bus)
            loss = (
                config["w_cost"] * cost.mean()
                + config["w_balance"] * balance.square().mean()
                + config["w_flow_quadratic"] * violation.square().mean()
                + config["w_flow_l1"] * violation.mean()
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("DEQ production loss is nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [parameter.grad for parameter in steps.parameters()]
            if any(
                gradient is None or not bool(torch.isfinite(gradient).all())
                for gradient in gradients
            ):
                raise FloatingPointError("DEQ scalar-step gradient is missing or nonfinite")
            if not deq.forward_converged:
                raise RuntimeError(f"DEQ forward solve failed in epoch {epoch}")
            if (
                not deq.backward_residuals
                or deq.backward_residuals[-1] > config["backward_tol"]
            ):
                raise RuntimeError(f"DEQ backward solve failed in epoch {epoch}")
            gradient_maxima.append(
                max(float(gradient.detach().abs().max().item()) for gradient in gradients)
            )
            optimizer.step()
            loss_values.append(float(loss.detach().item()))
            forward_iterations.append(int(deq.forward_iterations))
            backward_iterations.append(int(deq.backward_iterations))
            forward_residuals.append(float(deq.forward_residuals[-1]))
            backward_residuals.append(float(deq.backward_residuals[-1]))
        _synchronize(execution_device)
        elapsed = time.perf_counter() - started
        row: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(loss_values)),
            "train_time_sec": elapsed,
            "train_samples_per_sec": float(len(order) / elapsed),
            "forward_iterations_mean": float(np.mean(forward_iterations)),
            "forward_iterations_max": int(max(forward_iterations)),
            "forward_residual_abs_max": float(max(forward_residuals)),
            "backward_iterations_mean": float(np.mean(backward_iterations)),
            "backward_iterations_max": int(max(backward_iterations)),
            "backward_residual_abs_max": float(max(backward_residuals)),
            "step_gradient_abs_max": float(max(gradient_maxima)),
            "rho": float(steps.rho().detach().item()),
            "s": float(steps.s().detach().item()),
        }
        if epoch % eval_every == 0 or epoch == config["epochs"]:
            _, metrics, eval_sec, residual, convergence = _evaluate_deq(
                deq,
                operator,
                pd_bus,
                target,
                indices["val"],
                scoring_operator,
                pd_bus,
                target,
                batch_size=eval_batch_size,
            )
            key = _deq_validation_key(metrics, convergence)
            if not np.all(np.isfinite(key)):
                raise RuntimeError(
                    f"DEQ validation security/convergence gate failed in epoch {epoch}: "
                    f"metrics={metrics}, convergence={convergence}"
                )
            row["validation"] = {
                "metrics": metrics,
                "convergence": convergence,
                "fixed_point_residual_mean": residual,
                "eval_time_sec": eval_sec,
                "selection_key": list(key),
            }
            if key < best_key:
                best_key = key
                best_epoch = epoch
                best_steps_state = _cpu_state_dict(steps)
        history.append(row)
        progress = {
            "checkpoint_type": "epoch_progress",
            "contract": contract,
            "norm_k": norm_k,
            "completed_epochs": epoch,
            "steps_state": _cpu_state_dict(steps),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_key": list(best_key),
            "best_epoch": best_epoch,
            "best_steps_state": best_steps_state,
        }
        _atomic_torch_save(progress, progress_path)
        _atomic_json(
            {
                "status": "IN_PROGRESS" if epoch < config["epochs"] else "EPOCHS_COMPLETE",
                "completed_epochs": epoch,
                "target_epochs": config["epochs"],
                "best_epoch": best_epoch,
                "latest": row,
            },
            out_dir / "deq_progress.json",
        )
        print(json.dumps({"event": "deq_epoch", **row}), flush=True)

    if best_steps_state is None or not np.all(np.isfinite(best_key)):
        raise RuntimeError("no validation-secure DEQ checkpoint was produced")
    steps.load_state_dict(best_steps_state)
    predictions: Dict[str, torch.Tensor] = {}
    metrics_by_split: Dict[str, Dict[str, float]] = {}
    convergence_by_split: Dict[str, Dict[str, Any]] = {}
    eval_time_by_split: Dict[str, float] = {}
    residual_by_split: Dict[str, float] = {}
    for name in ("train", "val", "test"):
        prediction, metrics, elapsed, residual, convergence = _evaluate_deq(
            deq,
            operator,
            pd_bus,
            target,
            indices[name],
            scoring_operator,
            pd_bus,
            target,
            batch_size=eval_batch_size,
        )
        predictions[name] = prediction
        metrics_by_split[name] = metrics
        convergence_by_split[name] = convergence
        eval_time_by_split[name] = elapsed
        residual_by_split[name] = residual
    payload = {
        "checkpoint_type": "selected_final",
        "protocol": PROTOCOL,
        "contract": contract,
        "dataset_path": str(data_path.resolve()),
        "dataset_sha256": contract["dataset_sha256"],
        "security_set_digest": contract["security_set_digest"],
        "config": config,
        "norm_k": norm_k,
        "selected_epoch": best_epoch,
        "selection_key": list(best_key),
        "steps_state": _cpu_state_dict(steps),
        "history": history,
        "metrics": metrics_by_split,
        "eval_convergence": convergence_by_split,
        "eval_time_sec": eval_time_by_split,
        "eval_fixed_point_residual_mean": residual_by_split,
        "reference_test_prediction": predictions["test"],
        "reference_test_metrics": metrics_by_split["test"],
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(execution_device))
            if execution_device.type == "cuda"
            else None
        ),
        "peak_cuda_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(execution_device))
            if execution_device.type == "cuda"
            else None
        ),
    }
    _atomic_torch_save(payload, final_path)
    return payload


@torch.no_grad()
def _baseline_validation_loss(
    model: MLP,
    normalized_input: torch.Tensor,
    normalized_target: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        prediction = model(normalized_input[batch].to(device))
        target = normalized_target[batch].to(device)
        total += float((prediction - target).square().sum().item())
        count += int(target.numel())
    return total / max(1, count)


@torch.no_grad()
def _evaluate_baseline(
    model: MLP,
    operator: PreventiveSCOperator,
    scoring_operator: PreventiveSCOperator,
    pd_bus: torch.Tensor,
    target: torch.Tensor,
    effective: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float], float]:
    model.eval()
    rows = []
    _synchronize(device)
    started = time.perf_counter()
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        raw = model(((effective[batch] - input_mean) / input_std).to(device)).cpu()
        raw = raw * target_std + target_mean
        rows.append(balance_repair(operator, raw, pd_bus[batch]))
    _synchronize(device)
    elapsed = time.perf_counter() - started
    prediction = torch.cat(rows, dim=0)
    return (
        prediction,
        score(scoring_operator, prediction, pd_bus[indices].double(), target[indices].double()),
        elapsed,
    )


def _train_baseline(
    data_path: Path,
    data: Dict[str, Any],
    out_dir: Path,
    execution_device: torch.device,
    contract: Dict[str, Any],
    eval_every: int,
    resume: bool,
) -> Dict[str, Any]:
    config = contract["baseline_config"]
    pd_bus, target, split = _tensor_data(data)
    indices = {
        name: np.flatnonzero(split == value)
        for name, value in zip(("train", "val", "test"), (0, 1, 2))
    }
    operator = PreventiveSCOperator(data)
    scoring_operator = PreventiveSCOperator(data, dtype=torch.float64)
    effective = pd_bus + operator.gs_bus[None, :]
    input_mean = effective[indices["train"]].mean(dim=0, keepdim=True)
    input_std = effective[indices["train"]].std(dim=0, keepdim=True).clamp_min(1e-6)
    target_mean = target[indices["train"]].mean(dim=0, keepdim=True)
    target_std = target[indices["train"]].std(dim=0, keepdim=True).clamp_min(1e-6)
    normalized_input = (effective - input_mean) / input_std
    normalized_target = (target - target_mean) / target_std

    torch.manual_seed(config["seed"])
    model = MLP(operator.nbus, operator.ngen).to(execution_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    progress_path = out_dir / "baseline_progress.pt"
    final_path = out_dir / "baseline_final.pt"
    completed_epochs = 0
    history = []
    best_loss = math.inf
    best_epoch = 0
    best_model_state = None
    if resume and progress_path.is_file():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        _require_contract(progress, contract)
        model.load_state_dict(progress["model_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        _move_optimizer_state(optimizer, execution_device)
        completed_epochs = int(progress["completed_epochs"])
        history = list(progress["history"])
        best_loss = float(progress["best_loss"])
        best_epoch = int(progress["best_epoch"])
        best_model_state = progress["best_model_state"]
        print(json.dumps({"event": "baseline_resume", "completed_epochs": completed_epochs}), flush=True)
    elif progress_path.exists():
        raise FileExistsError(
            f"{progress_path} exists; pass --resume or choose a new output directory"
        )

    for epoch_zero in range(completed_epochs, config["epochs"]):
        epoch = epoch_zero + 1
        order = np.random.default_rng(config["seed"] + epoch_zero).permutation(
            indices["train"]
        )
        losses = []
        _synchronize(execution_device)
        started = time.perf_counter()
        model.train()
        for start in range(0, len(order), config["batch_size"]):
            batch = order[start : start + config["batch_size"]]
            prediction = model(normalized_input[batch].to(execution_device))
            expected = normalized_target[batch].to(execution_device)
            loss = (prediction - expected).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("baseline production loss is nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ):
                raise FloatingPointError("baseline gradient is missing or nonfinite")
            optimizer.step()
            losses.append(float(loss.detach().item()))
        _synchronize(execution_device)
        elapsed = time.perf_counter() - started
        row: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)),
            "train_time_sec": elapsed,
            "train_samples_per_sec": float(len(order) / elapsed),
        }
        if epoch % eval_every == 0 or epoch == config["epochs"]:
            val_loss = _baseline_validation_loss(
                model,
                normalized_input,
                normalized_target,
                indices["val"],
                config["batch_size"],
                execution_device,
            )
            row["validation_normalized_mse"] = val_loss
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_model_state = _cpu_state_dict(model)
        history.append(row)
        _atomic_torch_save(
            {
                "checkpoint_type": "epoch_progress",
                "contract": contract,
                "completed_epochs": epoch,
                "model_state": _cpu_state_dict(model),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "best_model_state": best_model_state,
            },
            progress_path,
        )
        _atomic_json(
            {
                "status": "IN_PROGRESS" if epoch < config["epochs"] else "EPOCHS_COMPLETE",
                "completed_epochs": epoch,
                "target_epochs": config["epochs"],
                "best_epoch": best_epoch,
                "latest": row,
            },
            out_dir / "baseline_progress.json",
        )
        print(json.dumps({"event": "baseline_epoch", **row}), flush=True)

    if best_model_state is None or not math.isfinite(best_loss):
        raise RuntimeError("no finite validation-selected baseline checkpoint was produced")
    model.load_state_dict(best_model_state)
    predictions: Dict[str, torch.Tensor] = {}
    metrics_by_split: Dict[str, Dict[str, float]] = {}
    eval_time_by_split: Dict[str, float] = {}
    for name in ("train", "val", "test"):
        prediction, metrics, elapsed = _evaluate_baseline(
            model,
            operator,
            scoring_operator,
            pd_bus,
            target,
            effective,
            input_mean,
            input_std,
            target_mean,
            target_std,
            indices[name],
            config["batch_size"],
            execution_device,
        )
        predictions[name] = prediction
        metrics_by_split[name] = metrics
        eval_time_by_split[name] = elapsed
    payload = {
        "checkpoint_type": "selected_final",
        "protocol": PROTOCOL,
        "contract": contract,
        "dataset_path": str(data_path.resolve()),
        "dataset_sha256": contract["dataset_sha256"],
        "security_set_digest": contract["security_set_digest"],
        "config": config,
        "selected_epoch": best_epoch,
        "selection_validation_normalized_mse": best_loss,
        "model_state": _cpu_state_dict(model),
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "history": history,
        "metrics": metrics_by_split,
        "eval_time_sec": eval_time_by_split,
        "reference_test_prediction": predictions["test"],
        "reference_test_metrics": metrics_by_split["test"],
    }
    _atomic_torch_save(payload, final_path)
    return payload


def train(
    data_path: Path,
    out_dir: Path,
    requested_device: str,
    epochs: int,
    batch_size: int,
    baseline_batch_size: int,
    eval_batch_size: int,
    eval_every: int,
    seed: int,
    resume: bool,
) -> Dict[str, Any]:
    _validate_run_arguments(
        epochs, batch_size, baseline_batch_size, eval_batch_size, eval_every
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    execution_device = _execution_device(requested_device)
    data = load_audited_dataset(data_path)
    label_metrics = _label_self_check(data)
    deq_config = _deq_config(epochs, batch_size, seed)
    baseline_config = _baseline_config(epochs, baseline_batch_size, seed)
    contract = _contract(data_path, data, deq_config, baseline_config, eval_every)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "production_results.json"
    if resume and results_path.is_file():
        prior = json.loads(results_path.read_text(encoding="utf-8"))
        if prior.get("status") == "PRODUCTION_TRAIN_PASS" and prior.get("contract") == contract:
            print(json.dumps({"event": "already_complete", "path": str(results_path)}), flush=True)
            return prior

    deq = _train_deq(
        data_path,
        data,
        out_dir,
        execution_device,
        contract,
        eval_every,
        eval_batch_size,
        resume,
    )
    baseline = _train_baseline(
        data_path,
        data,
        out_dir,
        execution_device,
        contract,
        eval_every,
        resume,
    )
    split = data["split"].astype(np.int64)
    indices = {
        name: np.flatnonzero(split == value)
        for name, value in zip(("train", "val", "test"), (0, 1, 2))
    }
    deq_test = deq["metrics"]["test"]
    deq_val = deq["metrics"]["val"]
    gates = {
        "audited_label_operator": (
            label_metrics["balance_abs_max_mw"] <= 1e-6
            and label_metrics["all_contingency_violation_abs_max_mw"] <= 1e-4
        ),
        "deq_validation_all_batches_converged": deq["eval_convergence"]["val"][
            "all_batches_converged"
        ],
        "deq_test_all_batches_converged": deq["eval_convergence"]["test"][
            "all_batches_converged"
        ],
        "deq_validation_balance": deq_val["balance_abs_max_mw"] <= SECURE_TOL_MW,
        "deq_test_balance": deq_test["balance_abs_max_mw"] <= SECURE_TOL_MW,
        "deq_validation_security": deq_val[
            "all_contingency_secure_rate_at_1e3_mw"
        ]
        == 1.0,
        "deq_test_security": deq_test["all_contingency_secure_rate_at_1e3_mw"]
        == 1.0,
        "baseline_finite": all(
            np.isfinite(value)
            for metrics in baseline["metrics"].values()
            for value in metrics.values()
        ),
        "checkpoints_written": (out_dir / "deq_final.pt").is_file()
        and (out_dir / "baseline_final.pt").is_file(),
    }
    result = {
        "status": "PRODUCTION_TRAIN_PASS" if all(gates.values()) else "PRODUCTION_TRAIN_FAIL",
        "gates": gates,
        "contract": contract,
        "case": "case300",
        "scope": "preventive fixed declared K=32 line-contingency SC-DCOPF",
        "dataset": {
            "path": str(data_path.resolve()),
            "sha256": contract["dataset_sha256"],
            "N": int(len(split)),
            "split_counts": {name: int(len(value)) for name, value in indices.items()},
            "pd_bus_array_digest": _array_digest(data["pd_bus"]),
            "declared_contingencies": int(len(data["security_branch_rows"])),
            "security_set_digest": contract["security_set_digest"],
        },
        "label_operator_self_check": label_metrics,
        "deq": {
            "selected_epoch": deq["selected_epoch"],
            "selection_key": deq["selection_key"],
            "history": deq["history"],
            "metrics": deq["metrics"],
            "eval_convergence": deq["eval_convergence"],
            "eval_time_sec": deq["eval_time_sec"],
            "peak_cuda_memory_allocated_bytes": deq[
                "peak_cuda_memory_allocated_bytes"
            ],
            "peak_cuda_memory_reserved_bytes": deq[
                "peak_cuda_memory_reserved_bytes"
            ],
            "checkpoint": str((out_dir / "deq_final.pt").resolve()),
            "checkpoint_sha256": sha256_file(out_dir / "deq_final.pt"),
        },
        "baseline": {
            "selected_epoch": baseline["selected_epoch"],
            "selection_validation_normalized_mse": baseline[
                "selection_validation_normalized_mse"
            ],
            "history": baseline["history"],
            "metrics": baseline["metrics"],
            "eval_time_sec": baseline["eval_time_sec"],
            "checkpoint": str((out_dir / "baseline_final.pt").resolve()),
            "checkpoint_sha256": sha256_file(out_dir / "baseline_final.pt"),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(execution_device),
            "cuda_runtime": torch.version.cuda,
            "cuda_device_name": (
                torch.cuda.get_device_name(execution_device)
                if execution_device.type == "cuda"
                else None
            ),
        },
    }
    _atomic_json(result, results_path)
    if result["status"] != "PRODUCTION_TRAIN_PASS":
        raise RuntimeError(f"production training gates failed: {gates}")
    return result


def verify(
    data_path: Path,
    out_dir: Path,
    requested_device: str,
    eval_batch_size: int,
) -> Dict[str, Any]:
    execution_device = _execution_device(requested_device)
    data = load_audited_dataset(data_path)
    split = data["split"].astype(np.int64)
    test_indices = np.flatnonzero(split == 2)
    pd_bus, target, _ = _tensor_data(data)
    pd_bus64 = torch.as_tensor(data["pd_bus"], dtype=torch.float64)
    target64 = torch.as_tensor(data["pg_scopf"], dtype=torch.float64)
    scoring_operator = PreventiveSCOperator(data, dtype=torch.float64)

    deq_path = out_dir / "deq_final.pt"
    baseline_path = out_dir / "baseline_final.pt"
    deq_payload = torch.load(deq_path, map_location="cpu", weights_only=False)
    baseline_payload = torch.load(baseline_path, map_location="cpu", weights_only=False)
    dataset_hash = sha256_file(data_path)
    for name, payload in (("deq", deq_payload), ("baseline", baseline_payload)):
        if payload.get("protocol") != PROTOCOL or payload.get("dataset_sha256") != dataset_hash:
            raise ValueError(f"{name} final checkpoint contract mismatch")

    deq_steps = LearnableSteps(
        float(deq_payload["norm_k"]),
        s_init=float(deq_payload["config"]["validation_selected_s_init"]),
        s_max=float(deq_payload["config"]["s_max"]),
    )
    deq_steps.load_state_dict(deq_payload["steps_state"])
    deq_operator = PreventiveSCOperator(data, steps=deq_steps, dtype=torch.float64)
    deq_operator.to(execution_device)
    deq_model = DEQFixedPoint(deq_operator, deq_payload["config"])
    deq_prediction, deq_metrics, deq_elapsed, deq_residual, deq_convergence = _evaluate_deq(
        deq_model,
        deq_operator,
        pd_bus64,
        target64,
        test_indices,
        scoring_operator,
        pd_bus64,
        target64,
        batch_size=eval_batch_size,
    )
    deq_error = float(
        torch.max(
            torch.abs(deq_prediction - deq_payload["reference_test_prediction"])
        ).item()
    )

    operator = PreventiveSCOperator(data)
    baseline_model = MLP(operator.nbus, operator.ngen).to(execution_device)
    baseline_model.load_state_dict(baseline_payload["model_state"])
    effective = pd_bus + operator.gs_bus[None, :]
    baseline_prediction, baseline_metrics, baseline_elapsed = _evaluate_baseline(
        baseline_model,
        operator,
        scoring_operator,
        pd_bus,
        target,
        effective,
        baseline_payload["input_mean"],
        baseline_payload["input_std"],
        baseline_payload["target_mean"],
        baseline_payload["target_std"],
        test_indices,
        baseline_payload["config"]["batch_size"],
        execution_device,
    )
    baseline_error = float(
        torch.max(
            torch.abs(
                baseline_prediction - baseline_payload["reference_test_prediction"]
            )
        ).item()
    )
    if deq_error > 1e-9 or baseline_error > 1e-6:
        raise ValueError(
            f"fresh-process prediction mismatch: deq={deq_error}, baseline={baseline_error}"
        )
    verification = {
        "status": "PRODUCTION_CHECKPOINT_RELOAD_PASS",
        "fresh_process": True,
        "dataset_sha256": dataset_hash,
        "deq": {
            "checkpoint_sha256": sha256_file(deq_path),
            "prediction_abs_max_difference_mw": deq_error,
            "test_metrics": deq_metrics,
            "eval_time_sec": deq_elapsed,
            "fixed_point_residual_mean": deq_residual,
            "convergence": deq_convergence,
        },
        "baseline": {
            "checkpoint_sha256": sha256_file(baseline_path),
            "prediction_abs_max_difference_mw": baseline_error,
            "test_metrics": baseline_metrics,
            "eval_time_sec": baseline_elapsed,
        },
    }
    verification_path = out_dir / "checkpoint_reload.json"
    _atomic_json(verification, verification_path)
    result_path = out_dir / "production_results.json"
    source_paths = {
        "production_training_source": Path(__file__).resolve(),
        "solver_training_source": Path(__file__).resolve().parent
        / "train_phase_a_smoke.py",
        "common_source": Path(__file__).resolve().parent
        / "phase_a_scopf"
        / "common.py",
    }
    artifacts = {
        "dataset": data_path,
        "audited_provenance": data["provenance_path"],
        "production_results": result_path,
        "deq_checkpoint": deq_path,
        "baseline_checkpoint": baseline_path,
        "checkpoint_reload": verification_path,
        **source_paths,
    }
    manifest = {
        "manifest_schema": "phase_a_scopf_production_training_manifest_v1",
        "status": "PASS",
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
    }
    _atomic_json(manifest, out_dir / "production_training_manifest.json")
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("train", "verify"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=35)
    parser.add_argument("--baseline-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "train":
        result = train(
            args.data,
            args.out_dir,
            args.device,
            args.epochs,
            args.batch_size,
            args.baseline_batch_size,
            args.eval_batch_size,
            args.eval_every,
            args.seed,
            args.resume,
        )
    else:
        result = verify(
            args.data, args.out_dir, args.device, args.eval_batch_size
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
