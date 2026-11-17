#!/usr/bin/env python3
"""Local one-epoch DEQ and MLP smoke for preventive IEEE 300 SC-DCOPF.

The DEQ is the completed-study scalar-step PDHG fixed-point model with its
flow operator extended from one base topology to the base topology plus every
declared line outage simultaneously. The MLP is a supervised reference
baseline trained on exactly the same preventive dispatch target.
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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from phase_a_scopf.common import SCHEMA_NAME, SCHEMA_VERSION, sha256_file


PROTOCOL = "phase_a_preventive_sc_deq_smoke_v2"
SECURE_TOL_MW = 1e-3


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_torch_save(value: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_json(value: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_audited_dataset(path: Path) -> Dict[str, Any]:
    provenance_path = path.with_suffix(".provenance.json")
    if not provenance_path.is_file():
        raise FileNotFoundError(f"audited provenance is missing: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "AUDITED_PASS":
        raise ValueError("dataset provenance is not AUDITED_PASS")
    expected_data_hash = provenance["artifacts"]["dataset"]["sha256"]
    if sha256_file(path) != expected_data_hash:
        raise ValueError("dataset hash no longer matches audited provenance")
    audit_path = Path(provenance["artifacts"]["audit_report"]["path"])
    if not audit_path.is_file():
        # Audited provenance is generated on Windows, but the immutable
        # dataset bundle is also consumed on Vista.  Resolve the audit report
        # beside the dataset when the recorded host-specific absolute path is
        # unavailable; its SHA-256 contract remains mandatory below.
        portable_audit_path = path.with_suffix(".audit.json")
        if portable_audit_path.is_file():
            audit_path = portable_audit_path
    if not audit_path.is_file() or sha256_file(audit_path) != provenance["artifacts"]["audit_report"]["sha256"]:
        raise ValueError("audit report hash no longer matches provenance")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or not all(audit.get("gates", {}).values()):
        raise ValueError("dataset audit does not pass every gate")

    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    schema_name = str(np.asarray(data["dataset_schema_name"]).item())
    schema_version = int(np.asarray(data["dataset_schema_version"]).item())
    if schema_name != SCHEMA_NAME or schema_version != SCHEMA_VERSION:
        raise ValueError("unexpected preventive SC-DCOPF dataset schema")
    if "pg_scopf" not in data or "split" not in data:
        raise ValueError("training data is missing targets or fixed split")
    split = data["split"].astype(np.int64)
    counts = [int(np.count_nonzero(split == value)) for value in (0, 1, 2)]
    n = int(len(split))
    expected = [int(round(0.70 * n)), int(round(0.10 * n))]
    expected.append(n - expected[0] - expected[1])
    if counts != expected or set(np.unique(split).tolist()) != {0, 1, 2}:
        raise ValueError(
            f"expected deterministic 70/10/20-percent split {expected}, got {counts}"
        )
    data["provenance"] = provenance
    data["provenance_path"] = provenance_path
    return data


class LearnableSteps(nn.Module):
    def __init__(
        self,
        norm_k: float,
        rho_init: float = 0.9,
        s_init: float = 0.0,
        s_max: float = 3.0,
    ):
        super().__init__()
        if not -s_max < s_init < s_max:
            raise ValueError("s_init must lie strictly inside (-s_max, s_max)")
        probability = min(max(rho_init / 0.99, 1e-4), 1.0 - 1e-4)
        self.rho_logit = nn.Parameter(
            torch.tensor(math.log(probability / (1.0 - probability)), dtype=torch.float32)
        )
        self.s_raw = nn.Parameter(
            torch.tensor(math.atanh(s_init / s_max), dtype=torch.float32)
        )
        self.norm_k = float(norm_k)
        self.s_max = float(s_max)

    def rho(self) -> torch.Tensor:
        return 0.99 * torch.sigmoid(self.rho_logit)

    def s(self) -> torch.Tensor:
        return self.s_max * torch.tanh(self.s_raw)

    def tau_sigma(self, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        rho = self.rho().to(dtype=dtype, device=device)
        shift = self.s().to(dtype=dtype, device=device)
        scale = torch.sqrt(rho) / torch.tensor(self.norm_k, dtype=dtype, device=device)
        return scale * torch.exp(shift), scale * torch.exp(-shift)


class PreventiveSCOperator(nn.Module):
    """PDHG operator for one dispatch constrained in all declared states."""

    def __init__(
        self,
        data: Dict[str, Any],
        steps: LearnableSteps | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        arrays = {
            "ptdf": data["ptdf"],
            "flow_offset": data["flow_offset"],
            "fmax": data["fmax"],
            "pmin": data["pmin"],
            "pmax": data["pmax"],
            "c2": data["cost_c2"],
            "c1": data["cost_c1"],
            "c0": data["cost_c0"],
            "gs_bus": data["gs_bus"],
            "lodf": data["security_lodf_columns"].T,
        }
        for name, value in arrays.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=dtype))
        bus_of_gen = torch.as_tensor(data["bus_of_gen"], dtype=torch.long)
        self.register_buffer("bus_of_gen", bus_of_gen)
        generator_incidence = torch.zeros(
            int(self.ptdf.shape[1]),
            int(self.pmin.numel()),
            dtype=dtype,
        )
        generator_incidence[
            bus_of_gen, torch.arange(int(self.pmin.numel()), dtype=torch.long)
        ] = 1.0
        self.register_buffer("generator_incidence", generator_incidence)
        self.register_buffer(
            "security_rows",
            torch.as_tensor(data["security_branch_rows"], dtype=torch.long),
        )
        self.nbus = int(self.ptdf.shape[1])
        self.nbranch = int(self.ptdf.shape[0])
        self.ngen = int(self.pmin.numel())
        self.ncont = int(self.security_rows.numel())
        self.nstates = self.ncont + 1
        mask = torch.ones(self.nstates, self.nbranch, dtype=torch.bool)
        mask[torch.arange(self.ncont) + 1, self.security_rows] = False
        self.register_buffer("flow_mask", mask)
        self.steps = steps

    @property
    def state_dim(self) -> int:
        return self.ngen + 1 + 2 * self.nstates * self.nbranch

    def gen_to_bus(self, pg: torch.Tensor) -> torch.Tensor:
        return pg @ self.generator_incidence.T

    def effective_demand(self, pd_bus: torch.Tensor) -> torch.Tensor:
        return pd_bus + self.gs_bus[None, :]

    def expand_base_flows(self, base_flow: torch.Tensor) -> torch.Tensor:
        outage_flow = base_flow[:, self.security_rows]
        contingency = (
            base_flow[:, None, :]
            + self.lodf[None, :, :] * outage_flow[:, :, None]
        )
        return torch.cat([base_flow[:, None, :], contingency], dim=1)

    def flow_states(self, pg: torch.Tensor, pd_bus: torch.Tensor) -> torch.Tensor:
        injection = self.gen_to_bus(pg) - self.effective_demand(pd_bus)
        base = injection @ self.ptdf.T + self.flow_offset[None, :]
        return self.expand_base_flows(base)

    def generator_flow_states(self, pg: torch.Tensor) -> torch.Tensor:
        base = self.gen_to_bus(pg) @ self.ptdf.T
        return self.expand_base_flows(base)

    def constant_flow_states(self, pd_bus: torch.Tensor) -> torch.Tensor:
        base = self.flow_offset[None, :] - self.effective_demand(pd_bus) @ self.ptdf.T
        return self.expand_base_flows(base)

    def flow_adjoint(self, state_weights: torch.Tensor) -> torch.Tensor:
        base_weights = state_weights.sum(dim=1)
        contingency_weights = state_weights[:, 1:, :]
        outage_extra = (contingency_weights * self.lodf[None, :, :]).sum(dim=2)
        base_weights = base_weights.clone()
        base_weights.scatter_add_(
            1,
            self.security_rows[None, :].expand(state_weights.shape[0], -1),
            outage_extra,
        )
        bus_gradient = base_weights @ self.ptdf
        return bus_gradient[:, self.bus_of_gen]

    def split(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pg = z[:, : self.ngen]
        lam = z[:, self.ngen : self.ngen + 1]
        dual = z[:, self.ngen + 1 :]
        return pg, lam, dual

    def prox_cost_bounds(self, value: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        denominator = 1.0 + 2.0 * tau * self.c2[None, :]
        result = (value - tau * self.c1[None, :]) / denominator
        return torch.minimum(torch.maximum(result, self.pmin[None, :]), self.pmax[None, :])

    def forward(self, z: torch.Tensor, pd_bus: torch.Tensor) -> torch.Tensor:
        if self.steps is None:
            raise RuntimeError("PDHG forward requires learnable steps")
        pg, lam, dual = self.split(z)
        tau, sigma = self.steps.tau_sigma(z.dtype, z.device)
        bsz = pg.shape[0]
        dual = dual.reshape(bsz, 2, self.nstates, self.nbranch)
        generated = self.generator_flow_states(pg)
        constant = self.constant_flow_states(pd_bus)
        fmax = self.fmax[None, None, :]
        rhs_upper = fmax - constant
        rhs_lower = fmax + constant
        lam_next = lam + sigma * (
            pg.sum(dim=1, keepdim=True)
            - self.effective_demand(pd_bus).sum(dim=1, keepdim=True)
        )
        upper_next = F.relu(dual[:, 0] + sigma * (generated - rhs_upper))
        lower_next = F.relu(dual[:, 1] + sigma * (-generated - rhs_lower))
        flow_gradient = self.flow_adjoint(upper_next - lower_next)
        value = pg - tau * (lam_next.expand(-1, self.ngen) + flow_gradient)
        pg_next = self.prox_cost_bounds(value, tau)
        return torch.cat(
            [
                pg_next,
                lam_next,
                torch.stack([upper_next, lower_next], dim=1).reshape(bsz, -1),
            ],
            dim=1,
        )

    def objective(self, pg: torch.Tensor) -> torch.Tensor:
        return (
            self.c2[None, :] * pg.square()
            + self.c1[None, :] * pg
            + self.c0[None, :]
        ).sum(dim=1)

    def violation(self, pg: torch.Tensor, pd_bus: torch.Tensor) -> torch.Tensor:
        raw = torch.clamp(
            self.flow_states(pg, pd_bus).abs() - self.fmax[None, None, :],
            min=0.0,
        )
        return torch.where(self.flow_mask[None, :, :], raw, torch.zeros_like(raw))


@torch.no_grad()
def estimate_norm_k(operator: PreventiveSCOperator, iterations: int = 100) -> float:
    x = torch.randn(1, operator.ngen, dtype=operator.c2.dtype, device=operator.c2.device)
    x = x / x.norm().clamp_min(1e-12)
    for _ in range(iterations):
        flows = operator.generator_flow_states(x)
        x_new = x.sum(dim=1, keepdim=True).expand(-1, operator.ngen)
        x_new = x_new + operator.flow_adjoint(2.0 * flows)
        x = x_new / x_new.norm(dim=1, keepdim=True).clamp_min(1e-12)
    flows = operator.generator_flow_states(x)
    mapped_norm = torch.sqrt(x.sum(dim=1).square() + 2.0 * flows.square().sum(dim=(1, 2)))
    return float((mapped_norm / x.norm(dim=1).clamp_min(1e-12)).item())


def anderson(
    function,
    x0: torch.Tensor,
    *,
    m: int,
    lam: float,
    max_iter: int,
    tol: float,
) -> Tuple[torch.Tensor, List[float]]:
    batch, dimension = x0.shape
    x_memory = torch.zeros(batch, m, dimension, dtype=x0.dtype, device=x0.device)
    f_memory = torch.zeros_like(x_memory)
    x_memory[:, 0] = x0
    f_memory[:, 0] = function(x0)
    x_memory[:, 1] = f_memory[:, 0]
    f_memory[:, 1] = function(x_memory[:, 1])
    system = torch.zeros(batch, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    system[:, 0, 1:] = 1.0
    system[:, 1:, 0] = 1.0
    target = torch.zeros(batch, m + 1, 1, dtype=x0.dtype, device=x0.device)
    target[:, 0, 0] = 1.0
    residuals: List[float] = []
    used = 1
    for iteration in range(2, max_iter):
        count = min(iteration, m)
        residual_memory = f_memory[:, :count] - x_memory[:, :count]
        gram = torch.bmm(residual_memory, residual_memory.transpose(1, 2))
        system[:, 1 : count + 1, 1 : count + 1] = gram + lam * torch.eye(
            count, dtype=x0.dtype, device=x0.device
        )[None]
        local_system = system[:, : count + 1, : count + 1]
        local_target = target[:, : count + 1]
        solution, info = torch.linalg.solve_ex(local_system, local_target, check_errors=False)
        bad = (info != 0) | ~torch.isfinite(solution).all(dim=(1, 2))
        if bool(bad.any()):
            solution64 = torch.linalg.solve(
                local_system[bad].double(), local_target[bad].double()
            ).to(x0.dtype)
            solution = solution.clone()
            solution[bad] = solution64
        weights = solution[:, 1 : count + 1, 0]
        mixed_x = (weights[:, None, :] @ x_memory[:, :count]).squeeze(1)
        mixed_f = (weights[:, None, :] @ f_memory[:, :count]).squeeze(1)
        used = iteration % m
        x_memory[:, used] = mixed_f
        f_memory[:, used] = function(x_memory[:, used])
        relative = (f_memory[:, used] - x_memory[:, used]).norm(dim=1) / f_memory[
            :, used
        ].norm(dim=1).clamp_min(1e-12)
        # Every sample in a batch must converge; a batch mean can hide one bad
        # implicit solve and is therefore not a valid stopping statistic.
        residuals.append(float(relative.max().item()))
        if residuals[-1] < tol:
            break
    return x_memory[:, used], residuals


class DEQFixedPoint(nn.Module):
    def __init__(self, operator: PreventiveSCOperator, config: Dict[str, Any]):
        super().__init__()
        self.operator = operator
        self.config = config
        self.forward_residuals: List[float] = []
        self.backward_residuals: List[float] = []
        self.backward_iterations = 0
        self.forward_iterations = 0
        self.forward_converged = False
        self.forward_diagnostics: Dict[str, float] = {}

    @torch.no_grad()
    def _forward_diagnostics(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        pd_bus: torch.Tensor,
    ) -> Tuple[Dict[str, float], bool]:
        residual = current - previous
        pg, _, _ = self.operator.split(current)
        pg_residual, _, dual_residual = self.operator.split(residual)
        _, sigma = self.operator.steps.tau_sigma(current.dtype, current.device)
        balance = (
            pg.sum(dim=1)
            - self.operator.effective_demand(pd_bus).sum(dim=1)
        ).abs()
        bound = torch.maximum(
            (self.operator.pmin[None, :] - pg).clamp_min(0.0),
            (pg - self.operator.pmax[None, :]).clamp_min(0.0),
        ).max(dim=1).values
        flow = self.operator.violation(pg, pd_bus).reshape(pg.shape[0], -1).max(dim=1).values
        relative = residual.norm(dim=1) / current.norm(dim=1).clamp_min(1e-12)
        diagnostics = {
            "relative_fixed_point_residual_mean": float(relative.mean().item()),
            "relative_fixed_point_residual_abs_max": float(relative.max().item()),
            "pg_step_abs_max_mw": float(pg_residual.abs().max().item()),
            "balance_abs_max_mw": float(balance.max().item()),
            "generator_bound_violation_abs_max_mw": float(bound.max().item()),
            "all_contingency_violation_abs_max_mw": float(flow.max().item()),
            "dual_step_scaled_abs_max_mw": float(
                (dual_residual.abs() / sigma).max().item()
            ),
        }
        converged = bool(
            diagnostics["pg_step_abs_max_mw"]
            <= float(self.config["forward_pg_step_tol_mw"])
            and diagnostics["balance_abs_max_mw"]
            <= float(self.config["forward_balance_tol_mw"])
            and diagnostics["generator_bound_violation_abs_max_mw"]
            <= float(self.config["forward_bound_tol_mw"])
            and diagnostics["all_contingency_violation_abs_max_mw"]
            <= float(self.config["forward_flow_tol_mw"])
            and diagnostics["dual_step_scaled_abs_max_mw"]
            <= float(self.config["forward_dual_step_tol_mw"])
        )
        return diagnostics, converged

    def forward(self, pd_bus: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(
            pd_bus.shape[0],
            self.operator.state_dim,
            dtype=pd_bus.dtype,
            device=pd_bus.device,
        )

        def fixed_map(value: torch.Tensor) -> torch.Tensor:
            return self.operator(value, pd_bus)

        if self.config.get("forward_solver") != "adaptive_picard_physics_gated":
            raise ValueError("unsupported DEQ forward solver")
        self.forward_residuals = []
        self.forward_iterations = 0
        self.forward_converged = False
        self.forward_diagnostics = {}
        with torch.no_grad():
            previous = z
            minimum = int(self.config["forward_min_iter"])
            maximum = int(self.config["forward_max_iter"])
            check_every = int(self.config["forward_check_every"])
            for iteration in range(1, maximum + 1):
                z = fixed_map(previous)
                should_check = iteration >= minimum and (
                    iteration % check_every == 0 or iteration == maximum
                )
                if should_check:
                    diagnostics, converged = self._forward_diagnostics(
                        previous, z, pd_bus
                    )
                    self.forward_diagnostics = diagnostics
                    self.forward_residuals.append(
                        diagnostics["relative_fixed_point_residual_mean"]
                    )
                    if converged:
                        self.forward_converged = True
                        self.forward_iterations = iteration
                        break
                previous = z
            if self.forward_iterations == 0:
                self.forward_iterations = maximum
            z_star = z
        z = fixed_map(z_star)
        if self.training:
            detached = z.detach().clone().requires_grad_()
            mapped = self.operator(detached, pd_bus)

            def backward_hook(gradient: torch.Tensor) -> torch.Tensor:
                def linear_map(value: torch.Tensor) -> torch.Tensor:
                    jacobian_t = autograd.grad(
                        mapped, detached, value, retain_graph=True
                    )[0]
                    return jacobian_t + gradient

                if self.config.get("backward_solver") != "adaptive_picard_worst_sample":
                    raise ValueError("unsupported DEQ backward solver")
                with torch.no_grad():
                    result = gradient
                    self.backward_residuals = []
                    self.backward_iterations = 0
                    minimum = int(self.config["backward_min_iter"])
                    maximum = int(self.config["backward_max_iter"])
                    check_every = int(self.config["backward_check_every"])
                    for iteration in range(1, maximum + 1):
                        updated = linear_map(result)
                        should_check = iteration >= minimum and (
                            iteration % check_every == 0 or iteration == maximum
                        )
                        if should_check:
                            relative = (updated - result).norm(dim=1) / updated.norm(
                                dim=1
                            ).clamp_min(1e-30)
                            worst = float(relative.max().item())
                            self.backward_residuals.append(worst)
                            if worst <= float(self.config["backward_tol"]):
                                self.backward_iterations = iteration
                                break
                        result = updated
                    if self.backward_iterations == 0:
                        self.backward_iterations = maximum
                return result

            z.register_hook(backward_hook)
        return z


class MLP(nn.Module):
    def __init__(self, nbus: int, ngen: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(nbus, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ngen),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def balance_repair(operator: PreventiveSCOperator, pg: torch.Tensor, pd_bus: torch.Tensor) -> torch.Tensor:
    result = torch.minimum(torch.maximum(pg, operator.pmin[None, :]), operator.pmax[None, :])
    demand = operator.effective_demand(pd_bus).sum(dim=1, keepdim=True)
    for _ in range(4):
        residual = demand - result.sum(dim=1, keepdim=True)
        upward = (operator.pmax[None, :] - result).clamp_min(0.0)
        downward = (result - operator.pmin[None, :]).clamp_min(0.0)
        headroom = torch.where(residual >= 0.0, upward, downward)
        result = result + residual * headroom / headroom.sum(dim=1, keepdim=True).clamp_min(1e-12)
        result = torch.minimum(torch.maximum(result, operator.pmin[None, :]), operator.pmax[None, :])
    return result


@torch.no_grad()
def score(
    operator: PreventiveSCOperator,
    prediction: torch.Tensor,
    pd_bus: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    dtype = operator.ptdf.dtype
    prediction = prediction.to(dtype=dtype, device=operator.ptdf.device)
    pd_bus = pd_bus.to(dtype=dtype, device=operator.ptdf.device)
    target = target.to(dtype=dtype, device=operator.ptdf.device)
    balance = prediction.sum(dim=1) - operator.effective_demand(pd_bus).sum(dim=1)
    lower = (operator.pmin[None, :] - prediction).clamp_min(0.0)
    upper = (prediction - operator.pmax[None, :]).clamp_min(0.0)
    bound = torch.maximum(lower, upper).max(dim=1).values
    violation = operator.violation(prediction, pd_bus)
    base_worst = violation[:, 0].max(dim=1).values
    contingency_worst = violation[:, 1:].reshape(prediction.shape[0], -1).max(dim=1).values
    all_worst = torch.maximum(base_worst, contingency_worst)
    predicted_cost = operator.objective(prediction)
    target_cost = operator.objective(target)
    return {
        "balance_abs_mean_mw": float(balance.abs().mean().item()),
        "balance_abs_max_mw": float(balance.abs().max().item()),
        "generator_bound_violation_abs_max_mw": float(bound.max().item()),
        "base_flow_violation_mean_mw": float(base_worst.mean().item()),
        "base_flow_violation_abs_max_mw": float(base_worst.max().item()),
        "all_contingency_violation_mean_mw": float(all_worst.mean().item()),
        "all_contingency_violation_p95_mw": float(torch.quantile(all_worst, 0.95).item()),
        "all_contingency_violation_abs_max_mw": float(all_worst.max().item()),
        "all_contingency_secure_rate_at_1e3_mw": float(
            (all_worst <= SECURE_TOL_MW).float().mean().item()
        ),
        "dispatch_rmse_mw": float(torch.sqrt((prediction - target).square().mean()).item()),
        "cost_relative_gap_abs_mean": float(
            ((predicted_cost - target_cost).abs() / target_cost.abs().clamp_min(1.0)).mean().item()
        ),
    }


def _tensor_data(data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    pd_bus = torch.as_tensor(data["pd_bus"], dtype=torch.float32)
    target = torch.as_tensor(data["pg_scopf"], dtype=torch.float32)
    split = data["split"].astype(np.int64)
    return pd_bus, target, split


def _execution_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA execution was requested but torch.cuda.is_available() is false"
        )
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _evaluate_deq(
    deq: DEQFixedPoint,
    operator: PreventiveSCOperator,
    pd_bus: torch.Tensor,
    target: torch.Tensor,
    indices: np.ndarray,
    scoring_operator: PreventiveSCOperator,
    scoring_pd_bus: torch.Tensor,
    scoring_target: torch.Tensor,
    batch_size: int = 20,
) -> Tuple[torch.Tensor, Dict[str, float], float, float, Dict[str, Any]]:
    deq.eval()
    execution_device = operator.ptdf.device
    predictions = []
    residuals = []
    iterations = []
    convergence_rows = []
    _synchronize(execution_device)
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            z = deq(pd_bus[batch].to(execution_device))
            predictions.append(z[:, : operator.ngen].cpu())
            residuals.append(deq.forward_residuals[-1] if deq.forward_residuals else float("nan"))
            iterations.append(deq.forward_iterations)
            convergence_rows.append(
                {
                    "converged": deq.forward_converged,
                    **deq.forward_diagnostics,
                }
            )
    _synchronize(execution_device)
    elapsed = time.perf_counter() - started
    prediction = torch.cat(predictions, dim=0)
    metrics = score(
        scoring_operator,
        prediction,
        scoring_pd_bus[indices],
        scoring_target[indices],
    )
    diagnostic_keys = (
        "relative_fixed_point_residual_mean",
        "relative_fixed_point_residual_abs_max",
        "pg_step_abs_max_mw",
        "balance_abs_max_mw",
        "generator_bound_violation_abs_max_mw",
        "all_contingency_violation_abs_max_mw",
        "dual_step_scaled_abs_max_mw",
    )
    convergence = {
        "all_batches_converged": bool(
            convergence_rows and all(row["converged"] for row in convergence_rows)
        ),
        "iterations_mean": float(np.mean(iterations)),
        "iterations_max": int(max(iterations)),
        **{
            key: float(max(row[key] for row in convergence_rows))
            for key in diagnostic_keys
        },
    }
    return prediction, metrics, elapsed, float(np.nanmean(residuals)), convergence


def train_models(
    data_path: Path,
    out_dir: Path,
    requested_device: str = "cpu",
) -> Dict[str, Any]:
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    execution_device = _execution_device(requested_device)
    data = load_audited_dataset(data_path)
    pd_bus, target, split = _tensor_data(data)
    deq_pd_bus = torch.as_tensor(data["pd_bus"], dtype=torch.float64)
    scoring_pd_bus = torch.as_tensor(data["pd_bus"], dtype=torch.float64)
    scoring_target = torch.as_tensor(data["pg_scopf"], dtype=torch.float64)
    indices = {name: np.flatnonzero(split == value) for name, value in zip(("train", "val", "test"), (0, 1, 2))}

    static_operator = PreventiveSCOperator(data)
    scoring_operator = PreventiveSCOperator(data, dtype=torch.float64)
    label_metrics = score(
        scoring_operator, scoring_target, scoring_pd_bus, scoring_target
    )
    if (
        label_metrics["balance_abs_max_mw"] > 1e-3
        or label_metrics["all_contingency_violation_abs_max_mw"] > 1e-3
        or label_metrics["dispatch_rmse_mw"] != 0.0
    ):
        raise ValueError(f"training operator does not reproduce audited labels: {label_metrics}")

    norm_k = estimate_norm_k(static_operator, iterations=100)
    deq_config = {
        "epochs": 1,
        "batch_size": 35,
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
        "seed": 0,
        "step_mode": "scalar",
        "execution_device": str(execution_device),
    }
    steps = LearnableSteps(
        norm_k,
        s_init=deq_config["validation_selected_s_init"],
        s_max=deq_config["s_max"],
    )
    deq_operator = PreventiveSCOperator(data, steps=steps, dtype=torch.float64)
    deq_operator.to(execution_device)
    deq = DEQFixedPoint(deq_operator, deq_config)
    optimizer = torch.optim.Adam(steps.parameters(), lr=deq_config["learning_rate"])
    train_order = indices["train"].copy()
    rng = np.random.default_rng(0)
    rng.shuffle(train_order)
    if execution_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(execution_device)
    _synchronize(execution_device)
    deq_started = time.perf_counter()
    train_loss_values = []
    train_fwd_residuals = []
    train_fwd_iterations = []
    train_fwd_converged = []
    train_bwd_residuals = []
    train_bwd_iterations = []
    train_step_gradient_abs_max = []
    for start in range(0, len(train_order), deq_config["batch_size"]):
        batch = train_order[start : start + deq_config["batch_size"]]
        deq.train()
        batch_pd_bus = deq_pd_bus[batch].to(execution_device)
        z = deq(batch_pd_bus)
        prediction = z[:, : deq_operator.ngen]
        cost = deq_operator.objective(prediction) / deq_operator.effective_demand(
            batch_pd_bus
        ).sum(dim=1).clamp_min(1e-9)
        balance = prediction.sum(dim=1) - deq_operator.effective_demand(
            batch_pd_bus
        ).sum(dim=1)
        violation = deq_operator.violation(prediction, batch_pd_bus)
        loss = (
            deq_config["w_cost"] * cost.mean()
            + deq_config["w_balance"] * balance.square().mean()
            + deq_config["w_flow_quadratic"] * violation.square().mean()
            + deq_config["w_flow_l1"] * violation.mean()
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("DEQ smoke loss is nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in steps.parameters():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("DEQ scalar-step gradient is missing or nonfinite")
        train_step_gradient_abs_max.append(
            max(float(parameter.grad.detach().abs().max().item()) for parameter in steps.parameters())
        )
        optimizer.step()
        train_loss_values.append(float(loss.detach().item()))
        train_fwd_residuals.append(deq.forward_residuals[-1])
        train_fwd_iterations.append(deq.forward_iterations)
        train_fwd_converged.append(deq.forward_converged)
        train_bwd_residuals.append(deq.backward_residuals[-1])
        train_bwd_iterations.append(deq.backward_iterations)
    _synchronize(execution_device)
    deq_train_time = time.perf_counter() - deq_started
    deq_predictions = {}
    deq_metrics = {}
    deq_eval_time = {}
    deq_residual = {}
    deq_convergence = {}
    for name in ("train", "val", "test"):
        prediction, metrics, elapsed, residual, convergence = _evaluate_deq(
            deq,
            deq_operator,
            deq_pd_bus,
            scoring_target,
            indices[name],
            scoring_operator,
            scoring_pd_bus,
            scoring_target,
        )
        deq_predictions[name] = prediction
        deq_metrics[name] = metrics
        deq_eval_time[name] = elapsed
        deq_residual[name] = residual
        deq_convergence[name] = convergence

    deq_checkpoint = out_dir / "deq_one_epoch.pt"
    deq_payload = {
        "protocol": PROTOCOL,
        "model": "preventive_sc_pdhg_deq_scalar_steps_float64_physics_gated",
        "dataset_path": str(data_path.resolve()),
        "dataset_sha256": sha256_file(data_path),
        "security_set_digest": data["provenance"]["configuration"]["security_set_digest"],
        "config": deq_config,
        "norm_k": norm_k,
        "steps_state": steps.state_dict(),
        "reference_test_prediction": deq_predictions["test"],
        "reference_test_metrics": deq_metrics["test"],
    }
    _atomic_torch_save(deq_payload, deq_checkpoint)

    torch.manual_seed(0)
    effective = pd_bus + static_operator.gs_bus[None, :]
    train_effective = effective[indices["train"]]
    input_mean = train_effective.mean(dim=0, keepdim=True)
    input_std = train_effective.std(dim=0, keepdim=True).clamp_min(1e-6)
    target_mean = target[indices["train"]].mean(dim=0, keepdim=True)
    target_std = target[indices["train"]].std(dim=0, keepdim=True).clamp_min(1e-6)
    baseline = MLP(static_operator.nbus, static_operator.ngen)
    baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    baseline_started = time.perf_counter()
    baseline.train()
    normalized_input = (train_effective - input_mean) / input_std
    normalized_target = (target[indices["train"]] - target_mean) / target_std
    baseline_prediction = baseline(normalized_input)
    baseline_loss = (baseline_prediction - normalized_target).square().mean()
    if not bool(torch.isfinite(baseline_loss)):
        raise FloatingPointError("baseline smoke loss is nonfinite")
    baseline_optimizer.zero_grad(set_to_none=True)
    baseline_loss.backward()
    if any(
        parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
        for parameter in baseline.parameters()
    ):
        raise FloatingPointError("baseline gradient is missing or nonfinite")
    baseline_optimizer.step()
    baseline_train_time = time.perf_counter() - baseline_started
    baseline_predictions = {}
    baseline_metrics = {}
    baseline_eval_time = {}
    baseline.eval()
    with torch.no_grad():
        for name in ("train", "val", "test"):
            started = time.perf_counter()
            raw = (
                baseline((effective[indices[name]] - input_mean) / input_std)
                * target_std
                + target_mean
            )
            repaired = balance_repair(static_operator, raw, pd_bus[indices[name]])
            baseline_eval_time[name] = time.perf_counter() - started
            baseline_predictions[name] = repaired.cpu()
            baseline_metrics[name] = score(
                scoring_operator,
                repaired,
                scoring_pd_bus[indices[name]],
                scoring_target[indices[name]],
            )
    baseline_checkpoint = out_dir / "mlp_one_epoch.pt"
    baseline_payload = {
        "protocol": PROTOCOL,
        "model": "supervised_mlp_balance_repaired",
        "dataset_path": str(data_path.resolve()),
        "dataset_sha256": sha256_file(data_path),
        "security_set_digest": data["provenance"]["configuration"]["security_set_digest"],
        "config": {"epochs": 1, "learning_rate": 1e-3, "seed": 0, "hidden": [256, 256]},
        "model_state": baseline.state_dict(),
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "reference_test_prediction": baseline_predictions["test"],
        "reference_test_metrics": baseline_metrics["test"],
    }
    _atomic_torch_save(baseline_payload, baseline_checkpoint)

    training_gates = {
        "audited_label_operator": (
            label_metrics["balance_abs_max_mw"] <= 1e-6
            and label_metrics["all_contingency_violation_abs_max_mw"] <= 1e-4
        ),
        "deq_forward_backward_finite": bool(
            np.all(np.isfinite(train_loss_values))
            and np.all(np.isfinite(train_fwd_residuals))
            and np.all(np.isfinite(train_bwd_residuals))
        ),
        "deq_configured_forward_tolerance": bool(
            deq_convergence["val"]["all_batches_converged"]
        ),
        "deq_configured_backward_tolerance": bool(
            max(train_bwd_residuals) <= deq_config["backward_tol"]
        ),
        "deq_validation_balance": bool(
            deq_metrics["val"]["balance_abs_max_mw"] <= SECURE_TOL_MW
        ),
        "deq_validation_all_contingency_security": bool(
            deq_metrics["val"]["all_contingency_secure_rate_at_1e3_mw"] == 1.0
        ),
        "baseline_forward_backward_finite": bool(
            np.isfinite(float(baseline_loss.detach().item()))
        ),
        "checkpoints_written": deq_checkpoint.is_file() and baseline_checkpoint.is_file(),
    }
    mechanics_pass = all(
        training_gates[name]
        for name in (
            "audited_label_operator",
            "deq_forward_backward_finite",
            "baseline_forward_backward_finite",
            "checkpoints_written",
        )
    )
    readiness_pass = all(training_gates.values())
    result = {
        "status": (
            "TRAIN_AND_CONVERGENCE_PASS"
            if readiness_pass
            else "TRAIN_MECHANICS_PASS_DEQ_CONVERGENCE_BLOCKED"
            if mechanics_pass
            else "TRAIN_SMOKE_FAIL"
        ),
        "training_gates": training_gates,
        "production_training_readiness": "PASS" if readiness_pass else "BLOCKED",
        "protocol": PROTOCOL,
        "case": "case300",
        "dataset": {
            "path": str(data_path.resolve()),
            "sha256": sha256_file(data_path),
            "audited_provenance": str(data["provenance_path"].resolve()),
            "N": int(pd_bus.shape[0]),
            "split_counts": {name: int(len(value)) for name, value in indices.items()},
            "declared_contingencies": int(static_operator.ncont),
            "security_set_digest": data["provenance"]["configuration"]["security_set_digest"],
        },
        "label_operator_self_check": label_metrics,
        "deq": {
            "config": deq_config,
            "norm_k": norm_k,
            "train_loss_mean": float(np.mean(train_loss_values)),
            "train_forward_residual_mean": float(np.mean(train_fwd_residuals)),
            "train_forward_iterations_mean": float(np.mean(train_fwd_iterations)),
            "train_forward_all_batches_converged": bool(all(train_fwd_converged)),
            "train_backward_residual_mean": float(np.mean(train_bwd_residuals)),
            "train_backward_residual_abs_max": float(max(train_bwd_residuals)),
            "train_backward_iterations_mean": float(np.mean(train_bwd_iterations)),
            "train_step_gradient_abs_max": float(max(train_step_gradient_abs_max)),
            "rho": float(steps.rho().detach().item()),
            "s": float(steps.s().detach().item()),
            "train_time_sec": deq_train_time,
            "train_samples_per_sec": float(len(indices["train"]) / deq_train_time),
            "eval_time_sec": deq_eval_time,
            "eval_samples_per_sec": float(
                len(split) / sum(deq_eval_time.values())
            ),
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
            "eval_forward_residual": deq_residual,
            "eval_convergence": deq_convergence,
            "metrics": deq_metrics,
            "checkpoint": str(deq_checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(deq_checkpoint),
            "parameters": int(sum(parameter.numel() for parameter in steps.parameters())),
        },
        "baseline": {
            "model": "MLP-256-256 with box clamp and balance repair",
            "epochs": 1,
            "train_loss": float(baseline_loss.detach().item()),
            "train_time_sec": baseline_train_time,
            "eval_time_sec": baseline_eval_time,
            "metrics": baseline_metrics,
            "checkpoint": str(baseline_checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(baseline_checkpoint),
            "parameters": int(sum(parameter.numel() for parameter in baseline.parameters())),
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
            "cuda_device_memory_bytes": (
                int(torch.cuda.get_device_properties(execution_device).total_memory)
                if execution_device.type == "cuda"
                else None
            ),
            "torch_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(result, out_dir / "one_epoch_results.json")
    return result


def verify_checkpoints(
    data_path: Path,
    out_dir: Path,
    requested_device: str = "cpu",
) -> Dict[str, Any]:
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    execution_device = _execution_device(requested_device)
    data = load_audited_dataset(data_path)
    pd_bus, target, split = _tensor_data(data)
    scoring_pd_bus = torch.as_tensor(data["pd_bus"], dtype=torch.float64)
    scoring_target = torch.as_tensor(data["pg_scopf"], dtype=torch.float64)
    test_indices = np.flatnonzero(split == 2)
    dataset_hash = sha256_file(data_path)

    deq_checkpoint_path = out_dir / "deq_one_epoch.pt"
    baseline_checkpoint_path = out_dir / "mlp_one_epoch.pt"
    deq_payload = torch.load(deq_checkpoint_path, map_location="cpu", weights_only=False)
    baseline_payload = torch.load(
        baseline_checkpoint_path, map_location="cpu", weights_only=False
    )
    for name, payload in (("deq", deq_payload), ("baseline", baseline_payload)):
        if payload.get("protocol") != PROTOCOL or payload.get("dataset_sha256") != dataset_hash:
            raise ValueError(f"{name} checkpoint contract mismatch")

    deq_steps = LearnableSteps(
        float(deq_payload["norm_k"]),
        s_init=float(deq_payload["config"]["validation_selected_s_init"]),
        s_max=float(deq_payload["config"]["s_max"]),
    )
    deq_steps.load_state_dict(deq_payload["steps_state"])
    deq_operator = PreventiveSCOperator(
        data, steps=deq_steps, dtype=torch.float64
    )
    deq_operator.to(execution_device)
    deq = DEQFixedPoint(deq_operator, deq_payload["config"])
    deq_pd_bus = torch.as_tensor(data["pd_bus"], dtype=torch.float64)
    deq_prediction, deq_metrics, deq_elapsed, deq_residual, deq_convergence = _evaluate_deq(
        deq,
        deq_operator,
        deq_pd_bus,
        scoring_target,
        test_indices,
        PreventiveSCOperator(data, dtype=torch.float64),
        scoring_pd_bus,
        scoring_target,
    )
    deq_prediction_error = float(
        torch.max(torch.abs(deq_prediction - deq_payload["reference_test_prediction"])).item()
    )

    static_operator = PreventiveSCOperator(data)
    baseline = MLP(static_operator.nbus, static_operator.ngen)
    baseline.load_state_dict(baseline_payload["model_state"])
    baseline.eval()
    effective = pd_bus[test_indices] + static_operator.gs_bus[None, :]
    with torch.no_grad():
        raw = (
            baseline(
                (effective - baseline_payload["input_mean"])
                / baseline_payload["input_std"]
            )
            * baseline_payload["target_std"]
            + baseline_payload["target_mean"]
        )
        baseline_prediction = balance_repair(
            static_operator, raw, pd_bus[test_indices]
        )
    baseline_metrics = score(
        PreventiveSCOperator(data, dtype=torch.float64),
        baseline_prediction,
        scoring_pd_bus[test_indices],
        scoring_target[test_indices],
    )
    baseline_prediction_error = float(
        torch.max(
            torch.abs(
                baseline_prediction - baseline_payload["reference_test_prediction"]
            )
        ).item()
    )
    if deq_prediction_error > 1e-9 or baseline_prediction_error > 1e-6:
        raise ValueError(
            f"checkpoint prediction reload mismatch: deq={deq_prediction_error}, "
            f"baseline={baseline_prediction_error}"
        )
    verification = {
        "status": "CHECKPOINT_RELOAD_PASS",
        "fresh_process": True,
        "dataset_sha256": dataset_hash,
        "deq": {
            "checkpoint_sha256": sha256_file(deq_checkpoint_path),
            "prediction_abs_max_difference_mw": deq_prediction_error,
            "test_metrics": deq_metrics,
            "eval_time_sec": deq_elapsed,
            "forward_residual": deq_residual,
            "forward_convergence": deq_convergence,
            "execution_device": str(execution_device),
            "cuda_device_name": (
                torch.cuda.get_device_name(execution_device)
                if execution_device.type == "cuda"
                else None
            ),
        },
        "baseline": {
            "checkpoint_sha256": sha256_file(baseline_checkpoint_path),
            "prediction_abs_max_difference_mw": baseline_prediction_error,
            "test_metrics": baseline_metrics,
        },
    }
    _atomic_json(verification, out_dir / "checkpoint_reload.json")
    source = Path(__file__).resolve()
    training_results = json.loads(
        (out_dir / "one_epoch_results.json").read_text(encoding="utf-8")
    )
    readiness = training_results.get("production_training_readiness", "UNKNOWN")
    manifest = {
        "manifest_schema": "phase_a_scopf_training_smoke_manifest_v1",
        "status": (
            "PASS" if readiness == "PASS" else "CHECKPOINT_RELOAD_PASS_WITH_TRAINING_BLOCKER"
        ),
        "production_training_readiness": readiness,
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in {
                "dataset": data_path,
                "audited_provenance": data["provenance_path"],
                "training_results": out_dir / "one_epoch_results.json",
                "deq_checkpoint": deq_checkpoint_path,
                "baseline_checkpoint": baseline_checkpoint_path,
                "checkpoint_reload": out_dir / "checkpoint_reload.json",
                "training_source": source,
            }.items()
        },
    }
    _atomic_json(manifest, out_dir / "training_smoke_manifest.json")
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("train", "verify"), required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.mode == "train":
        result = train_models(args.data, args.out_dir, requested_device=args.device)
    else:
        result = verify_checkpoints(
            args.data, args.out_dir, requested_device=args.device
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
