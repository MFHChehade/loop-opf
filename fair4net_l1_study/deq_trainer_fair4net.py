#!/usr/bin/env python3
"""
dc_opf_train_deq_steps_selfsup_case300_9k_gsfix_v2.py

Self-supervised training of DEQ/PDHG step sizes for large DC-OPF (incl. 9k bus).

Loss (Option A):
  L = w_cost * J_cost(p) + w_bal * J_bal(p,d) + w_flow * J_flow(p,d)

- No supervision on pg is required.
- If the dataset contains pg labels, RMSE/cost-gap are reported as diagnostics only.

Key 9k changes vs the dense case300 selfsup script:
- Operator-form PDHG updates with sparse PTDF and scatter-add for Gp (no dense G/F/A).
- Operator-form ||K|| estimation.

Expected inputs:
- Dataset npz with at least:  pd_bus  (N, nbus)
- Optional: pg (N, ngen) for diagnostics (not used in loss).

Example (9k stable defaults):
  python dc_opf_train_deq_steps_selfsup_case300_9k.py ^
    --data io/dc_opf_dataset_case9241pegase_train1000.npz ^
    --device cuda --epochs 30 --batch_size 16 --lr 1e-3 ^
    --picard_iters 200 --anderson_m 10 --anderson_max_iter 800 ^
    --w_cost 1.0 --w_bal 50.0 --w_flow 50.0 ^
    --no_eval_test_each_epoch ^
    --save_plots ^
    --out_dir io/case9241_selfsup --ckpt_name deq_9k_selfsup.pt
"""

from __future__ import annotations

import sys
# --- Windows/PowerShell UTF-8 stdout safety ---
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import copy
import argparse
import hashlib
import time
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


# ============================================================
# Shared single-source settings (edit there only)
# ============================================================
try:
    import dc_opf_pandapower_case300_9k as shared
except Exception as e:
    raise ImportError(
        "Could not import dc_opf_pandapower_case300_9k.py. "
        "Place this training script in the same folder as the shared case file."
    ) from e

CODE_VERSION = "fair4net-l1-v2-phase-aware-resumable"
FLOW_MODEL_CONTRACT = "f=PTDF@(Pg-PD-GS)+phase_shift_offset"


def sha256_array(value: np.ndarray) -> str:
    """Stable digest for small static arrays used in result provenance."""
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(str(a.shape).encode("ascii"))
    h.update(a.view(np.uint8))
    return h.hexdigest()

CASE_NAME = shared.CASE_NAME
LINE_LIMIT_SCALE = shared.LINE_LIMIT_SCALE
LOAD_SCALE = shared.LOAD_SCALE
GEN_MARGIN_MW = shared.GEN_MARGIN_MW
VM_BOUNDS = shared.VM_BOUNDS
SEED = shared.SEED

load_ieee_case = shared.load_ieee_case
ensure_controllable = shared.ensure_controllable
set_voltage_limits = shared.set_voltage_limits
scale_loads = shared.scale_loads
scale_line_and_trafo_limits = shared.scale_line_and_trafo_limits
add_heterogeneous_quadratic_costs_and_bounds = shared.add_heterogeneous_quadratic_costs_and_bounds


# ============================================================
# Config
# ============================================================
@dataclass
class TrainCfg:
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    # Forward fixed-point solve
    picard_iters: int = 200
    anderson_m: int = 10
    anderson_lam: float = 1e-3
    anderson_max_iter: int = 800
    anderson_tol: float = 1e-10
    anderson_beta: float = 1.0

    # Backward fixed-point solve
    b_anderson_m: int = 10
    b_anderson_lam: float = 1e-4
    b_anderson_max_iter: int = 100
    b_anderson_tol: float = 1e-6
    b_anderson_beta: float = 1.0

    # Training
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.0
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed_split: int = 0

    # Step-size parameterization clamp
    s_max: float = 1.0  # clamp s via tanh to [-s_max, s_max]

    # Self-supervised weights
    w_cost: float = 1.0
    w_bal: float = 50.0
    w_flow: float = 50.0          # quadratic flow penalty (gradient vanishes near feasibility)
    w_flow_l1: float = 0.0        # linear (exact) flow penalty; non-vanishing gradient -> drives viol lower

    # Normalization eps
    eps: float = 1e-9

    # Diagnostics (does NOT affect loss)
    track_fixed_point: bool = True
    track_rmse_if_labels_exist: bool = True

    # Constraint satisfaction thresholds (MW)
    sat_tol_MW: float = 1e-3

    # Evaluation/test frequency
    eval_test_each_epoch: bool = True

    # Save
    out_dir: str = "io/selfsup"
    ckpt_name: str = "dc_opf_train_deq_steps_selfsup.pt"

    # Publishable outputs / reporting
    save_plots: bool = False
    plot_format: str = "pdf"   # pdf|png|svg
    plot_dpi: int = 300

    # Data export
    save_csv: bool = True
    save_history_every_epoch: bool = True

    # Post-train eval vs GT (requires pg in dataset)
    eval_vs_gt: bool = False
    eval_split: str = "val"  # train|val|all


# ============================================================
# PyPower/PTDF extraction (version-tolerant)
# ============================================================
def _import_pypower():
    try:
        from pypower.ext2int import ext2int
        from pypower.makeBdc import makeBdc
        from pypower.makePTDF import makePTDF
        from pypower.idx_bus import BUS_TYPE, REF, GS
        from pypower.idx_gen import GEN_STATUS, GEN_BUS, PMAX, PMIN
        from pypower.idx_cost import POLYNOMIAL
        from pypower.idx_brch import BR_X, RATE_A, SHIFT, TAP
        return (
            ext2int, makeBdc, makePTDF, BUS_TYPE, REF, GS, GEN_STATUS,
            GEN_BUS, PMAX, PMIN, POLYNOMIAL, BR_X, RATE_A, SHIFT, TAP,
        )
    except Exception:
        from pandapower.pypower.ext2int import ext2int
        from pandapower.pypower.makeBdc import makeBdc
        from pandapower.pypower.makePTDF import makePTDF
        from pandapower.pypower.idx_bus import BUS_TYPE, REF, GS
        from pandapower.pypower.idx_gen import GEN_STATUS, GEN_BUS, PMAX, PMIN
        from pandapower.pypower.idx_cost import POLYNOMIAL
        from pandapower.pypower.idx_brch import BR_X, RATE_A, SHIFT, TAP
        return (
            ext2int, makeBdc, makePTDF, BUS_TYPE, REF, GS, GEN_STATUS,
            GEN_BUS, PMAX, PMIN, POLYNOMIAL, BR_X, RATE_A, SHIFT, TAP,
        )


def _dense_1d(x: Any) -> np.ndarray:
    """Convert dense/sparse PYPOWER vectors to a flat float64 array."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _compute_phase_shift_flow_offset(
    makeBdc,
    baseMVA: float,
    bus: np.ndarray,
    branch: np.ndarray,
    ptdf: np.ndarray,
) -> np.ndarray:
    """Return the constant MW branch-flow term induced by phase shifters.

    PYPOWER and pandapower ship two compatible ``makeBdc`` variants with
    different call signatures.  Both return Pbusinj and Pfinj in per-unit.
    For an MW bus-injection vector x, the complete DC branch flow is

        PTDF @ x + baseMVA * (Pfinj - PTDF @ Pbusinj).
    """
    try:
        bdc = makeBdc(baseMVA, bus, branch)
    except TypeError:
        bdc = makeBdc(bus, branch)
    if len(bdc) < 4:
        raise RuntimeError("makeBdc did not return Pbusinj and Pfinj")
    pbusinj = _dense_1d(bdc[2])
    pfinj = _dense_1d(bdc[3])
    offset = float(baseMVA) * (pfinj - np.asarray(ptdf, dtype=np.float64) @ pbusinj)
    if offset.shape != (branch.shape[0],) or not np.all(np.isfinite(offset)):
        raise RuntimeError("Invalid phase-shift branch-flow offset")
    return offset.astype(np.float32)


def build_ppc_from_net(net) -> Dict[str, Any]:
    import pandapower as pp
    pp.rundcopp(net, verbose=False)
    if hasattr(net, "_ppc") and isinstance(net._ppc, dict):
        return net._ppc
    raise RuntimeError("Could not build ppc from pandapower net. Check pandapower version.")


def extract_static_mats(ppc: Dict[str, Any]) -> Dict[str, Any]:
    (
        ext2int, makeBdc, makePTDF, BUS_TYPE, REF, GS, GEN_STATUS,
        GEN_BUS, PMAX, PMIN, POLYNOMIAL, BR_X, RATE_A, SHIFT, TAP,
    ) = _import_pypower()

    ppc_i = ext2int(ppc.copy())
    baseMVA = float(ppc_i["baseMVA"])
    bus = ppc_i["bus"]
    branch = ppc_i["branch"]
    gen = ppc_i["gen"]
    gencost = ppc_i["gencost"]

    ref_buses = np.where(bus[:, BUS_TYPE] == REF)[0]
    ref = int(ref_buses[0]) if len(ref_buses) else 0

    on = np.where(gen[:, GEN_STATUS] > 0)[0]
    gen = gen[on, :]
    gencost = gencost[on, :]

    nbus = int(bus.shape[0])
    ngen = int(gen.shape[0])
    nbranch = int(branch.shape[0])

    bus_of_gen = gen[:, GEN_BUS].astype(int)
    # PYPOWER/MATPOWER balance includes fixed active bus shunts GS (MW at
    # V=1 p.u.).  The dataset's pd_bus contains conventional load PD only, so
    # retain GS as a separate static demand term instead of silently changing
    # the meaning of pd_bus.
    gs_bus = bus[:, GS].astype(np.float32)

    fmax = branch[:, RATE_A].astype(float)
    big = 1e9
    fmax = np.where(fmax <= 0.0, big, fmax).astype(np.float32)

    # PEGASE PTDF is effectively dense; COO sparse indices are larger and can
    # hang during torch sparse coalesce on Windows.
    PTDF64 = np.asarray(makePTDF(baseMVA, bus, branch, ref), dtype=np.float64)  # (m, nbus)
    flow_offset = _compute_phase_shift_flow_offset(makeBdc, baseMVA, bus, branch, PTDF64)
    PTDF = PTDF64.astype(np.float32, copy=False)
    phase_shift_nonzero_count = int(np.count_nonzero(np.abs(branch[:, SHIFT]) > 1e-12))
    branch_x = branch[:, BR_X].astype(np.float32)
    branch_tap = branch[:, TAP].astype(np.float32)
    # MATPOWER encodes a nominal transformer ratio as zero.
    branch_tap = np.where(np.abs(branch_tap) <= 1e-12, 1.0, branch_tap).astype(np.float32)
    branch_shift_deg = branch[:, SHIFT].astype(np.float32)

    pmin = gen[:, PMIN].astype(np.float32)
    pmax = gen[:, PMAX].astype(np.float32)

    # quadratic costs
    model = gencost[:, 0].astype(int)
    if not np.all(model == POLYNOMIAL):
        raise RuntimeError("Non-polynomial gencost encountered. Use quadratic poly_cost only.")
    ncost = gencost[:, 3].astype(int)

    c2 = np.zeros(ngen, dtype=np.float32)
    c1 = np.zeros(ngen, dtype=np.float32)
    for i in range(ngen):
        nc = int(ncost[i])
        coeffs = gencost[i, 4:4 + nc].astype(np.float32)
        if nc == 3:
            c2[i], c1[i], _ = coeffs
        else:
            raise RuntimeError(f"NCOST={nc} not supported. Use quadratic (NCOST=3).")
    if np.any(c2 <= 0.0):
        raise RuntimeError("Need strictly convex quadratic costs (c2>0).")

    return dict(
        nbus=nbus, ngen=ngen, nbranch=nbranch,
        ref_bus=ref,
        bus_of_gen=bus_of_gen.astype(np.int64),
        branch_x=branch_x,
        branch_tap=branch_tap,
        branch_shift_deg=branch_shift_deg,
        ptdf_dense=PTDF,
        gs_bus=gs_bus,
        flow_offset=flow_offset,
        flow_model_contract=FLOW_MODEL_CONTRACT,
        phase_shift_nonzero_count=phase_shift_nonzero_count,
        flow_offset_abs_max_MW=float(np.max(np.abs(flow_offset))) if flow_offset.size else 0.0,
        fmax=fmax,
        pmin=pmin, pmax=pmax,
        c2=c2, c1=c1,
    )


def extract_lodf(ppc: Dict[str, Any], static: Dict[str, Any]) -> np.ndarray:
    """Return a finite base-order LODF matrix for N-1 line scenarios.

    The compact line-contingency dataset stores only ``cont_line_idx``.  The
    operator applies the selected LODF column through a rank-one update, so we
    do not materialize one PTDF per sample.
    """
    try:
        from pypower.ext2int import ext2int
        from pypower.makeLODF import makeLODF
    except Exception:
        from pandapower.pypower.ext2int import ext2int
        from pandapower.pypower.makeLODF import makeLODF
    ppc_i = ext2int(ppc.copy())
    branch = np.asarray(ppc_i["branch"], dtype=float)
    lodf = np.asarray(makeLODF(branch, np.asarray(static["ptdf_dense"], dtype=float)), dtype=float)
    if lodf.shape != (int(static["nbranch"]), int(static["nbranch"])):
        raise RuntimeError(f"unexpected LODF shape {lodf.shape}")
    # Invalid/islanding columns are rejected by the dataset generator.  Keep
    # the static tensor finite so a malformed sample cannot poison a GPU job;
    # the dataset audit remains the gate that rejects such indices.
    lodf = np.where(np.isfinite(lodf), lodf, 0.0)
    return lodf.astype(np.float32)


def to_torch_static(static: Dict[str, Any], cfg: TrainCfg) -> Dict[str, torch.Tensor]:
    dev = torch.device(cfg.device)

    def t(x, dtype=cfg.dtype):
        return torch.as_tensor(x, dtype=dtype, device=dev)

    bus_of_gen = torch.as_tensor(static["bus_of_gen"], dtype=torch.long, device=dev)
    PTDF = torch.as_tensor(static["ptdf_dense"], dtype=cfg.dtype, device=dev)

    return dict(
        nbus=int(static["nbus"]),
        ngen=int(static["ngen"]),
        nbranch=int(static["nbranch"]),
        bus_of_gen=bus_of_gen,
        PTDF=PTDF,
        LODF=t(static.get("lodf", np.zeros((int(static["nbranch"]), int(static["nbranch"])), dtype=np.float32))),
        gs_bus=t(static["gs_bus"]),
        flow_offset=t(static["flow_offset"]),
        fmax=t(static["fmax"]),
        pmin=t(static["pmin"]),
        pmax=t(static["pmax"]),
        c2=t(static["c2"]),
        c1=t(static["c1"]),
    )


# ============================================================
# Anderson acceleration for vectors (bsz, dim)
# ============================================================
def anderson_vec(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-6, beta=1.0):
    if m < 2:
        raise ValueError("Anderson memory m must be >= 2")
    bsz, dim = x0.shape

    X = torch.zeros(bsz, m, dim, dtype=x0.dtype, device=x0.device)
    Fv = torch.zeros(bsz, m, dim, dtype=x0.dtype, device=x0.device)

    X[:, 0] = x0
    Fv[:, 0] = f(x0)
    X[:, 1] = Fv[:, 0]
    Fv[:, 1] = f(X[:, 1])

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = 1.0
    H[:, 1:, 0] = 1.0
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0, 0] = 1.0

    res = []
    fp64_fallback_rows = 0
    k_used = 1
    for k in range(2, max_iter):
        n = min(k, m)
        Gm = Fv[:, :n] - X[:, :n]
        GTG = torch.bmm(Gm, Gm.transpose(1, 2))
        eye = torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        H[:, 1:n + 1, 1:n + 1] = GTG + lam * eye

        Hs = H[:, :n + 1, :n + 1]
        ys = y[:, :n + 1, :]
        # The regularized KKT system is nonsingular for finite residuals, but
        # forming G G^T in fp32 can still lose the ridge numerically on large
        # instances.  Keep the original fp32 path for healthy rows and rebuild
        # only failed rows in fp64 from Gm (rather than casting the damaged H).
        sol, info = torch.linalg.solve_ex(Hs, ys, check_errors=False)
        bad = ((info != 0)
               | ~torch.isfinite(Hs).all(dim=(-2, -1))
               | ~torch.isfinite(sol).all(dim=(-2, -1)))
        if bool(bad.any()):
            Gbad = Gm[bad].to(torch.float64)
            nbad = Gbad.shape[0]
            Hbad = torch.zeros(nbad, n + 1, n + 1, dtype=torch.float64, device=x0.device)
            Hbad[:, 0, 1:] = 1.0
            Hbad[:, 1:, 0] = 1.0
            eye64 = torch.eye(n, dtype=torch.float64, device=x0.device)[None]
            Hbad[:, 1:, 1:] = torch.bmm(Gbad, Gbad.transpose(1, 2)) + float(lam) * eye64
            ybad = torch.zeros(nbad, n + 1, 1, dtype=torch.float64, device=x0.device)
            ybad[:, 0, 0] = 1.0
            solbad, infobad = torch.linalg.solve_ex(Hbad, ybad, check_errors=False)
            if bool((infobad != 0).any()) or not bool(torch.isfinite(solbad).all()):
                raise FloatingPointError(
                    f"Anderson KKT solve remained invalid in fp64 for {int(bad.sum().item())} batch row(s)."
                )
            sol = sol.clone()
            sol[bad] = solbad.to(dtype=sol.dtype)
            fp64_fallback_rows += int(bad.sum().item())
        alpha = sol[:, 1:n + 1, 0]  # (bsz, n)

        Xa = (alpha[:, None, :] @ X[:, :n]).squeeze(1)
        Fa = (alpha[:, None, :] @ Fv[:, :n]).squeeze(1)

        X[:, k % m] = beta * Fa + (1.0 - beta) * Xa
        Fv[:, k % m] = f(X[:, k % m])

        rel = (Fv[:, k % m] - X[:, k % m]).norm(dim=1) / (1e-12 + Fv[:, k % m].norm(dim=1))
        res.append(float(rel.mean().item()))
        k_used = k % m
        if res[-1] < tol:
            break

    return X[:, k_used], res, fp64_fallback_rows


# ============================================================
# Learnable step sizes (rho, s) with s clamped via tanh
# ============================================================
class LearnableSteps(nn.Module):
    """
    Parameters: rho_logit, s_raw

      rho = 0.99 * sigmoid(rho_logit) in (0, 0.99)
      s = s_max * tanh(s_raw) in [-s_max, s_max]

    tau = sqrt(rho)/||K|| * exp(s)
    sigma = sqrt(rho)/||K|| * exp(-s)
    """
    def __init__(self, nK: float, s_max: float, rho_init: float = 0.9, s_init: float = 0.0):
        super().__init__()
        self.nK = float(nK)
        self.s_max = float(s_max)

        p = float(rho_init / 0.99)
        p = min(max(p, 1e-4), 1 - 1e-4)
        rho_logit = np.log(p / (1 - p))
        self.rho_logit = nn.Parameter(torch.tensor(rho_logit, dtype=torch.float32))

        s_init = float(np.clip(s_init, -self.s_max + 1e-6, self.s_max - 1e-6))
        s_raw_init = np.arctanh(s_init / self.s_max) if self.s_max > 0 else 0.0
        self.s_raw = nn.Parameter(torch.tensor(s_raw_init, dtype=torch.float32))

    def rho(self) -> torch.Tensor:
        return 0.99 * torch.sigmoid(self.rho_logit)

    def s(self) -> torch.Tensor:
        return self.s_max * torch.tanh(self.s_raw)

    def tau_sigma(self, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        rho = self.rho().to(dtype=dtype, device=device)
        s = self.s().to(dtype=dtype, device=device)
        scale = torch.sqrt(rho) / (torch.tensor(self.nK, dtype=dtype, device=device) + 1e-12)
        tau = scale * torch.exp(s)
        sigma = scale * torch.exp(-s)
        return tau, sigma


class LearnableStepsDiag(LearnableSteps):
    """Generator-wise primal step scaling with scalar dual step."""

    def __init__(self, nK: float, s_max: float, ngen: int, rho_init: float = 0.9, s_init: float = 0.0):
        super().__init__(nK=nK, s_max=s_max, rho_init=rho_init, s_init=s_init)
        self.diag_raw = nn.Parameter(torch.zeros(int(ngen), dtype=torch.float32))

    def tau_sigma(self, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        tau, sigma = super().tau_sigma(dtype=dtype, device=device)
        # Keep the average primal step close to the scalar baseline at init.
        diag = torch.exp(torch.clamp(self.diag_raw.to(dtype=dtype, device=device), -2.0, 2.0))
        diag = diag / (diag.mean() + 1e-12)
        return tau * diag, sigma


# ============================================================
# PDHG operator with demand input d_bus (operator form)
# State z = [p (ngen), lam (1), nu (2m)]
# ============================================================
class PDHGOperatorLearnable(nn.Module):
    def __init__(self, static: Dict[str, torch.Tensor], steps: LearnableSteps):
        super().__init__()
        self.steps = steps
        for k, v in static.items():
            if isinstance(v, torch.Tensor):
                self.register_buffer(k, v)

        self.nbus = int(static["nbus"])
        self.ngen = int(static["ngen"])
        self.nbranch = int(static["nbranch"])
        if not hasattr(self, "LODF"):
            # Unit tests and legacy callers may construct a static dictionary
            # without scenario data; zero LODF preserves base-case behavior.
            self.register_buffer("LODF", torch.zeros(self.nbranch, self.nbranch, dtype=self.PTDF.dtype, device=self.PTDF.device))

    def split(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = z[:, :self.ngen]
        lam = z[:, self.ngen:self.ngen + 1]
        nu = z[:, self.ngen + 1:]
        return p, lam, nu

    def join(self, p: torch.Tensor, lam: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        return torch.cat([p, lam, nu], dim=1)

    def _bounds(self, bsz: int, cont_gen_idx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        pmin = self.pmin[None, :].expand(bsz, -1)
        pmax = self.pmax[None, :].expand(bsz, -1)
        if cont_gen_idx is None:
            return pmin, pmax
        idx = cont_gen_idx.to(device=pmax.device, dtype=torch.long)
        valid = idx >= 0
        if not torch.any(valid):
            return pmin, pmax
        pmin = pmin.clone()
        pmax = pmax.clone()
        rows = torch.arange(bsz, device=pmax.device)[valid]
        cols = idx[valid].clamp(min=0, max=self.ngen - 1)
        pmin[rows, cols] = 0.0
        pmax[rows, cols] = 0.0
        return pmin, pmax

    def prox_f(self, v: torch.Tensor, tau: torch.Tensor, cont_gen_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        denom = 1.0 + 2.0 * tau * self.c2[None, :]
        x = (v - tau * self.c1[None, :]) / denom
        pmin, pmax = self._bounds(v.shape[0], cont_gen_idx)
        return torch.minimum(torch.maximum(x, pmin), pmax)

    def gen_to_bus_inj(self, p: torch.Tensor) -> torch.Tensor:
        bsz = p.shape[0]
        out = torch.zeros(bsz, self.nbus, dtype=p.dtype, device=p.device)
        idx = self.bus_of_gen[None, :].expand(bsz, -1)
        out.scatter_add_(1, idx, p)
        return out

    def ptdf_apply(self, x_bus: torch.Tensor) -> torch.Tensor:
        return x_bus @ self.PTDF.t()

    def ptdf_t_apply(self, y_br: torch.Tensor) -> torch.Tensor:
        return y_br @ self.PTDF

    def _line_parts(self, cont_line_idx: Optional[torch.Tensor], bsz: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (valid, selected LODF column, selected base PTDF row)."""
        dev = self.PTDF.device
        if cont_line_idx is None:
            idx = torch.zeros(bsz, dtype=torch.long, device=dev)
            valid = torch.zeros(bsz, dtype=torch.bool, device=dev)
        else:
            idx = cont_line_idx.to(device=dev, dtype=torch.long).reshape(-1)
            if idx.numel() != bsz:
                raise ValueError(f"cont_line_idx has {idx.numel()} entries for batch size {bsz}")
            valid = idx >= 0
            idx = idx.clamp(min=0, max=self.nbranch - 1)
        # LODF uses the standard [monitored_branch, outaged_branch] layout.
        # Advanced indexing on the first axis would select rows; gather the
        # requested outage columns and transpose to (batch, nbranch).
        lodf = self.LODF[:, idx].t()
        ptdf_row = self.PTDF[idx]
        if torch.any(~valid):
            lodf = lodf.clone()
            ptdf_row = ptdf_row.clone()
            lodf[~valid] = 0.0
            ptdf_row[~valid] = 0.0
        return valid, lodf, ptdf_row

    def scenario_ptdf_apply(self, x_bus: torch.Tensor, cont_line_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply H_k = H + LODF[:,k] H[k,:] without a Bxmxn tensor."""
        base = self.ptdf_apply(x_bus)
        if cont_line_idx is None:
            return base
        valid, lodf, ptdf_row = self._line_parts(cont_line_idx, x_bus.shape[0])
        row = (x_bus * ptdf_row).sum(dim=1)
        return base + lodf * row[:, None]

    def scenario_ptdf_t_apply(self, y_br: torch.Tensor, cont_line_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Transpose of the compact rank-one post-outage PTDF update."""
        base = self.ptdf_t_apply(y_br)
        if cont_line_idx is None:
            return base
        valid, lodf, ptdf_row = self._line_parts(cont_line_idx, y_br.shape[0])
        corr = (y_br * lodf).sum(dim=1)
        return base + ptdf_row * corr[:, None]

    def scenario_mask(self, cont_line_idx: Optional[torch.Tensor], bsz: int) -> torch.Tensor:
        mask = torch.ones(bsz, self.nbranch, dtype=self.PTDF.dtype, device=self.PTDF.device)
        if cont_line_idx is not None:
            idx = cont_line_idx.to(device=mask.device, dtype=torch.long).reshape(-1)
            valid = idx >= 0
            if torch.any(valid):
                rows = torch.arange(bsz, device=mask.device)[valid]
                cols = idx[valid].clamp(min=0, max=self.nbranch - 1)
                mask[rows, cols] = 0.0
        return mask

    def scenario_flow_offset(self, cont_line_idx: Optional[torch.Tensor], bsz: int) -> torch.Tensor:
        base = self.flow_offset[None, :].expand(bsz, -1)
        if cont_line_idx is None:
            return base
        valid, lodf, ptdf_row = self._line_parts(cont_line_idx, bsz)
        selected = self.flow_offset[None, :].expand(bsz, -1)
        k_offset = self.flow_offset.gather(0, cont_line_idx.to(device=base.device, dtype=torch.long).clamp(min=0, max=self.nbranch - 1))
        out = selected + lodf * k_offset[:, None]
        out[~valid] = base[~valid]
        return out

    def nu_to_gen_grad(self, nu: torch.Tensor, cont_line_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        m = self.nbranch
        w = nu[:, :m] - nu[:, m:]
        bus_grad = self.scenario_ptdf_t_apply(w, cont_line_idx)
        return bus_grad[:, self.bus_of_gen]

    def gen_cost(self, p: torch.Tensor) -> torch.Tensor:
        return (self.c2[None, :] * (p ** 2) + self.c1[None, :] * p).sum(dim=1)

    def effective_demand(self, d_bus: torch.Tensor) -> torch.Tensor:
        """Conventional load PD plus fixed active shunt demand GS."""
        return d_bus + self.gs_bus[None, :]

    def branch_flows(self, p: torch.Tensor, d_bus: torch.Tensor, cont_line_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Complete DC branch flows, including the phase-shifter affine term."""
        inj = self.gen_to_bus_inj(p) - self.effective_demand(d_bus)
        base = self.scenario_ptdf_apply(inj, cont_line_idx)
        return base + self.scenario_flow_offset(cont_line_idx, p.shape[0])

    def flow_constraint_rhs(self, d_bus: torch.Tensor, cont_line_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Right-hand side g for [H_G; -H_G] p <= g."""
        ptdf_d = self.scenario_ptdf_apply(self.effective_demand(d_bus), cont_line_idx)
        offset = self.scenario_flow_offset(cont_line_idx, d_bus.shape[0])
        g_pos = self.fmax[None, :] + ptdf_d - offset
        g_neg = self.fmax[None, :] - ptdf_d + offset
        return torch.cat([g_pos, g_neg], dim=1)

    def flow_violation_vec(self, p: torch.Tensor, d_bus: torch.Tensor, cont_line_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = self.branch_flows(p, d_bus, cont_line_idx)
        viol = torch.clamp(f.abs() - self.fmax[None, :], min=0.0)
        return viol * self.scenario_mask(cont_line_idx, p.shape[0])

    @torch.no_grad()
    def diagnostics(self, p: torch.Tensor, d_bus: torch.Tensor, cont_line_idx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b = self.effective_demand(d_bus).sum(dim=1)
        bal = p.sum(dim=1) - b
        viol = self.flow_violation_vec(p, d_bus, cont_line_idx)
        max_viol = viol.max(dim=1).values
        return bal, max_viol

    def forward(self, z: torch.Tensor, d_bus: torch.Tensor, cont_gen_idx: Optional[torch.Tensor] = None, cont_line_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        p, lam, nu = self.split(z)
        tau, sigma = self.steps.tau_sigma(dtype=z.dtype, device=z.device)

        # Demand-induced component (conventional load plus fixed active shunts)
        d_eff = self.effective_demand(d_bus)
        g = self.flow_constraint_rhs(d_bus, cont_line_idx)

        # Generator-induced branch flows
        inj_gen = self.gen_to_bus_inj(p)
        f_gen = self.scenario_ptdf_apply(inj_gen, cont_line_idx)
        flow_term = torch.cat([f_gen, -f_gen], dim=1)

        # Dual updates
        b = d_eff.sum(dim=1, keepdim=True)
        lam_next = lam + sigma * (p.sum(dim=1, keepdim=True) - b)
        nu_next = F.relu(nu + sigma * (flow_term - g))
        nu_next = nu_next * torch.cat([
            self.scenario_mask(cont_line_idx, z.shape[0]),
            self.scenario_mask(cont_line_idx, z.shape[0]),
        ], dim=1)

        # Primal update
        grad = lam_next.expand(-1, self.ngen) + self.nu_to_gen_grad(nu_next, cont_line_idx)
        v = p - tau * grad
        p_next = self.prox_f(v, tau, cont_gen_idx)

        return self.join(p_next, lam_next, nu_next)


# ============================================================
# ||K|| estimation (power iteration), operator form
# ============================================================
@torch.no_grad()
def estimate_norm_K(op: PDHGOperatorLearnable, iters: int = 100) -> float:
    ngen = int(op.ngen)
    x = torch.randn(1, ngen, device=op.c2.device, dtype=op.c2.dtype)
    x = x / (x.norm() + 1e-12)
    for _ in range(iters):
        inj_gen = op.gen_to_bus_inj(x)
        f_gen = op.ptdf_apply(inj_gen)
        a = x.sum(dim=1, keepdim=True)
        x_new = a.expand(-1, ngen) + op.nu_to_gen_grad(torch.cat([f_gen, -f_gen], dim=1))
        x_new = x_new / (x_new.norm() + 1e-12)
        x = x_new
    inj_gen = op.gen_to_bus_inj(x)
    f_gen = op.ptdf_apply(inj_gen)
    y_norm = torch.sqrt((x.sum(dim=1) ** 2) + 2.0 * (f_gen ** 2).sum(dim=1))
    return float((y_norm / (x.norm(dim=1) + 1e-12)).mean().item())


# ============================================================
# DEQ wrapper (forward fixed-point + implicit backward hook)
# ============================================================
class DEQFixedPoint(nn.Module):
    def __init__(
        self,
        f: nn.Module,
        dim: int,
        fwd_kwargs: Dict[str, Any],
        bwd_kwargs: Dict[str, Any],
        picard_iters: int = 0,
    ):
        super().__init__()
        self.f = f
        self.dim = int(dim)
        self.fwd_kwargs = dict(fwd_kwargs)
        self.bwd_kwargs = dict(bwd_kwargs)
        self.picard_iters = int(picard_iters)
        self.forward_res: List[float] = []
        self.backward_res: List[float] = []
        self.forward_fp64_fallback_rows = 0
        self.backward_fp64_fallback_rows = 0

    def forward(self, d_bus: torch.Tensor, cont_gen_idx: Optional[torch.Tensor] = None, cont_line_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz = d_bus.shape[0]
        z0 = torch.zeros(bsz, self.dim, device=d_bus.device, dtype=d_bus.dtype)
        self.forward_fp64_fallback_rows = 0
        self.backward_fp64_fallback_rows = 0

        def T(z):
            return self.f(z, d_bus, cont_gen_idx, cont_line_idx)

        with torch.no_grad():
            z = z0
            for _ in range(self.picard_iters):
                z = T(z)
            z_star, self.forward_res, self.forward_fp64_fallback_rows = anderson_vec(T, z, **self.fwd_kwargs)

        z = T(z_star)

        if self.training:
            z0 = z.clone().detach().requires_grad_()
            f0 = self.f(z0, d_bus, cont_gen_idx, cont_line_idx)

            def backward_hook(grad):
                def lin_map(y):
                    Jt_y = autograd.grad(f0, z0, y, retain_graph=True)[0]
                    return Jt_y + grad

                with torch.no_grad():
                    g, self.backward_res, self.backward_fp64_fallback_rows = anderson_vec(lin_map, grad, **self.bwd_kwargs)
                return g

            z.register_hook(backward_hook)

        return z


# ============================================================
# Dataset (pd_bus required; pg optional)
# ============================================================
class DCOPFDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: str):
        z = np.load(npz_path)
        if "pd_bus" not in z.files:
            raise KeyError("Dataset npz must contain key 'pd_bus'.")
        self.pd_bus = z["pd_bus"].astype(np.float32)
        self.pg = z["pg"].astype(np.float32) if "pg" in z.files else None
        self.cont_gen_idx = z["cont_gen_idx"].astype(np.int64) if "cont_gen_idx" in z.files else None
        self.cont_line_idx = z["cont_line_idx"].astype(np.int64) if "cont_line_idx" in z.files else None
        self.lodf = z["lodf"].astype(np.float32) if "lodf" in z.files else None
        self.gs_bus = z["gs_bus"].astype(np.float32) if "gs_bus" in z.files else None
        self.has_gs_contract = "gs_bus" in z.files and "pd_bus_includes_gs" in z.files
        self.pd_bus_includes_gs = bool(
            int(np.asarray(z["pd_bus_includes_gs"]).item())
        ) if "pd_bus_includes_gs" in z.files else False
        if self.pd_bus_includes_gs:
            raise ValueError(
                "This trainer expects pd_bus to contain conventional PD only; "
                "pd_bus_includes_gs=1 would double-count GS."
            )
        if self.gs_bus is not None:
            if self.gs_bus.shape != (self.pd_bus.shape[1],):
                raise ValueError(
                    f"gs_bus must have shape ({self.pd_bus.shape[1]},), got {self.gs_bus.shape}."
                )
            if not np.all(np.isfinite(self.gs_bus)):
                raise ValueError("gs_bus contains non-finite values.")
        # Optional precomputed split: 0=train, 1=val, 2=test (used instead of random_split)
        self.split = z["split"].astype(np.int64) if "split" in z.files else None

    def __len__(self):
        return int(self.pd_bus.shape[0])

    def __getitem__(self, i: int):
        out = {"pd_bus": torch.from_numpy(self.pd_bus[i])}
        if self.pg is not None:
            out["pg"] = torch.from_numpy(self.pg[i])
        if self.cont_gen_idx is not None:
            out["cont_gen_idx"] = torch.tensor(self.cont_gen_idx[i], dtype=torch.long)
        if self.cont_line_idx is not None:
            out["cont_line_idx"] = torch.tensor(self.cont_line_idx[i], dtype=torch.long)
        return out


# ============================================================
# Build base net (static matrices must match dataset generation)
# ============================================================
def build_base_net():
    net = load_ieee_case(CASE_NAME)
    scale_loads(net, LOAD_SCALE)
    ensure_controllable(net)
    set_voltage_limits(net, *VM_BOUNDS)
    scale_line_and_trafo_limits(net, LINE_LIMIT_SCALE)
    add_heterogeneous_quadratic_costs_and_bounds(net, GEN_MARGIN_MW, seed=SEED)
    return net


# ============================================================
# Self-supervised loss (Option A)
# ============================================================
def optionA_loss(op: PDHGOperatorLearnable, p: torch.Tensor, d_bus: torch.Tensor, cfg: TrainCfg, cont_line_idx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    cost = op.gen_cost(p)
    demand_sum = op.effective_demand(d_bus).sum(dim=1).clamp(min=cfg.eps)
    j_cost = (cost / demand_sum).mean()

    bal = p.sum(dim=1) - demand_sum
    j_bal = (bal ** 2).mean()

    viol = op.flow_violation_vec(p, d_bus, cont_line_idx)
    j_flow = (viol ** 2).mean()
    j_flow_l1 = viol.mean()          # linear (exact) penalty: gradient does not vanish near feasibility

    total = (cfg.w_cost * j_cost + cfg.w_bal * j_bal
             + cfg.w_flow * j_flow + cfg.w_flow_l1 * j_flow_l1)
    meters = {
        "L_total": float(total.detach().cpu().item()),
        "J_cost": float(j_cost.detach().cpu().item()),
        "J_bal": float(j_bal.detach().cpu().item()),
        "J_flow": float(j_flow.detach().cpu().item()),
        "J_flow_l1": float(j_flow_l1.detach().cpu().item()),
    }
    return total, meters


# ============================================================
# Plotting (publishable curves)
# ============================================================
def save_publishable_plots(history: list[dict], fig_dir: Path, stem: str, *, fmt: str = "pdf", dpi: int = 300) -> None:
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np
    except Exception as e:
        (fig_dir / f"{stem}_plots_skipped.txt").write_text(f"Plotting skipped: {e}\n", encoding="utf-8")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)

    def _paper_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for s in ax.spines.values():
            s.set_linewidth(1.2)
        ax.tick_params(width=1.2, length=3)

    epochs = [r.get("epoch") for r in history]

    def _get(split: str, key: str):
        return [((r.get(split, {}) or {}).get(key, float("nan"))) for r in history]

    def _plot_train_val(key: str, ylabel: str, fname: str, *, logy: bool = False):
        ytr = _get("train", key)
        yva = _get("val", key)
        fig = plt.figure(figsize=(3.6, 2.4))
        ax = fig.add_subplot(111)
        _paper_axes(ax)
        if logy:
            ax.set_yscale("log")
        ax.plot(epochs, ytr, linewidth=1.6)
        ax.plot(epochs, yva, linewidth=1.6, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        fig.tight_layout(pad=0.2)
        fig.savefig(fig_dir / f"{stem}_{fname}.{fmt}", dpi=dpi)
        plt.close(fig)

    def _plot_single(key: str, ylabel: str, fname: str, *, logy: bool = False):
        ys = [r.get(key, float("nan")) for r in history]
        fig = plt.figure(figsize=(3.6, 2.4))
        ax = fig.add_subplot(111)
        _paper_axes(ax)
        if logy:
            ax.set_yscale("log")
        ax.plot(epochs, ys, linewidth=1.6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        fig.tight_layout(pad=0.2)
        fig.savefig(fig_dir / f"{stem}_{fname}.{fmt}", dpi=dpi)
        plt.close(fig)

    _plot_train_val("L_total", "Loss", "loss_total", logy=True)
    _plot_train_val("J_cost", "Cost term", "loss_cost", logy=True)
    _plot_train_val("J_bal", "Balance penalty", "loss_balance", logy=True)
    _plot_train_val("J_flow", "Flow penalty", "loss_flow", logy=True)
    _plot_train_val("bal_abs_MW", r"$|{\rm balance}|$ (MW)", "balance_abs", logy=True)
    _plot_train_val("viol_max_MW", r"max flow viol. (MW)", "flow_viol_max", logy=True)
    _plot_train_val("alg_res", "Fixed-point residual", "alg_res", logy=True)
    _plot_train_val("fwd_res", "Anderson residual", "fwd_res", logy=True)

    _plot_single("rho", r"$\rho$", "rho")
    _plot_single("s", r"$s$", "s")
    _plot_single("tau", r"$\tau$", "tau", logy=True)
    _plot_single("sigma", r"$\sigma$", "sigma", logy=True)


# ============================================================
# Eval vs GT (optional; requires labels)
# ============================================================
@torch.no_grad()
def eval_vs_gt_csv(op: PDHGOperatorLearnable, deq: DEQFixedPoint, static: Dict[str, torch.Tensor],
                   loader: torch.utils.data.DataLoader, cfg: TrainCfg, out_dir: Path, stem: str) -> None:
    dev = torch.device(cfg.device)
    ngen = int(static["ngen"])
    rows: List[Dict[str, float]] = []
    for batch in loader:
        cont_gen_idx = None
        cont_line_idx = None
        if isinstance(batch, dict):
            d_bus = batch["pd_bus"]
            p_t = batch.get("pg")
            cont_gen_idx = batch.get("cont_gen_idx")
            cont_line_idx = batch.get("cont_line_idx")
            if p_t is None:
                continue
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            d_bus, p_t = batch
        else:
            continue
        d_bus = d_bus.to(dev, dtype=cfg.dtype)
        p_t = p_t.to(dev, dtype=cfg.dtype)
        if cont_gen_idx is not None:
            cont_gen_idx = cont_gen_idx.to(dev, dtype=torch.long)
        if cont_line_idx is not None:
            cont_line_idx = cont_line_idx.to(dev, dtype=torch.long)
        z = deq(d_bus, cont_gen_idx, cont_line_idx)
        p = z[:, :ngen]

        rmse = torch.sqrt(((p - p_t) ** 2).mean(dim=1))
        cp = op.gen_cost(p)
        ct = op.gen_cost(p_t)
        rel_gap_abs = (cp - ct).abs() / (ct.abs() + cfg.eps)

        bal_p, viol_p = op.diagnostics(p, d_bus, cont_line_idx)
        bal_t, viol_t = op.diagnostics(p_t, d_bus, cont_line_idx)
        viol_matrix_p = op.flow_violation_vec(p, d_bus, cont_line_idx)
        viol_matrix_t = op.flow_violation_vec(p_t, d_bus, cont_line_idx)
        viol_branch_p = viol_matrix_p.mean(dim=1)
        viol_branch_t = viol_matrix_t.mean(dim=1)
        j_flow_p = (viol_matrix_p ** 2).mean(dim=1)

        for i in range(p.shape[0]):
            gen_idx = int(cont_gen_idx[i].cpu().item()) if cont_gen_idx is not None else -1
            line_idx = int(cont_line_idx[i].cpu().item()) if cont_line_idx is not None else -1
            if gen_idx >= 0 and line_idx >= 0:
                raise ValueError("eval_vs_gt encountered a simultaneous line+generator outage")
            contingency_type = "generator" if gen_idx >= 0 else ("line" if line_idx >= 0 else "clean")
            rows.append({
                "subset_row": int(len(rows)),
                "contingency_type": contingency_type,
                "cont_gen_idx": gen_idx,
                "cont_line_idx": line_idx,
                "rmse_MW": float(rmse[i].cpu().item()),
                "cost_pred": float(cp[i].cpu().item()),
                "cost_true": float(ct[i].cpu().item()),
                "abs_rel_cost_gap": float(rel_gap_abs[i].cpu().item()),
                "bal_abs_pred": float(bal_p[i].abs().cpu().item()),
                "viol_max_pred": float(viol_p[i].cpu().item()),
                "viol_branch_mean_pred": float(viol_branch_p[i].cpu().item()),
                "J_flow_pred": float(j_flow_p[i].cpu().item()),
                "flow_sat_1e-3_pred": float((viol_p[i] <= cfg.sat_tol_MW).cpu().item()),
                "bal_abs_true": float(bal_t[i].abs().cpu().item()),
                "viol_max_true": float(viol_t[i].cpu().item()),
                "viol_branch_mean_true": float(viol_branch_t[i].cpu().item()),
            })

    if not rows:
        (out_dir / f"{stem}_eval_vs_gt_skipped.txt").write_text(
            "Dataset split has no pg labels; skipping eval_vs_gt.\n", encoding="utf-8"
        )
        return

    csv_path = out_dir / f"{stem}_eval_vs_gt.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # quick summary
    arr_rmse = np.array([r["rmse_MW"] for r in rows], dtype=float)
    arr_gap = np.array([r["abs_rel_cost_gap"] for r in rows], dtype=float)
    arr_viol = np.array([r["viol_max_pred"] for r in rows], dtype=float)
    arr_bal = np.array([r["bal_abs_pred"] for r in rows], dtype=float)

    summ = {
        "N": int(len(rows)),
        "rmse_mean": float(np.mean(arr_rmse)),
        "rmse_median": float(np.median(arr_rmse)),
        "abs_rel_cost_gap_mean": float(np.mean(arr_gap)),
        "abs_rel_cost_gap_median": float(np.median(arr_gap)),
        "viol_max_pred_mean": float(np.mean(arr_viol)),
        "viol_max_pred_p95": float(np.quantile(arr_viol, 0.95)),
        "bal_abs_pred_mean": float(np.mean(arr_bal)),
        "bal_abs_pred_p95": float(np.quantile(arr_bal, 0.95)),
    }
    category_summary = {}
    for category in ("clean", "line", "generator"):
        selected = [r for r in rows if r["contingency_type"] == category]
        if not selected:
            continue
        category_summary[category] = {
            "N": int(len(selected)),
            "rmse_mean": float(np.mean([r["rmse_MW"] for r in selected])),
            "abs_rel_cost_gap_mean": float(np.mean([r["abs_rel_cost_gap"] for r in selected])),
            "viol_max_pred_mean": float(np.mean([r["viol_max_pred"] for r in selected])),
            "viol_max_pred_p95": float(np.quantile([r["viol_max_pred"] for r in selected], 0.95)),
            "flow_sat_rate_1e-3": float(np.mean([r["flow_sat_1e-3_pred"] for r in selected])),
            "bal_abs_pred_mean": float(np.mean([r["bal_abs_pred"] for r in selected])),
        }
    summ["category_summary"] = category_summary
    (out_dir / f"{stem}_eval_vs_gt_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", type=str, default="", help="path to dataset npz (default: io/dc_opf_dataset_<case>.npz)")
    ap.add_argument("--epochs", type=int, default=TrainCfg.epochs)
    ap.add_argument("--batch_size", type=int, default=TrainCfg.batch_size)
    ap.add_argument("--lr", type=float, default=TrainCfg.lr)
    ap.add_argument("--weight_decay", type=float, default=TrainCfg.weight_decay)
    ap.add_argument("--train_frac", type=float, default=TrainCfg.train_frac)
    ap.add_argument("--val_frac", type=float, default=TrainCfg.val_frac)
    ap.add_argument("--seed_split", type=int, default=TrainCfg.seed_split)
    ap.add_argument("--device", type=str, default=TrainCfg.device)
    ap.add_argument("--step_mode", type=str, default="scalar", choices=["scalar", "diag"])

    # solve / acceleration controls
    ap.add_argument("--picard_iters", type=int, default=TrainCfg.picard_iters)
    ap.add_argument("--anderson_m", type=int, default=TrainCfg.anderson_m)
    ap.add_argument("--anderson_lam", type=float, default=TrainCfg.anderson_lam)
    ap.add_argument("--anderson_max_iter", type=int, default=TrainCfg.anderson_max_iter)
    ap.add_argument("--anderson_tol", type=float, default=TrainCfg.anderson_tol)
    ap.add_argument("--norm_iters", type=int, default=100)

    ap.add_argument("--b_anderson_m", type=int, default=TrainCfg.b_anderson_m)
    ap.add_argument("--b_anderson_lam", type=float, default=TrainCfg.b_anderson_lam)
    ap.add_argument("--b_anderson_max_iter", type=int, default=TrainCfg.b_anderson_max_iter)
    ap.add_argument("--b_anderson_tol", type=float, default=TrainCfg.b_anderson_tol)

    # selfsup weights
    ap.add_argument("--w_cost", type=float, default=TrainCfg.w_cost)
    ap.add_argument("--w_bal", type=float, default=TrainCfg.w_bal)
    ap.add_argument("--w_flow", type=float, default=TrainCfg.w_flow)
    ap.add_argument("--w_flow_l1", type=float, default=TrainCfg.w_flow_l1)

    # eval / save
    ap.add_argument("--sat_tol_MW", "--sat_tol_mw", dest="sat_tol_MW", type=float, default=TrainCfg.sat_tol_MW)
    ap.add_argument("--out_dir", type=str, default=TrainCfg.out_dir)
    ap.add_argument("--ckpt_name", type=str, default=TrainCfg.ckpt_name)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="resume from the per-epoch training-state checkpoint when present",
    )
    ap.add_argument(
        "--require_gs_metadata",
        action="store_true",
        help="fail if the dataset lacks the explicit gs_bus/PD-only contract",
    )

    ap.add_argument("--no_eval_test_each_epoch", action="store_true", help="skip test eval each epoch")
    ap.add_argument(
        "--skip_final_test",
        action="store_true",
        help="smoke phase only: save the resumable epoch state without the final test/export pass",
    )
    ap.add_argument("--save_plots", action="store_true", help="save publishable curves to out_dir/figs")
    ap.add_argument("--plot_format", type=str, default=TrainCfg.plot_format, choices=["pdf", "png", "svg"])
    ap.add_argument("--plot_dpi", type=int, default=TrainCfg.plot_dpi)

    ap.add_argument("--no_csv", action="store_true", help="disable CSV history export (JSON still written)")
    ap.add_argument("--no_history_every_epoch", action="store_true", help="only write history on best checkpoint")

    ap.add_argument("--eval_vs_gt", action="store_true", help="after training: evaluate vs pg labels (if present) and export CSV")
    ap.add_argument(
        "--eval_split",
        type=str,
        default=TrainCfg.eval_split,
        choices=["train", "val", "test", "all"],
    )

    args = ap.parse_args()

    cfg = TrainCfg(
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed_split=args.seed_split,
        picard_iters=args.picard_iters,
        anderson_m=args.anderson_m,
        anderson_lam=args.anderson_lam,
        anderson_max_iter=args.anderson_max_iter,
        anderson_tol=args.anderson_tol,
        b_anderson_m=args.b_anderson_m,
        b_anderson_lam=args.b_anderson_lam,
        b_anderson_max_iter=args.b_anderson_max_iter,
        b_anderson_tol=args.b_anderson_tol,
        w_cost=args.w_cost,
        w_bal=args.w_bal,
        w_flow=args.w_flow,
        w_flow_l1=args.w_flow_l1,
        sat_tol_MW=args.sat_tol_MW,
        out_dir=args.out_dir,
        ckpt_name=args.ckpt_name,
        eval_test_each_epoch=(not args.no_eval_test_each_epoch),
        save_plots=args.save_plots,
        plot_format=args.plot_format,
        plot_dpi=args.plot_dpi,
        save_csv=(not args.no_csv),
        save_history_every_epoch=(not args.no_history_every_epoch),
        eval_vs_gt=args.eval_vs_gt,
        eval_split=args.eval_split,
    )
    cfg.step_mode = args.step_mode

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figs"
    stem = Path(cfg.ckpt_name).stem

    # dataset
    data_path = Path(args.data) if args.data else Path("io") / f"dc_opf_dataset_{CASE_NAME}.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    ds = DCOPFDataset(str(data_path))
    N = len(ds)

    if getattr(ds, "split", None) is not None:
        # Honor the dataset's precomputed split (e.g. disjoint-contingency variant).
        split = ds.split
        train_idx = np.where(split == 0)[0].tolist()
        val_idx = np.where(split == 1)[0].tolist()
        test_idx = np.where(split == 2)[0].tolist()
        if len(test_idx) == 0:  # safety: ensure non-empty test
            test_idx = val_idx[-1:] or train_idx[-1:]
        train_ds = torch.utils.data.Subset(ds, train_idx)
        val_ds = torch.utils.data.Subset(ds, val_idx)
        test_ds = torch.utils.data.Subset(ds, test_idx)
        n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)
        print(f"[split] using dataset 'split' array: train/val/test = {n_train}/{n_val}/{n_test}")
    else:
        n_train = int(cfg.train_frac * N)
        n_val = int(cfg.val_frac * N)
        n_test = N - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_train = max(1, n_train - 1)

        gen = torch.Generator().manual_seed(cfg.seed_split)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [n_train, n_val, n_test], generator=gen)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # static matrices
    net = build_base_net()
    ppc = build_ppc_from_net(net)
    static_np = extract_static_mats(ppc)
    if ds.lodf is not None:
        expected_lodf_shape = (int(static_np["nbranch"]), int(static_np["nbranch"]))
        if ds.lodf.shape != expected_lodf_shape:
            raise ValueError(
                f"dataset lodf shape {ds.lodf.shape} does not match {expected_lodf_shape}"
            )
        if not np.all(np.isfinite(ds.lodf)):
            raise ValueError("dataset lodf contains non-finite values")
        static_np["lodf"] = ds.lodf
        print("[data] using finite outage-specific LODF columns embedded in dataset")
    else:
        static_np["lodf"] = extract_lodf(ppc, static_np)
    if ds.cont_line_idx is not None:
        if ds.cont_line_idx.shape != (N,):
            raise ValueError(f"cont_line_idx shape {ds.cont_line_idx.shape} does not match N={N}.")
        if np.any((ds.cont_line_idx < -1) | (ds.cont_line_idx >= int(static_np["nbranch"]))):
            raise ValueError("cont_line_idx contains an out-of-range base branch index.")
    physics_signature = {
        "code_version": CODE_VERSION,
        "flow_model_contract": FLOW_MODEL_CONTRACT,
        "flow_offset_sha256": sha256_array(static_np["flow_offset"]),
        "gs_sha256": sha256_array(static_np["gs_bus"]),
        "fmax_sha256": sha256_array(static_np["fmax"]),
        "lodf_sha256": sha256_array(static_np["lodf"]),
        "bus_of_gen_sha256": sha256_array(static_np["bus_of_gen"]),
    }

    # Corrected datasets record the static GS vector for an explicit data/model
    # contract.  Legacy datasets remain supported because GS is recoverable
    # from the network, but a present vector must match exactly.
    if (ds.gs_bus is None or not ds.has_gs_contract) and args.require_gs_metadata:
        raise ValueError(
            "Fair-study run requires explicit gs_bus metadata; prepare the corrected dataset first."
        )
    if ds.gs_bus is None:
        print(
            "[data] dataset has no gs_bus metadata; using GS from the network "
            f"(sum={float(np.sum(static_np['gs_bus'])):.6f} MW)."
        )
    elif not np.allclose(ds.gs_bus, static_np["gs_bus"], rtol=0.0, atol=1e-6):
        max_err = float(np.max(np.abs(ds.gs_bus - static_np["gs_bus"])))
        raise ValueError(
            "Dataset gs_bus does not match the selected network/internal bus ordering; "
            f"max_abs_error={max_err:.6g} MW."
        )
    else:
        print(f"[data] validated gs_bus metadata (sum={float(np.sum(ds.gs_bus)):.6f} MW).")

    dev = torch.device(cfg.device)
    static = to_torch_static(static_np, cfg)

    # learnable steps (need ||K||)
    torch.manual_seed(cfg.seed_split)
    steps_tmp = LearnableSteps(nK=1.0, s_max=cfg.s_max, rho_init=0.9, s_init=0.0).to(dev)
    op_tmp = PDHGOperatorLearnable(static, steps_tmp).to(dev)
    nK = estimate_norm_K(op_tmp, iters=args.norm_iters)

    if args.step_mode == "diag":
        steps = LearnableStepsDiag(nK=nK, s_max=cfg.s_max, ngen=static["ngen"], rho_init=0.9, s_init=0.0).to(dev)
    else:
        steps = LearnableSteps(nK=nK, s_max=cfg.s_max, rho_init=0.9, s_init=0.0).to(dev)
    op = PDHGOperatorLearnable(static, steps).to(dev)

    dim = static["ngen"] + 1 + 2 * static["nbranch"]
    deq = DEQFixedPoint(
        f=op,
        dim=dim,
        picard_iters=cfg.picard_iters,
        fwd_kwargs=dict(m=cfg.anderson_m, lam=cfg.anderson_lam, max_iter=cfg.anderson_max_iter, tol=cfg.anderson_tol, beta=cfg.anderson_beta),
        bwd_kwargs=dict(m=cfg.b_anderson_m, lam=cfg.b_anderson_lam, max_iter=cfg.b_anderson_max_iter, tol=cfg.b_anderson_tol, beta=cfg.b_anderson_beta),
    ).to(dev)

    opt = torch.optim.Adam(steps.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    resume_path = out_dir / f"{stem}_training_state.pt"
    t_start = time.time()
    if str(dev).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dev)

    def run_epoch(loader, train: bool) -> Tuple[dict, bool]:
        if train:
            steps.train(); deq.train()
        else:
            steps.eval(); deq.eval()

        meters = {
            "L_total": 0.0, "J_cost": 0.0, "J_bal": 0.0, "J_flow": 0.0, "J_flow_l1": 0.0,
            "fwd_res": 0.0, "alg_res": 0.0,
            "bal_abs_MW": 0.0, "viol_MW": 0.0,
            "viol_p95_MW": float("nan"),
            "bal_max_MW": 0.0, "viol_max_MW": 0.0,
            "anderson_fwd_fp64_fallback_rows": 0.0,
            "anderson_bwd_fp64_fallback_rows": 0.0,
            "rmse_MW_diag": float("nan"),
            "cost_gap": float("nan"),
            "cost_rel_gap_abs": float("nan"),
            "bal_sat_rate": float("nan"),
            "flow_sat_rate": float("nan"),
        }
        batch_count = 0
        sample_count = 0
        have_labels_epoch = False

        rmse_sq_sum = 0.0; rmse_elem_n = 0
        gap_sum = 0.0; gap_n = 0
        relgap_sum = 0.0; relgap_n = 0
        bal_sat_sum = 0.0; bal_sat_n = 0
        flow_sat_sum = 0.0; flow_sat_n = 0
        viol_epoch_values: list[np.ndarray] = []

        for batch in loader:
            cont_gen_idx = None
            cont_line_idx = None
            if isinstance(batch, dict):
                d_bus = batch["pd_bus"]
                p_t = batch.get("pg")
                if p_t is not None:
                    have_labels_epoch = True
                    p_t = p_t.to(dev, dtype=cfg.dtype)
                cont_gen_idx = batch.get("cont_gen_idx")
                cont_line_idx = batch.get("cont_line_idx")
                if cont_gen_idx is not None:
                    cont_gen_idx = cont_gen_idx.to(dev, dtype=torch.long)
                if cont_line_idx is not None:
                    cont_line_idx = cont_line_idx.to(dev, dtype=torch.long)
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                d_bus, p_t = batch
                have_labels_epoch = True
                p_t = p_t.to(dev, dtype=cfg.dtype)
            else:
                d_bus = batch
                p_t = None

            d_bus = d_bus.to(dev, dtype=cfg.dtype)
            batch_n = int(d_bus.shape[0])

            z = deq(d_bus, cont_gen_idx, cont_line_idx)
            p = z[:, :static["ngen"]]

            loss, parts = optionA_loss(op, p, d_bus, cfg, cont_line_idx)

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            with torch.no_grad():
                fwd_last = float(deq.forward_res[-1]) if getattr(deq, "forward_res", None) and len(deq.forward_res) else float("nan")
                z_next = op(z, d_bus, cont_gen_idx, cont_line_idx)
                alg = (z_next - z).norm(dim=1) / (1e-12 + z_next.norm(dim=1))
                alg = float(alg.mean().item())

                bal, viol = op.diagnostics(p, d_bus, cont_line_idx)
                bal_abs = float(bal.abs().mean().item())
                bal_max = float(bal.abs().max().item())
                viol_m = float(viol.mean().item())
                viol_max = float(viol.max().item())
                viol_epoch_values.append(viol.detach().cpu().numpy())

                bal_sat_sum += float((bal.abs() <= cfg.sat_tol_MW).float().sum().item())
                bal_sat_n += int(bal.numel())
                flow_sat_sum += float((viol <= cfg.sat_tol_MW).float().sum().item())
                flow_sat_n += int(viol.numel())

                if have_labels_epoch and cfg.track_rmse_if_labels_exist and (p_t is not None):
                    rmse_sq_sum += float(((p - p_t) ** 2).sum().item())
                    rmse_elem_n += int(p.numel())

                    cp = op.gen_cost(p)
                    ct = op.gen_cost(p_t)
                    gap_sum += float((cp - ct).sum().item())
                    gap_n += int(cp.numel())
                    relgap_sum += float(
                        ((cp - ct).abs() / (ct.abs() + cfg.eps)).sum().item()
                    )
                    relgap_n += int(cp.numel())

            meters["L_total"] += parts["L_total"] * batch_n
            meters["J_cost"] += parts["J_cost"] * batch_n
            meters["J_bal"] += parts["J_bal"] * batch_n
            meters["J_flow"] += parts["J_flow"] * batch_n
            meters["J_flow_l1"] += parts["J_flow_l1"] * batch_n
            meters["fwd_res"] += (fwd_last if np.isfinite(fwd_last) else 0.0) * batch_n
            meters["alg_res"] += (alg if np.isfinite(alg) else 0.0) * batch_n
            meters["bal_abs_MW"] += (bal_abs if np.isfinite(bal_abs) else 0.0) * batch_n
            meters["viol_MW"] += (viol_m if np.isfinite(viol_m) else 0.0) * batch_n
            meters["anderson_fwd_fp64_fallback_rows"] += float(deq.forward_fp64_fallback_rows)
            meters["anderson_bwd_fp64_fallback_rows"] += float(deq.backward_fp64_fallback_rows)

            if np.isfinite(bal_max):
                meters["bal_max_MW"] = max(meters["bal_max_MW"], bal_max)
            if np.isfinite(viol_max):
                meters["viol_max_MW"] = max(meters["viol_max_MW"], viol_max)

            batch_count += 1
            sample_count += batch_n

        for k in [
            "L_total", "J_cost", "J_bal", "J_flow", "J_flow_l1", "fwd_res",
            "alg_res", "bal_abs_MW", "viol_MW",
        ]:
            meters[k] /= max(1, sample_count)
        for k in [
            "anderson_fwd_fp64_fallback_rows", "anderson_bwd_fp64_fallback_rows",
        ]:
            meters[k] /= max(1, batch_count)

        meters["rmse_MW_diag"] = (
            float(np.sqrt(rmse_sq_sum / rmse_elem_n))
            if rmse_elem_n > 0 else float("nan")
        )
        meters["cost_gap"] = (gap_sum / gap_n) if gap_n > 0 else float("nan")
        meters["cost_rel_gap_abs"] = (relgap_sum / relgap_n) if relgap_n > 0 else float("nan")
        meters["bal_sat_rate"] = (bal_sat_sum / bal_sat_n) if bal_sat_n > 0 else float("nan")
        meters["flow_sat_rate"] = (flow_sat_sum / flow_sat_n) if flow_sat_n > 0 else float("nan")
        if viol_epoch_values:
            meters["viol_p95_MW"] = float(
                np.quantile(np.concatenate(viol_epoch_values, axis=0), 0.95)
            )
        return meters, have_labels_epoch

    best_val = float("inf")
    best_row = None
    best_steps_state = None
    history: list[dict] = []
    start_epoch = 1

    resume_signature = {
        **physics_signature,
        "case_name": CASE_NAME,
        "data_path": str(data_path.resolve()),
        "N": int(N),
        "step_mode": args.step_mode,
        "batch_size": int(cfg.batch_size),
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "seed_split": int(cfg.seed_split),
        "norm_iters": int(args.norm_iters),
        "picard_iters": int(cfg.picard_iters),
        "anderson_m": int(cfg.anderson_m),
        "anderson_lam": float(cfg.anderson_lam),
        "anderson_max_iter": int(cfg.anderson_max_iter),
        "anderson_tol": float(cfg.anderson_tol),
        "b_anderson_m": int(cfg.b_anderson_m),
        "b_anderson_lam": float(cfg.b_anderson_lam),
        "b_anderson_max_iter": int(cfg.b_anderson_max_iter),
        "b_anderson_tol": float(cfg.b_anderson_tol),
        "w_cost": float(cfg.w_cost),
        "w_bal": float(cfg.w_bal),
        "w_flow": float(cfg.w_flow),
        "w_flow_l1": float(cfg.w_flow_l1),
    }

    if args.resume and resume_path.exists():
        try:
            state = torch.load(resume_path, map_location=dev, weights_only=False)
        except TypeError:
            state = torch.load(resume_path, map_location=dev)
        if state.get("signature") != resume_signature:
            raise ValueError(
                "Resume checkpoint does not match this run configuration.\n"
                f"saved={state.get('signature')}\ncurrent={resume_signature}"
            )
        steps.load_state_dict(state["steps_state"])
        opt.load_state_dict(state["optimizer_state"])
        history = state.get("history", [])
        best_val = float(state.get("best_val", float("inf")))
        best_row = state.get("best_row")
        best_steps_state = state.get("best_steps_state")
        start_epoch = int(state.get("next_epoch", len(history) + 1))
        elapsed_before = float(state.get("elapsed_sec", 0.0))
        t_start = time.time() - elapsed_before
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if str(dev).startswith("cuda") and torch.cuda.is_available() and state.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng_state_all"]])
        if "numpy_rng_state" in state:
            np.random.set_state(state["numpy_rng_state"])
        print(
            f"[resume] loaded {resume_path}; completed_epochs={start_epoch - 1}, "
            f"continuing through epoch {cfg.epochs}"
        )

    def save_training_state(next_epoch: int) -> None:
        payload = {
            "version": 1,
            "signature": resume_signature,
            "next_epoch": int(next_epoch),
            "steps_state": steps.state_dict(),
            "optimizer_state": opt.state_dict(),
            "best_val": float(best_val),
            "best_row": best_row,
            "best_steps_state": best_steps_state,
            "history": history,
            "elapsed_sec": float(time.time() - t_start),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all()
            if str(dev).startswith("cuda") and torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
        }
        tmp = resume_path.with_suffix(resume_path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(resume_path)

    for ep in range(start_epoch, cfg.epochs + 1):
        tr_m, tr_have = run_epoch(train_loader, train=True)
        va_m, va_have = run_epoch(val_loader, train=False)

        te_m = None
        if cfg.eval_test_each_epoch:
            te_m, _ = run_epoch(test_loader, train=False)

        with torch.no_grad():
            rho = float(steps.rho().cpu().item())
            s_val = float(steps.s().cpu().item())
            tau, sigma = steps.tau_sigma(cfg.dtype, dev)
            tau = float(tau.detach().mean().cpu().item())
            sigma = float(sigma.cpu().item())
            prod = float(tau * sigma * (nK ** 2))

        row = dict(epoch=ep, step_mode=args.step_mode, rho=rho, s=s_val, tau=tau, sigma=sigma, tau_sigma_K2=prod, nK=nK,
                   train=tr_m, val=va_m)
        if te_m is not None:
            row["test"] = te_m
        history.append(row)

        if cfg.save_history_every_epoch:
            export_history(history, out_dir, cfg.ckpt_name, latest=True, save_csv=cfg.save_csv)

        # format opt_gap (requires pg labels; shown as --- when unavailable)
        def _fmt_opt_gap(m):
            v = m.get("cost_rel_gap_abs", float("nan"))
            return f"{v:.3e}" if np.isfinite(v) else "---"

        print(f"\n[Epoch {ep:03d}] train={tr_m['L_total']:.4e}  val={va_m['L_total']:.4e}")
        print(f"  rho={rho:.4f}  s={s_val:+.3f}  tau={tau:.4e}  sigma={sigma:.4e}  tau*sigma*||K||^2={prod:.4f}  (||K||≈{nK:.4f})")
        print(
            f"  train: J_bal={tr_m['J_bal']:.3e}  J_flow={tr_m['J_flow']:.3e}  "
            f"|bal|={tr_m['bal_abs_MW']:.3e}  viol={tr_m['viol_MW']:.3e}  "
            f"opt_gap={_fmt_opt_gap(tr_m)}"
        )
        print(
            f"  val:   J_bal={va_m['J_bal']:.3e}  J_flow={va_m['J_flow']:.3e}  "
            f"|bal|={va_m['bal_abs_MW']:.3e} (max={va_m['bal_max_MW']:.3e})  "
            f"viol={va_m['viol_MW']:.3e} (max={va_m['viol_max_MW']:.3e})  "
            f"fwd_res={va_m['fwd_res']:.3e}  alg_res={va_m['alg_res']:.3e}  "
            f"opt_gap={_fmt_opt_gap(va_m)}"
        )
        if va_have:
            print(
                f"  val:   RMSE(pg)={va_m['rmse_MW_diag']:.3e}  "
                f"cost_gap={va_m['cost_gap']:.3e}  "
                f"bal_sat={va_m['bal_sat_rate']:.3f}  flow_sat={va_m['flow_sat_rate']:.3f}"
            )
        if (tr_m["anderson_fwd_fp64_fallback_rows"] > 0
                or tr_m["anderson_bwd_fp64_fallback_rows"] > 0
                or va_m["anderson_fwd_fp64_fallback_rows"] > 0):
            print(
                "  Anderson fp64 fallback row-events/batch: "
                f"train fwd={tr_m['anderson_fwd_fp64_fallback_rows']:.3f} "
                f"bwd={tr_m['anderson_bwd_fp64_fallback_rows']:.3f}  "
                f"val fwd={va_m['anderson_fwd_fp64_fallback_rows']:.3f}"
            )

        if va_m["L_total"] < best_val:
            best_val = va_m["L_total"]
            best_row = row
            best_steps_state = copy.deepcopy(steps.state_dict())

            ckpt_path = out_dir / cfg.ckpt_name
            torch.save(
                dict(
                    case_name=CASE_NAME,
                    code_version=CODE_VERSION,
                    physics=physics_signature,
                    line_limit_scale=LINE_LIMIT_SCALE,
                    load_scale_base=LOAD_SCALE,
                    nK=nK,
                    cfg=cfg.__dict__,
                    steps_state=steps.state_dict(),
                    history=history,
                ),
                ckpt_path,
            )
            export_history(history, out_dir, cfg.ckpt_name, latest=False, save_csv=cfg.save_csv)
            print(f"  Saved best checkpoint: {ckpt_path}")

        # Save current optimizer/step/RNG state after every completed epoch.
        # A time-limited or interrupted job loses at most the active epoch.
        save_training_state(ep + 1)

    # Reload best steps for final exports
    if best_steps_state is not None:
        steps.load_state_dict(best_steps_state)

    # Final test eval with best checkpoint. The one-epoch smoke job can skip
    # this duplicated work; its dependent continuation performs the full pass.
    test_metrics = {}
    if not args.skip_final_test:
        test_metrics, _ = run_epoch(test_loader, train=False)

    # Resource summary
    resources = {
        "train_time_sec": float(time.time() - t_start),
        "train_samples": int(n_train),
        "val_samples": int(n_val),
        "test_samples": int(n_test),
        "params_steps": int(sum(p.numel() for p in steps.parameters())),
    }
    if str(dev).startswith("cuda") and torch.cuda.is_available():
        resources["peak_cuda_mem_alloc_bytes"] = int(torch.cuda.max_memory_allocated(dev))
        resources["peak_cuda_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved(dev))
    if psutil is not None:
        try:
            resources["rss_bytes"] = int(psutil.Process().memory_info().rss)
        except Exception:
            pass

    export_summary(
        best_row,
        best_val,
        out_dir,
        cfg.ckpt_name,
        test=test_metrics,
        resources=resources,
        cfg=cfg,
        physics=physics_signature,
    )

    if cfg.save_plots:
        save_publishable_plots(history, fig_dir, stem, fmt=cfg.plot_format, dpi=cfg.plot_dpi)

    if cfg.eval_vs_gt and not args.skip_final_test:
        if cfg.eval_split == "train":
            ev = train_ds
        elif cfg.eval_split == "val":
            ev = val_ds
        elif cfg.eval_split == "test":
            ev = test_ds
        else:
            ev = ds
        eval_loader = torch.utils.data.DataLoader(ev, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
        eval_vs_gt_csv(op, deq, static, eval_loader, cfg, out_dir, stem)

    print("\nDone.")


def export_history(history: list[dict], out_dir: Path, ckpt_name: str, *, latest: bool, save_csv: bool) -> None:
    stem = Path(ckpt_name).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / (f"{stem}_latest.json" if latest else f"{stem}.json")
    csv_path = out_dir / (f"{stem}_latest.csv" if latest else f"{stem}.csv")
    json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    if save_csv and history:
        # flatten train/val/test metrics
        metric_keys = [
            "L_total", "J_cost", "J_bal", "J_flow", "J_flow_l1",
            "fwd_res", "alg_res",
            "bal_abs_MW", "bal_max_MW",
            "viol_MW", "viol_p95_MW", "viol_max_MW",
            "anderson_fwd_fp64_fallback_rows",
            "anderson_bwd_fp64_fallback_rows",
            "rmse_MW_diag",
            "cost_gap",
            "cost_rel_gap_abs",
            "bal_sat_rate",
            "flow_sat_rate",
        ]
        rows: List[Dict[str, Any]] = []
        for r in history:
            row: Dict[str, Any] = {
                "epoch": r.get("epoch"),
                "rho": r.get("rho"),
                "s": r.get("s"),
                "tau": r.get("tau"),
                "sigma": r.get("sigma"),
                "tau_sigma_K2": r.get("tau_sigma_K2"),
                "nK": r.get("nK"),
            }
            for split in ["train", "val", "test"]:
                sd = r.get(split, {}) or {}
                for k in metric_keys:
                    row[f"{split}_{k}"] = sd.get(k, float("nan"))
            rows.append(row)

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for rr in rows:
                w.writerow(rr)


def export_summary(best_row: Optional[dict], best_val: float, out_dir: Path, ckpt_name: str, *, test: Optional[dict] = None, resources: Optional[dict] = None, cfg: Optional[TrainCfg] = None, physics: Optional[dict] = None) -> None:
    stem = Path(ckpt_name).stem
    summary_path_json = out_dir / f"{stem}_summary.json"
    summary_path_txt = out_dir / f"{stem}_summary.txt"

    run_config = ({
        "epochs": int(cfg.epochs),
        "batch_size": int(cfg.batch_size),
        "lr": float(cfg.lr),
        "seed_split": int(cfg.seed_split),
        "step_mode": str(cfg.step_mode),
        "picard_iters": int(cfg.picard_iters),
        "anderson_m": int(cfg.anderson_m),
        "anderson_lam": float(cfg.anderson_lam),
        "anderson_max_iter": int(cfg.anderson_max_iter),
        "anderson_tol": float(cfg.anderson_tol),
        "b_anderson_m": int(cfg.b_anderson_m),
        "b_anderson_lam": float(cfg.b_anderson_lam),
        "b_anderson_max_iter": int(cfg.b_anderson_max_iter),
        "b_anderson_tol": float(cfg.b_anderson_tol),
        "w_cost": float(cfg.w_cost),
        "w_bal": float(cfg.w_bal),
        "w_flow": float(cfg.w_flow),
        "w_flow_l1": float(cfg.w_flow_l1),
    } if cfg is not None else {})

    if best_row is None:
        summary = {"code_version": CODE_VERSION, "physics": (physics or {}), "config": run_config, "best_epoch": None, "best_val_L_total": best_val, "note": "No epochs recorded.", "test": (test or {}), "resources": (resources or {})}
    else:
        summary = {
            "code_version": CODE_VERSION,
            "physics": (physics or {}),
            "config": run_config,
            "case_name": CASE_NAME,
            "best_epoch": best_row.get("epoch"),
            "best_val_L_total": best_val,
            "steps": {
                "rho": best_row.get("rho"),
                "s": best_row.get("s"),
                "tau": best_row.get("tau"),
                "sigma": best_row.get("sigma"),
                "tau_sigma_K2": best_row.get("tau_sigma_K2"),
                "nK": best_row.get("nK"),
            },
            "val": best_row.get("val", {}),
            "train": best_row.get("train", {}),
            "test": (test or {}),
            "resources": (resources or {}),
        }

    summary_path_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"code_version: {summary.get('code_version')}")
    lines.append(f"case_name: {summary.get('case_name')}")
    lines.append(f"best_epoch: {summary.get('best_epoch')}")
    lines.append(f"best_val_L_total: {summary.get('best_val_L_total')}")
    steps = summary.get("steps", {})
    if steps:
        lines.append("steps:")
        for k in ["rho", "s", "tau", "sigma", "tau_sigma_K2", "nK"]:
            lines.append(f"  {k}: {steps.get(k)}")
    val = summary.get("val", {})
    if val:
        lines.append("val_metrics:")
        for k, v in val.items():
            lines.append(f"  {k}: {v}")
    te = summary.get("test", {})
    if te:
        lines.append("test_metrics:")
        for k, v in te.items():
            lines.append(f"  {k}: {v}")
    res = summary.get("resources", {})
    if res:
        lines.append("resources:")
        for k, v in res.items():
            lines.append(f"  {k}: {v}")
    summary_path_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
