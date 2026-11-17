#!/usr/bin/env python3
"""Literature-traceable DC-OPF baselines for the fair4net datasets.

This file intentionally distinguishes three levels of provenance:

* ``deepopf`` is a DC/mixed-contingency adaptation of the official
  Mzhou-cityu/DeepOPF-Codes architecture and slack reconstruction.
* ``dc3`` is a DC adaptation of the official locuslab/DC3 completion and
  differentiable correction algorithm.
* ``deeplde`` and ``pdl`` are paper-derived implementations because no public
  repository from the paper authors was located as of 2026-08-18.
* ``canos`` ports the CANOS Interaction-Network architecture to DC-OPF.  The
  implementation is informed by the paper and the MIT MOSSLab PFDelta
  third-party reimplementation; it is not original CANOS author code.

The runner supports two deliberately separate tasks:

``clean``
    Train/validate/test only on no-contingency samples.
``mixed``
    Train on clean plus disjoint line/generator N-1 identities and test on
    held-out outage identities.  Every model receives explicit scenario
    information; CANOS physically masks the outaged line in message passing.

Raw objective differences are first-class metrics.  Relative gaps are retained
only as secondary diagnostics.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import pathlib
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = pathlib.Path(__file__).resolve().parents[1]
STUDY = ROOT / "fair4net_l1_study"
if str(STUDY) not in sys.path:
    sys.path.insert(0, str(STUDY))

import baseline_harness_fair4net as H  # noqa: E402


METHODS = ("deepopf", "dc3", "deeplde", "pdl", "canos")
METHOD_LABELS = {
    "deepopf": "DeepOPF-DC (official-repo-derived adaptation)",
    "dc3": "DC3-DC (official-repo-derived adaptation)",
    "deeplde": "DeepLDE-DC (paper-derived adaptation)",
    "pdl": "PDL-DC (paper-derived adaptation)",
    "canos": "CANOS-DC (paper/PFDelta architecture port)",
}
PROTOCOL_VERSION = "literature_baselines_dc_v3"
FLOW_MODEL_CONTRACT = "f=PTDF@(Pg-PD-GS)+phase_shift_offset"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["torch_cuda"]])


def atomic_torch_save(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def torch_load_checkpoint(path: pathlib.Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before the weights_only keyword.
        return torch.load(path, map_location=device)


class TrainingCheckpoint:
    """Crash-safe method checkpoint plus validation-selected model state."""

    VERSION = "literature_baselines_dc_v3_resumable_1"

    def __init__(
        self,
        path: pathlib.Path,
        best_path: pathlib.Path,
        signature: dict,
        device: torch.device,
        resume: bool,
    ) -> None:
        self.path = path
        self.best_path = best_path
        self.signature = signature
        self.device = device
        self.resume = resume
        self.loaded: dict | None = None
        self.prior_elapsed_sec = 0.0
        self.session_start = time.perf_counter()
        self.best_score = float("inf")
        self.best_progress: dict | None = None

    def load(self) -> dict | None:
        if self.loaded is not None:
            return self.loaded
        if not self.resume or not self.path.is_file():
            self.loaded = {}
            return None
        state = torch_load_checkpoint(self.path, self.device)
        if state.get("checkpoint_version") != self.VERSION:
            raise ValueError(f"unsupported training checkpoint: {self.path}")
        if state.get("signature") != self.signature:
            raise ValueError(
                f"training checkpoint/config mismatch: {self.path}; "
                "use a new RUN_TAG instead of mixing experiments"
            )
        self.prior_elapsed_sec = float(state.get("elapsed_train_sec", 0.0))
        self.best_score = float(state.get("best_validation_score", float("inf")))
        self.best_progress = state.get("best_progress")
        if self.best_path.is_file():
            best = torch_load_checkpoint(self.best_path, self.device)
            if best.get("signature") != self.signature:
                raise ValueError(f"best-checkpoint/config mismatch: {self.best_path}")
            disk_score = float(best["validation_score"])
            if disk_score < self.best_score:
                self.best_score = disk_score
                self.best_progress = best["progress"]
        restore_rng_state(state["rng_state"])
        self.loaded = state
        print(
            f"[resume] {self.signature['method']} progress={state.get('progress')} "
            f"elapsed={self.prior_elapsed_sec:.1f}s best_val={self.best_score:.6g}",
            flush=True,
        )
        return state

    def elapsed_sec(self) -> float:
        return self.prior_elapsed_sec + (time.perf_counter() - self.session_start)

    def save(
        self,
        *,
        model: nn.Module,
        optimizers: dict[str, torch.optim.Optimizer],
        progress: dict,
        extra: dict | None = None,
        complete: bool = False,
    ) -> None:
        payload = {
            "checkpoint_version": self.VERSION,
            "signature": self.signature,
            "progress": progress,
            "model_state": model.state_dict(),
            "optimizer_states": {k: v.state_dict() for k, v in optimizers.items()},
            "extra": extra or {},
            "rng_state": capture_rng_state(),
            "elapsed_train_sec": self.elapsed_sec(),
            "best_validation_score": self.best_score,
            "best_progress": self.best_progress,
            "complete": bool(complete),
        }
        atomic_torch_save(self.path, payload)

    def update_best(self, model: nn.Module, score: float, progress: dict) -> bool:
        if not math.isfinite(score):
            raise ValueError(f"non-finite validation score at {progress}: {score!r}")
        if score >= self.best_score:
            return False
        self.best_score = float(score)
        self.best_progress = copy.deepcopy(progress)
        atomic_torch_save(
            self.best_path,
            {
                "checkpoint_version": self.VERSION,
                "signature": self.signature,
                "validation_score": self.best_score,
                "progress": self.best_progress,
                "model_state": model.state_dict(),
            },
        )
        print(
            f"[validation] {self.signature['method']} new_best={self.best_score:.6g} "
            f"progress={self.best_progress}",
            flush=True,
        )
        return True

    def load_best(self, model: nn.Module) -> None:
        if not self.best_path.is_file():
            raise FileNotFoundError(f"missing validation-selected checkpoint: {self.best_path}")
        best = torch_load_checkpoint(self.best_path, self.device)
        if best.get("signature") != self.signature:
            raise ValueError(f"best-checkpoint/config mismatch: {self.best_path}")
        model.load_state_dict(best["model_state"])
        self.best_score = float(best["validation_score"])
        self.best_progress = best["progress"]


def tensor_index(index: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(index, dtype=torch.long, device=device)


@dataclass
class Prepared:
    op: object
    static: dict
    device: torch.device
    protocol: str
    pd_bus: torch.Tensor
    d_eff: torch.Tensor
    pg: torch.Tensor
    pmin: torch.Tensor
    pmax_eff: torch.Tensor
    cont_line: torch.Tensor
    cont_gen: torch.Tensor
    contingency_type: torch.Tensor
    line_features: torch.Tensor
    X: torch.Tensor
    idx: dict[str, np.ndarray]
    source_index: np.ndarray
    completion_gen: torch.Tensor

    @property
    def nbus(self) -> int:
        return int(self.static["nbus"])

    @property
    def ngen(self) -> int:
        return int(self.static["ngen"])

    @property
    def nbranch(self) -> int:
        return int(self.static["nbranch"])


def _read_npz(path: pathlib.Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        required = {"pd_bus", "pg", "split"}
        missing = required - set(z.files)
        if missing:
            raise ValueError(f"dataset missing arrays: {sorted(missing)}")
        n = int(z["pd_bus"].shape[0])
        line = z["cont_line_idx"].astype(np.int64) if "cont_line_idx" in z.files else np.full(n, -1, np.int64)
        gen = z["cont_gen_idx"].astype(np.int64) if "cont_gen_idx" in z.files else np.full(n, -1, np.int64)
        ctype = z["contingency_type"].astype(np.int8) if "contingency_type" in z.files else np.where(line >= 0, 1, np.where(gen >= 0, 2, 0)).astype(np.int8)
        return {
            "pd_bus": z["pd_bus"].astype(np.float32),
            "pg": z["pg"].astype(np.float32),
            "split": z["split"].astype(np.int8),
            "cont_line": line,
            "cont_gen": gen,
            "ctype": ctype,
            "lodf": z["lodf"].astype(np.float32) if "lodf" in z.files else None,
            "gs_bus": z["gs_bus"].astype(np.float32) if "gs_bus" in z.files else None,
            "pd_bus_includes_gs": bool(int(np.asarray(z["pd_bus_includes_gs"]).item())) if "pd_bus_includes_gs" in z.files else None,
        }


def prepare(path: pathlib.Path, device: str, protocol: str) -> Prepared:
    if protocol not in {"clean", "mixed"}:
        raise ValueError(protocol)
    dev = torch.device(device)
    op, static, _ = H.build_operator(device)
    raw = _read_npz(path)
    contract = {
        "pd_bus": raw["pd_bus"], "pg": raw["pg"], "split": raw["split"],
        "cont": raw["cont_gen"], "cont_line": raw["cont_line"], "lodf": raw["lodf"],
        "gs_bus": raw["gs_bus"], "pd_bus_includes_gs": raw["pd_bus_includes_gs"],
    }
    H.validate_contract(contract, static, require_metadata=True)
    if raw["lodf"] is not None:
        expected = (int(static["nbranch"]), int(static["nbranch"]))
        if raw["lodf"].shape != expected or not np.isfinite(raw["lodf"]).all():
            raise ValueError(f"invalid LODF {raw['lodf'].shape}; expected {expected}")
        op.LODF = torch.as_tensor(raw["lodf"], dtype=op.PTDF.dtype, device=dev)

    n = raw["pd_bus"].shape[0]
    pd_bus = torch.as_tensor(raw["pd_bus"], dtype=torch.float32, device=dev)
    pg = torch.as_tensor(raw["pg"], dtype=torch.float32, device=dev)
    d_eff = op.effective_demand(pd_bus)
    cont_line = torch.as_tensor(raw["cont_line"], dtype=torch.long, device=dev)
    cont_gen = torch.as_tensor(raw["cont_gen"], dtype=torch.long, device=dev)
    ctype = torch.as_tensor(raw["ctype"], dtype=torch.long, device=dev)
    derived_ctype = torch.where(cont_line >= 0, 1, torch.where(cont_gen >= 0, 2, 0))
    if torch.any((cont_line >= 0) & (cont_gen >= 0)):
        raise ValueError("dataset contains simultaneous line and generator outages")
    if not torch.equal(ctype, derived_ctype):
        raise ValueError("contingency_type is inconsistent with outage-index arrays")
    pmax_eff = H.effective_pmax(static, raw["cont_gen"], device)
    pmin0 = torch.as_tensor(static["pmin"], dtype=torch.float32, device=dev)
    pmin = pmin0[None, :].expand(n, -1).clone()
    # An outaged generator has both bounds equal to zero.  This matters when a
    # test network contains a generator with positive PMIN.
    valid_gen = cont_gen >= 0
    if torch.any(valid_gen):
        rows = torch.arange(n, device=dev)[valid_gen]
        pmin[rows, cont_gen[valid_gen]] = 0.0

    line_features = torch.zeros(n, int(static["nbranch"]), dtype=torch.float32, device=dev)
    valid_line = cont_line >= 0
    if torch.any(valid_line):
        line_features[valid_line] = op.LODF[:, cont_line[valid_line]].t()

    split = raw["split"]
    idx = {}
    for name, code in (("train", 0), ("val", 1), ("test", 2)):
        mask = split == code
        if protocol == "clean":
            mask &= raw["ctype"] == 0
        idx[name] = np.where(mask)[0]
        if idx[name].size == 0:
            raise ValueError(f"protocol={protocol} has no {name} rows")

    if protocol == "mixed":
        # The mixed benchmark is deliberately an outage-identity OOD test.
        # Checking this at load time prevents a seemingly successful run from
        # silently training on a held-out topology.
        for outage_name, values in (("line", raw["cont_line"]), ("generator", raw["cont_gen"])):
            identities = {
                name: set(values[split == code][values[split == code] >= 0].tolist())
                for name, code in (("train", 0), ("val", 1), ("test", 2))
            }
            for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                overlap = identities[left] & identities[right]
                if overlap:
                    raise ValueError(
                        f"{outage_name} outage identities overlap between {left}/{right}: "
                        f"{sorted(overlap)}"
                    )

    if protocol == "clean":
        dense_raw = d_eff
    else:
        # The selected LODF column is a continuous topology descriptor that
        # remains meaningful for held-out line identities. pmax_eff is the
        # explicit generator-availability descriptor.
        dense_raw = torch.cat([d_eff, pmax_eff, line_features], dim=1)
    tr = tensor_index(idx["train"], dev)
    mu = dense_raw[tr].mean(0, keepdim=True)
    sd = dense_raw[tr].std(0, keepdim=True).clamp_min(1e-6)
    X = (dense_raw - mu) / sd

    # Equality-embedding methods complete one generator analytically, matching
    # DeepOPF's slack reconstruction and DC3/DeepLDE partial completion.  The
    # reference-bus generators are preferred; if one is itself outaged, the
    # largest remaining unit is selected deterministically for that sample.
    ref_bus = int(static.get("ref_bus", 0))
    bus_of_gen = np.asarray(static["bus_of_gen"], dtype=np.int64)
    pmax_np = np.asarray(static["pmax"], dtype=np.float64)
    priority = sorted(
        range(int(static["ngen"])),
        key=lambda g: (bus_of_gen[g] != ref_bus, -pmax_np[g], g),
    )
    available = (pmax_eff - pmin) > 1e-6
    completion_gen = torch.full((n,), -1, dtype=torch.long, device=dev)
    unresolved = torch.ones(n, dtype=torch.bool, device=dev)
    for g in priority:
        choose = unresolved & available[:, g]
        completion_gen[choose] = int(g)
        unresolved &= ~choose
    if torch.any(unresolved):
        raise ValueError("at least one sample has no available equality-completion generator")
    return Prepared(
        op=op, static=static, device=dev, protocol=protocol,
        pd_bus=pd_bus, d_eff=d_eff, pg=pg, pmin=pmin,
        pmax_eff=pmax_eff, cont_line=cont_line, cont_gen=cont_gen,
        contingency_type=ctype, line_features=line_features, X=X, idx=idx,
        source_index=np.arange(n, dtype=np.int64),
        completion_gen=completion_gen,
    )


def batch_indices(index: np.ndarray, batch_size: int, device: torch.device, shuffle: bool) -> Iterable[torch.Tensor]:
    t = tensor_index(index, device)
    if shuffle:
        t = t[torch.randperm(t.numel(), device=device)]
    for start in range(0, t.numel(), batch_size):
        yield t[start:start + batch_size]


def fixed_step_batches(index: np.ndarray, batch_size: int, device: torch.device, steps: int) -> Iterable[torch.Tensor]:
    """Yield exactly ``steps`` shuffled mini-batches, cycling as needed."""
    produced = 0
    while produced < steps:
        for batch in batch_indices(index, batch_size, device, True):
            yield batch
            produced += 1
            if produced >= steps:
                return


def distributed_balance(P: Prepared, p: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """PDL-style differentiable repair over all available headroom."""
    p = H.box_clamp(p, P.pmin[idx], P.pmax_eff[idx])
    return H.balance_repair(p, P.d_eff[idx], P.pmin[idx], P.pmax_eff[idx])


def equality_complete_from_pg(P: Prepared, p: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Complete one generator so that total generation equals total demand.

    Non-completion generators are kept inside their scenario-dependent bounds.
    The completed generator is intentionally *not* clipped: any resulting bound
    excess is a real inequality violation and is penalized/reported.  Clipping
    it would destroy the equality embedding and hide DeepOPF/DC3/DeepLDE
    failures behind the shared feasibility repair used by the old harness.
    """
    p = H.box_clamp(p, P.pmin[idx], P.pmax_eff[idx])
    rows = torch.arange(idx.numel(), device=p.device)
    complete = P.completion_gen[idx]
    without = p.clone()
    without[rows, complete] = 0.0
    # Accumulate the large network-wide sums in float64.  With PEGASE-scale
    # totals, a float32 reduction alone can introduce a visible MW-scale
    # equality residual even though the analytic formula is exact.
    target64 = P.d_eff[idx].double().sum(1)
    completed_value = (target64 - without.double().sum(1)).to(p.dtype)
    out = without
    out[rows, complete] = completed_value
    # Casting the completed value back to the model dtype may reintroduce one
    # final ULP. Correct that residual once, again measuring it in float64.
    residual = (target64 - out.double().sum(1)).to(p.dtype)
    corrected = out.clone()
    corrected[rows, complete] = out[rows, complete] + residual
    out = corrected
    return out


def equality_decode(P: Prepared, z: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    p = P.pmin[idx] + z * (P.pmax_eff[idx] - P.pmin[idx])
    return equality_complete_from_pg(P, p, idx)


def pdl_decode(P: Prepared, z: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    p = P.pmin[idx] + z * (P.pmax_eff[idx] - P.pmin[idx])
    return distributed_balance(P, p, idx)


def scaled_pg(P: Prepared, idx: torch.Tensor) -> torch.Tensor:
    width = (P.pmax_eff[idx] - P.pmin[idx]).clamp_min(1e-6)
    return ((P.pg[idx] - P.pmin[idx]) / width).clamp(0.0, 1.0)


def normalized_violation(P: Prepared, p: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    v = P.op.flow_violation_vec(p, P.pd_bus[idx], P.cont_line[idx])
    fmax = torch.as_tensor(P.static["fmax"], dtype=p.dtype, device=p.device)
    return v / fmax[None, :].clamp_min(1e-3)


def normalized_bound_violation(P: Prepared, p: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Upper/lower generator violations, scaled by each unit's usable range."""
    scale = (P.pmax_eff[idx] - P.pmin[idx]).clamp_min(1.0)
    upper = (p - P.pmax_eff[idx]).clamp_min(0.0) / scale
    lower = (P.pmin[idx] - p).clamp_min(0.0) / scale
    return torch.cat([upper, lower], dim=1)


def inequality_violation(P: Prepared, p: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [normalized_violation(P, p, idx), normalized_bound_violation(P, p, idx)],
        dim=1,
    )


@torch.no_grad()
def unsupervised_cost_scale(P: Prepared) -> torch.Tensor:
    """Objective scale computed without reading solver dispatch labels."""
    values = []
    for b in batch_indices(P.idx["train"], 256, P.device, False):
        # Capacity-proportional dispatch is a deterministic feasible reference
        # derived only from demand and operational bounds.
        p = distributed_balance(P, P.pmin[b].clone(), b)
        values.append(P.op.gen_cost(p).abs())
    return torch.cat(values).mean().clamp_min(1.0)


def deeplde_epoch_schedule(total_epochs: int) -> tuple[int, list[int]]:
    """Scale the paper's 100 + (25,30,...,95) = 1000 epoch schedule."""
    if total_epochs < 16:
        return 0, [1] * total_epochs
    weights = np.asarray([100] + [25 + 5 * t for t in range(15)], dtype=np.float64)
    raw = weights * (float(total_epochs) / float(weights.sum()))
    allocation = np.floor(raw).astype(np.int64)
    remainder = total_epochs - int(allocation.sum())
    order = np.argsort(-(raw - allocation), kind="stable")
    allocation[order[:remainder]] += 1
    return int(allocation[0]), [int(x) for x in allocation[1:]]


def scenario_label(ctype: torch.Tensor) -> list[str]:
    names = {0: "clean", 1: "line", 2: "generator"}
    return [names[int(x)] for x in ctype.detach().cpu().tolist()]


@torch.no_grad()
def score_predictions(P: Prepared, pred: torch.Tensor, idx_np: np.ndarray) -> tuple[dict, list[dict]]:
    idx = tensor_index(idx_np, P.device)
    true = P.pg[idx]
    pd = P.pd_bus[idx]
    c_line = P.cont_line[idx]
    c_gen = P.cont_gen[idx]
    d_eff = P.d_eff[idx]
    bal = pred.sum(1) - d_eff.sum(1)
    vmat = P.op.flow_violation_vec(pred, pd, c_line)
    worst = vmat.max(1).values
    upper = (pred - P.pmax_eff[idx]).clamp_min(0.0)
    lower = (P.pmin[idx] - pred).clamp_min(0.0)
    gen_bound = torch.maximum(upper, lower).max(1).values
    cp = P.op.gen_cost(pred)
    ct = P.op.gen_cost(true)
    gap = cp - ct
    abs_gap = gap.abs()
    rel_abs = abs_gap / ct.abs().clamp_min(1e-9)
    sample_rmse = torch.sqrt(torch.mean((pred - true) ** 2, dim=1))

    def aggregate(mask: torch.Tensor) -> dict:
        count = int(mask.sum().item())
        if count == 0:
            return {"samples": 0}
        g = gap[mask]; ag = abs_gap[mask]; w = worst[mask]; b = bal[mask].abs(); gb = gen_bound[mask]
        return {
            "samples": count,
            "cost_gap_raw_mean": float(g.mean()),
            "cost_gap_raw_median": float(g.median()),
            "cost_gap_abs_mean": float(ag.mean()),
            "cost_gap_abs_median": float(ag.median()),
            "cost_gap_abs_p95": float(torch.quantile(ag, 0.95)),
            "cost_rel_gap_abs_mean": float(rel_abs[mask].mean()),
            "rmse_MW": float(torch.sqrt(torch.mean((pred[mask] - true[mask]) ** 2))),
            "bal_abs_MW": float(b.mean()),
            "bal_max_MW": float(b.max()),
            "gen_bound_viol_MW": float(gb.mean()),
            "gen_bound_viol_p95_MW": float(torch.quantile(gb, 0.95)),
            "gen_bound_viol_max_MW": float(gb.max()),
            "gen_bound_sat_rate_1e-3": float((gb <= 1e-3).float().mean()),
            "viol_MW": float(w.mean()),
            "viol_p95_MW": float(torch.quantile(w, 0.95)),
            "viol_max_MW": float(w.max()),
            "flow_sat_rate_1e-3": float((w <= 1e-3).float().mean()),
        }

    ctype = P.contingency_type[idx]
    masks = {
        "all": torch.ones_like(ctype, dtype=torch.bool),
        "clean": ctype == 0,
        "line": ctype == 1,
        "generator": ctype == 2,
    }
    metrics = {name: aggregate(mask) for name, mask in masks.items()}
    metrics["cost_gap_definition"] = "predicted quadratic objective minus solver-label objective"
    metrics["cost_gap_units"] = "native PGLib generator-cost objective units per dispatch interval"
    metrics["primary_cost_metric"] = "cost_gap_abs_mean"
    metrics["primary_flow_metric"] = "viol_MW=mean over samples of worst branch overload"

    labels = scenario_label(ctype)
    rows = []
    for j, src in enumerate(idx_np.tolist()):
        rows.append({
            "source_index": int(src),
            "scenario": labels[j],
            "cont_line_idx": int(c_line[j].item()),
            "cont_gen_idx": int(c_gen[j].item()),
            "cost_true": float(ct[j].item()),
            "cost_pred": float(cp[j].item()),
            "cost_gap_raw": float(gap[j].item()),
            "cost_gap_abs": float(abs_gap[j].item()),
            "cost_rel_gap_abs": float(rel_abs[j].item()),
            "rmse_MW": float(sample_rmse[j].item()),
            "balance_abs_MW": float(bal[j].abs().item()),
            "gen_bound_violation_MW": float(gen_bound[j].item()),
            "flow_violation_MW": float(worst[j].item()),
            "completion_gen_idx": int(P.completion_gen[idx][j].item()),
        })
    return metrics, rows


class DeepOPFNet(nn.Module):
    """Architecture copied structurally from DeepOPF-SCDCOPF.

    The official code uses 4h -> 2h -> h ReLU layers and a sigmoid output.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_base: int = 8):
        super().__init__()
        self.l1 = nn.Linear(in_dim, 4 * hidden_base)
        self.l2 = nn.Linear(4 * hidden_base, 2 * hidden_base)
        self.l3 = nn.Linear(2 * hidden_base, hidden_base)
        self.out = nn.Linear(hidden_base, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.out(F.relu(self.l3(F.relu(self.l2(F.relu(self.l1(x))))))))


class DC3Net(nn.Module):
    """Two hidden-layer DC3 repository architecture."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class EqualityEmbeddedMLP(nn.Module):
    """DeepLDE paper setting: two 200-unit ELU layers, dropout 0.1."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 200), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(200, 200), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(200, out_dim), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourLayerMLP(nn.Module):
    """Four-layer ReLU network used for PDL-SCOPF primal/dual models."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, layer_norm_input: bool):
        super().__init__()
        layers: list[nn.Module] = []
        if layer_norm_input:
            layers.append(nn.LayerNorm(in_dim))
        d = in_dim
        for _ in range(4):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PDLBundle(nn.Module):
    """Checkpointable pair of primal and dual networks used during PDL."""

    def __init__(self, in_dim: int, primal_dim: int, dual_dim: int, hidden: int):
        super().__init__()
        self.primal = FourLayerMLP(in_dim, primal_dim, hidden, layer_norm_input=True)
        self.dual = FourLayerMLP(in_dim, dual_dim, hidden, layer_norm_input=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.primal(x)


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CANOSDC(nn.Module):
    """DC adaptation of CANOS' residual Interaction Network.

    Branch messages are scenario-specific: an outaged line is masked in both
    directions. Generator availability is likewise an explicit node feature.
    """

    def __init__(self, P: Prepared, hidden: int = 64, steps: int = 12):
        super().__init__()
        self.hidden = hidden
        self.steps = steps
        f, t = H.build_graph()
        f = torch.as_tensor(f, dtype=torch.long, device=P.device)
        t = torch.as_tensor(t, dtype=torch.long, device=P.device)
        branch_id = torch.arange(P.nbranch, device=P.device)
        self.register_buffer("src", torch.cat([f, t]))
        self.register_buffer("dst", torch.cat([t, f]))
        self.register_buffer("branch_id", torch.cat([branch_id, branch_id]))
        self.register_buffer(
            "edge_direction",
            torch.cat([
                torch.ones(P.nbranch, dtype=torch.float32, device=P.device),
                -torch.ones(P.nbranch, dtype=torch.float32, device=P.device),
            ]),
        )
        self.register_buffer("bus_of_gen", torch.as_tensor(P.static["bus_of_gen"], dtype=torch.long, device=P.device))

        fmax = torch.as_tensor(P.static["fmax"], dtype=torch.float32, device=P.device)
        finite = fmax < 1e8
        fscale = torch.where(finite, fmax, torch.zeros_like(fmax))
        if torch.any(finite):
            fscale = fscale / fscale[finite].median().clamp_min(1.0)
        branch_x = torch.as_tensor(P.static["branch_x"], dtype=torch.float32, device=P.device)
        xscale = branch_x / branch_x.abs().median().clamp_min(1e-6)
        tap = torch.as_tensor(P.static["branch_tap"], dtype=torch.float32, device=P.device)
        shift = torch.as_tensor(P.static["branch_shift_deg"], dtype=torch.float32, device=P.device) / 30.0
        offset = torch.as_tensor(P.static["flow_offset"], dtype=torch.float32, device=P.device) / 100.0
        # CANOS uses local branch parameters.  The prior port supplied a PTDF
        # row norm instead, which leaked a global operator summary and omitted
        # reactance/tap information.  These are the DC-relevant local features.
        directed_sign = self.edge_direction
        self.register_buffer(
            "edge_static",
            torch.stack([
                fscale[self.branch_id],
                xscale[self.branch_id],
                tap[self.branch_id],
                shift[self.branch_id] * directed_sign,
                offset[self.branch_id] * directed_sign,
                finite.float()[self.branch_id],
                directed_sign,
            ], dim=1),
        )

        # These normalizers are fixed from the training split (or immutable
        # case data).  Batch-derived scales make inference depend on which
        # unrelated samples happen to share a batch and are therefore invalid.
        train = tensor_index(P.idx["train"], P.device)
        train_d = P.d_eff[train]
        train_cap = torch.zeros(
            train.numel(), P.nbus, dtype=train_d.dtype, device=train_d.device,
        )
        train_cap.scatter_add_(
            1,
            self.bus_of_gen[None, :].expand(train.numel(), -1),
            P.pmax_eff[train],
        )
        c1 = torch.as_tensor(P.static["c1"], dtype=torch.float32, device=P.device)
        c2 = torch.as_tensor(P.static["c2"], dtype=torch.float32, device=P.device)
        self.register_buffer("d_scale", train_d.abs().mean().clamp_min(1.0))
        self.register_buffer("cap_scale", train_cap.abs().mean().clamp_min(1.0))
        self.register_buffer("p_scale", P.pmax_eff[train].abs().max().clamp_min(1.0))
        self.register_buffer("c1_scale", c1.abs().mean().clamp_min(1.0))
        self.register_buffer("c2_scale", c2.abs().mean().clamp_min(1.0))

        self.bus_encoder = nn.Linear(4, hidden)
        self.gen_encoder = nn.Linear(5, hidden)
        self.edge_encoder = nn.Linear(8, hidden)
        self.edge_updates = nn.ModuleList([MLPBlock(3 * hidden, hidden) for _ in range(steps)])
        self.bus_updates = nn.ModuleList([MLPBlock(2 * hidden, hidden) for _ in range(steps)])
        self.gen_to_bus = nn.ModuleList([MLPBlock(2 * hidden, hidden) for _ in range(steps)])
        self.bus_to_gen = nn.ModuleList([MLPBlock(2 * hidden, hidden) for _ in range(steps)])
        self.gen_updates = nn.ModuleList([MLPBlock(2 * hidden, hidden) for _ in range(steps)])
        self.decoder = nn.Sequential(nn.Linear(hidden + 2, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())

    @staticmethod
    def scatter_bus(msg: torch.Tensor, index: torch.Tensor, nbus: int) -> torch.Tensor:
        b, _, h = msg.shape
        out = torch.zeros(b, nbus, h, dtype=msg.dtype, device=msg.device)
        out.scatter_add_(1, index[None, :, None].expand(b, -1, h), msg)
        return out

    def forward(self, P: Prepared, idx: torch.Tensor) -> torch.Tensor:
        b = idx.numel()
        d = P.d_eff[idx]
        cap = torch.zeros(b, P.nbus, dtype=d.dtype, device=d.device)
        cap.scatter_add_(1, self.bus_of_gen[None, :].expand(b, -1), P.pmax_eff[idx])
        gen_count = torch.zeros(P.nbus, dtype=d.dtype, device=d.device)
        gen_count.scatter_add_(0, self.bus_of_gen, torch.ones(P.ngen, dtype=d.dtype, device=d.device))
        ref = torch.zeros(P.nbus, dtype=d.dtype, device=d.device)
        ref[int(P.static.get("ref_bus", 0))] = 1.0
        bus_raw = torch.stack([
            d / self.d_scale,
            cap / self.cap_scale,
            gen_count[None, :].expand(b, -1),
            ref[None, :].expand(b, -1),
        ], dim=2)

        pmax = P.pmax_eff[idx]
        pmin = P.pmin[idx]
        available = (pmax > 0).float()
        c1 = torch.as_tensor(P.static["c1"], dtype=d.dtype, device=d.device)[None, :].expand(b, -1)
        c2 = torch.as_tensor(P.static["c2"], dtype=d.dtype, device=d.device)[None, :].expand(b, -1)
        gen_raw = torch.stack([
            pmin / self.p_scale, pmax / self.p_scale,
            c1 / self.c1_scale, c2 / self.c2_scale, available,
        ], dim=2)

        line = P.cont_line[idx]
        active_branch = torch.ones(b, P.nbranch, dtype=d.dtype, device=d.device)
        valid = line >= 0
        if torch.any(valid):
            active_branch[torch.arange(b, device=d.device)[valid], line[valid]] = 0.0
        active_edge = active_branch[:, self.branch_id]
        edge_raw = torch.cat([self.edge_static[None, :, :].expand(b, -1, -1), active_edge[:, :, None]], dim=2)

        hb = self.bus_encoder(bus_raw)
        hg = self.gen_encoder(gen_raw)
        he = self.edge_encoder(edge_raw)
        for k in range(self.steps):
            src_h = hb[:, self.src, :]
            dst_h = hb[:, self.dst, :]
            edge_delta = self.edge_updates[k](torch.cat([he, src_h, dst_h], dim=2)) * active_edge[:, :, None]
            he = he + edge_delta
            branch_msg = self.scatter_bus(he * active_edge[:, :, None], self.dst, P.nbus)
            g2b = self.gen_to_bus[k](torch.cat([hg, hb[:, self.bus_of_gen, :]], dim=2)) * available[:, :, None]
            gen_msg = self.scatter_bus(g2b, self.bus_of_gen, P.nbus)
            hb = hb + self.bus_updates[k](torch.cat([hb, branch_msg + gen_msg], dim=2))
            b2g = self.bus_to_gen[k](torch.cat([hb[:, self.bus_of_gen, :], hg], dim=2))
            hg = hg + self.gen_updates[k](torch.cat([hg, b2g], dim=2))
        z = self.decoder(torch.cat([
            hg,
            (pmin / self.p_scale)[:, :, None],
            (pmax / self.p_scale)[:, :, None],
        ], dim=2)).squeeze(2)
        return z


def decode(P: Prepared, z: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Backward-compatible name for equality-embedded decoding."""
    return equality_decode(P, z, idx)


def predict_batched(P: Prepared, fn: Callable[[torch.Tensor], torch.Tensor], index: np.ndarray, batch_size: int) -> torch.Tensor:
    out = []
    for b in batch_indices(index, batch_size, P.device, False):
        out.append(fn(b))
    return torch.cat(out, dim=0)


def validation_score(
    P: Prepared,
    pred_fn: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int,
) -> float:
    """Scale-free held-out score used only for checkpoint selection.

    Solver labels are permitted for validation/model selection even for the
    self-supervised training methods.  The constraint term uses the same
    scenario operator and normalized limits used in training.
    """
    pred = predict_batched(P, pred_fn, P.idx["val"], batch_size).detach()
    idx = tensor_index(P.idx["val"], P.device)
    scale = (P.pmax_eff[idx] - P.pmin[idx]).clamp_min(1.0)
    nrmse = torch.sqrt(torch.mean(((pred - P.pg[idx]) / scale) ** 2))
    violation = inequality_violation(P, pred, idx).mean()
    return float((nrmse + 10.0 * violation).item())


def train_deepopf(
    P: Prepared,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_base: int,
    ckpt: TrainingCheckpoint,
    validation_interval: int,
) -> tuple[nn.Module, Callable]:
    model = DeepOPFNet(P.X.shape[1], P.ngen, hidden_base=hidden_base).to(P.device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer_steps = 0
    start_epoch = 0
    state = ckpt.load()
    if state:
        model.load_state_dict(state["model_state"])
        opt.load_state_dict(state["optimizer_states"]["optimizer"])
        start_epoch = int(state["progress"]["epoch"])
        optimizer_steps = int(state["extra"]["optimizer_steps"])
    pred_fn = lambda b: equality_decode(P, model(P.X[b]), b)
    for epoch in range(start_epoch, epochs):
        model.train()
        for b in batch_indices(P.idx["train"], batch_size, P.device, True):
            z = model(P.X[b])
            p = equality_decode(P, z, b)
            target = scaled_pg(P, b)
            rows = torch.arange(b.numel(), device=P.device)
            predicted_mask = torch.ones_like(z)
            predicted_mask[rows, P.completion_gen[b]] = 0.0
            supervised = (((z - target) ** 2) * predicted_mask).sum() / predicted_mask.sum().clamp_min(1.0)
            # Official SCDCOPF adds squared hinge penalties for base and
            # contingency branch loadings. Here the selected scenario operator
            # provides the corresponding per-example constraint.  Slack-bound
            # penalties are retained instead of clipping the reconstructed
            # generator and silently breaking power balance.
            constraint_pen = torch.mean(inequality_violation(P, p, b) ** 2)
            loss = supervised + constraint_pen
            opt.zero_grad(); loss.backward(); opt.step()
            optimizer_steps += 1
        progress = {"epoch": epoch + 1, "target_epochs": epochs}
        validation_due = (epoch + 1) % validation_interval == 0 or epoch + 1 == epochs
        if validation_due:
            model.eval()
            with torch.no_grad():
                score = validation_score(P, pred_fn, batch_size)
            ckpt.update_best(model, score, progress)
            ckpt.save(
                model=model,
                optimizers={"optimizer": opt},
                progress=progress,
                extra={"optimizer_steps": optimizer_steps},
            )
        if validation_due:
            print(
                f"[progress] deepopf epoch={epoch + 1}/{epochs} "
                f"steps={optimizer_steps} elapsed={ckpt.elapsed_sec():.1f}s",
                flush=True,
            )
    if not ckpt.best_path.is_file():
        model.eval()
        with torch.no_grad():
            ckpt.update_best(model, validation_score(P, pred_fn, batch_size), {"epoch": epochs, "target_epochs": epochs})
    ckpt.load_best(model)
    model.eval()
    model.training_metadata = {
        "optimizer_steps": optimizer_steps,
        "architecture": f"4h-2h-h ReLU, h={hidden_base}, sigmoid output",
        "optimizer": "SGD(momentum=0.9)",
        "supervision": "dispatch labels + selected-scenario inequality augmentation",
        "selection": "minimum validation normalized RMSE + 10x normalized inequality violation",
        "best_validation_score": ckpt.best_score,
        "best_progress": ckpt.best_progress,
    }
    ckpt.save(
        model=model,
        optimizers={"optimizer": opt},
        progress={"epoch": epochs, "target_epochs": epochs},
        extra={"optimizer_steps": optimizer_steps},
        complete=True,
    )
    return model, pred_fn


def dc3_correct(P: Prepared, p: torch.Tensor, idx: torch.Tensor, steps: int, lr: float, momentum: float, create_graph: bool) -> torch.Tensor:
    partial = p
    old = torch.zeros_like(partial)
    rows = torch.arange(idx.numel(), device=partial.device)
    complete = P.completion_gen[idx]
    for _ in range(steps):
        partial = partial.requires_grad_(True)
        completed = equality_complete_from_pg(P, partial, idx)
        violation = inequality_violation(P, completed, idx)
        if not create_graph and float(violation.max().detach().item()) <= 1e-4:
            return completed.detach()
        penalty = torch.sum(violation ** 2, dim=1).mean()
        grad = torch.autograd.grad(penalty, partial, create_graph=create_graph)[0]
        grad = grad.clone()
        grad[rows, complete] = 0.0
        step = lr * grad + momentum * old
        partial = H.box_clamp(partial - step, P.pmin[idx], P.pmax_eff[idx])
        old = step
    return equality_complete_from_pg(P, partial, idx)


def train_dc3(
    P: Prepared,
    epochs: int,
    batch_size: int,
    lr: float,
    ckpt: TrainingCheckpoint,
    validation_interval: int,
) -> tuple[nn.Module, Callable]:
    model = DC3Net(P.X.shape[1], P.ngen, hidden=200).to(P.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    cost_scale = unsupervised_cost_scale(P)
    optimizer_steps = 0
    start_epoch = 0
    state = ckpt.load()
    if state:
        model.load_state_dict(state["model_state"])
        opt.load_state_dict(state["optimizer_states"]["optimizer"])
        start_epoch = int(state["progress"]["epoch"])
        optimizer_steps = int(state["extra"]["optimizer_steps"])

    def pred(b: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            p0 = equality_decode(P, model(P.X[b]), b)
            p = dc3_correct(P, p0, b, steps=10, lr=1e-4, momentum=0.5, create_graph=False)
        return p.detach()

    for epoch in range(start_epoch, epochs):
        model.train()
        for b in batch_indices(P.idx["train"], batch_size, P.device, True):
            if b.numel() == 1:  # BatchNorm requires at least two training rows.
                continue
            p0 = equality_decode(P, model(P.X[b]), b)
            p = dc3_correct(P, p0, b, steps=5, lr=1e-4, momentum=0.5, create_graph=True)
            cost = P.op.gen_cost(p).mean() / cost_scale
            ineq = torch.norm(inequality_violation(P, p, b), dim=1).mean()
            loss = cost + 10.0 * ineq
            opt.zero_grad(); loss.backward(); opt.step()
            optimizer_steps += 1
        progress = {"epoch": epoch + 1, "target_epochs": epochs}
        validation_due = (epoch + 1) % validation_interval == 0 or epoch + 1 == epochs
        if validation_due:
            model.eval()
            ckpt.update_best(model, validation_score(P, pred, batch_size), progress)
            ckpt.save(
                model=model,
                optimizers={"optimizer": opt},
                progress=progress,
                extra={"optimizer_steps": optimizer_steps},
            )
        if validation_due:
            print(
                f"[progress] dc3 epoch={epoch + 1}/{epochs} "
                f"steps={optimizer_steps} elapsed={ckpt.elapsed_sec():.1f}s",
                flush=True,
            )
    if not ckpt.best_path.is_file():
        model.eval()
        ckpt.update_best(model, validation_score(P, pred, batch_size), {"epoch": epochs, "target_epochs": epochs})
    ckpt.load_best(model)
    model.eval()
    model.training_metadata = {
        "optimizer_steps": optimizer_steps,
        "architecture": "two 200-unit BN/ReLU/dropout hidden layers",
        "optimizer": "Adam",
        "supervision": "self-supervised objective + inequalities",
        "correction_train_steps": 5,
        "correction_test_max_steps": 10,
        "correction_tolerance": 1e-4,
        "correction_learning_rate": 1e-4,
        "correction_momentum": 0.5,
        "selection": "minimum validation normalized RMSE + 10x normalized inequality violation",
        "best_validation_score": ckpt.best_score,
        "best_progress": ckpt.best_progress,
    }
    ckpt.save(
        model=model,
        optimizers={"optimizer": opt},
        progress={"epoch": epochs, "target_epochs": epochs},
        extra={"optimizer_steps": optimizer_steps},
        complete=True,
    )
    return model, pred


def train_deeplde(
    P: Prepared,
    epochs: int,
    batch_size: int,
    lr: float,
    ckpt: TrainingCheckpoint,
    validation_interval: int,
) -> tuple[nn.Module, Callable]:
    model = EqualityEmbeddedMLP(P.X.shape[1], P.ngen).to(P.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dual = torch.zeros(P.nbranch + 2 * P.ngen, dtype=torch.float32, device=P.device)
    model.register_buffer("learned_inequality_dual", dual.detach().clone())
    cost_scale = unsupervised_cost_scale(P)
    warmup_epochs, outer_schedule = deeplde_epoch_schedule(epochs)
    optimizer_steps = 0
    warmup_done = 0
    outer_start = 0
    inner_start = 0
    state = ckpt.load()
    if state:
        model.load_state_dict(state["model_state"])
        opt.load_state_dict(state["optimizer_states"]["optimizer"])
        optimizer_steps = int(state["extra"]["optimizer_steps"])
        dual = state["extra"]["dual"].to(P.device)
        warmup_done = int(state["progress"].get("warmup_done", warmup_epochs))
        outer_start = int(state["progress"].get("outer_index", 0))
        inner_start = int(state["progress"].get("inner_done", 0))
    pred_fn = lambda b: equality_decode(P, model(P.X[b]), b)

    def primal_epoch() -> None:
        nonlocal optimizer_steps
        model.train()
        for b in batch_indices(P.idx["train"], batch_size, P.device, True):
            p = equality_decode(P, model(P.X[b]), b)
            v = inequality_violation(P, p, b)
            loss = P.op.gen_cost(p).mean() / cost_scale + (dual[None, :] * v).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            optimizer_steps += 1

    for epoch in range(warmup_done, warmup_epochs):
        primal_epoch()
        completed = epoch + 1
        progress = {
            "warmup_done": completed,
            "outer_index": 0,
            "inner_done": 0,
            "completed_epochs": completed,
            "target_epochs": epochs,
        }
        validation_due = completed % validation_interval == 0 or completed == epochs
        if validation_due:
            model.eval()
            with torch.no_grad():
                ckpt.update_best(model, validation_score(P, pred_fn, batch_size), progress)
            ckpt.save(
                model=model,
                optimizers={"optimizer": opt},
                progress=progress,
                extra={"optimizer_steps": optimizer_steps, "dual": dual.detach()},
            )
    for t in range(outer_start, len(outer_schedule)):
        inner_epochs = outer_schedule[t]
        begin = inner_start if t == outer_start else 0
        for inner in range(begin, inner_epochs):
            primal_epoch()
            completed = warmup_epochs + sum(outer_schedule[:t]) + inner + 1
            progress = {
                "warmup_done": warmup_epochs,
                "outer_index": t,
                "inner_done": inner + 1,
                "completed_epochs": completed,
                "target_epochs": epochs,
            }
            validation_due = completed % validation_interval == 0 or completed == epochs
            if validation_due:
                model.eval()
                with torch.no_grad():
                    ckpt.update_best(model, validation_score(P, pred_fn, batch_size), progress)
                ckpt.save(
                    model=model,
                    optimizers={"optimizer": opt},
                    progress=progress,
                    extra={"optimizer_steps": optimizer_steps, "dual": dual.detach()},
                )
            if validation_due:
                print(
                    f"[progress] deeplde epoch={completed}/{epochs} "
                    f"outer={t + 1}/{len(outer_schedule)} steps={optimizer_steps} "
                    f"elapsed={ckpt.elapsed_sec():.1f}s",
                    flush=True,
                )
        with torch.no_grad():
            vsum = torch.zeros_like(dual); count = 0
            for b in batch_indices(P.idx["train"], batch_size, P.device, False):
                p = equality_decode(P, model(P.X[b]), b)
                vsum += inequality_violation(P, p, b).sum(0); count += b.numel()
            rho = 0.05 / (1.0 + 0.01 * t)
            dual = torch.clamp(dual + rho * vsum / max(count, 1), min=0.0)
        progress = {
            "warmup_done": warmup_epochs,
            "outer_index": t + 1,
            "inner_done": 0,
            "completed_epochs": warmup_epochs + sum(outer_schedule[:t + 1]),
            "target_epochs": epochs,
        }
        ckpt.save(
            model=model,
            optimizers={"optimizer": opt},
            progress=progress,
            extra={"optimizer_steps": optimizer_steps, "dual": dual.detach()},
        )
        inner_start = 0
    if not ckpt.best_path.is_file():
        model.eval()
        with torch.no_grad():
            ckpt.update_best(
                model,
                validation_score(P, pred_fn, batch_size),
                {"completed_epochs": epochs, "target_epochs": epochs},
            )
    ckpt.load_best(model)
    model.learned_inequality_dual.copy_(dual.detach())
    model.eval()
    model.training_metadata = {
        "optimizer_steps": optimizer_steps,
        "architecture": "two 200-unit ELU/dropout=0.1 hidden layers",
        "optimizer": "Adam",
        "supervision": "self-supervised objective + equality-embedded Lagrangian",
        "warmup_epochs": warmup_epochs,
        "outer_inner_epoch_schedule": outer_schedule,
        "rho0": 0.05,
        "rho_decay_gamma": 0.01,
        "selection": "minimum validation normalized RMSE + 10x normalized inequality violation",
        "best_validation_score": ckpt.best_score,
        "best_progress": ckpt.best_progress,
    }
    ckpt.save(
        model=model,
        optimizers={"optimizer": opt},
        progress={
            "warmup_done": warmup_epochs,
            "outer_index": len(outer_schedule),
            "inner_done": 0,
            "completed_epochs": epochs,
            "target_epochs": epochs,
        },
        extra={"optimizer_steps": optimizer_steps, "dual": dual.detach()},
        complete=True,
    )
    return model, pred_fn


def train_pdl(
    P: Prepared,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_cap: int,
    outer_iters: int,
    inner_steps: int,
    ckpt: TrainingCheckpoint,
) -> tuple[nn.Module, Callable]:
    hidden = max(64, min(hidden_cap, int(1.5 * P.X.shape[1])))
    bundle = PDLBundle(P.X.shape[1], P.ngen, P.nbranch + 2 * P.ngen, hidden).to(P.device)
    primal, dual = bundle.primal, bundle.dual
    # PDL initializes the final dual layer at zero.
    final = [m for m in dual.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.zeros_(final.weight); nn.init.zeros_(final.bias)
    popt = torch.optim.Adam(primal.parameters(), lr=lr)
    dopt = torch.optim.Adam(dual.parameters(), lr=lr)
    cost_scale = unsupervised_cost_scale(P)
    outer = max(1, min(outer_iters, epochs)) if inner_steps <= 0 else outer_iters
    batches_per_epoch = max(1, math.ceil(len(P.idx["train"]) / batch_size))
    realized_inner_steps = (
        inner_steps if inner_steps > 0 else max(1, math.ceil(epochs * batches_per_epoch / outer))
    )
    rho = 1.0
    prev_violation = float("inf")
    primal_steps = 0
    dual_steps = 0
    start_outer = 0
    state = ckpt.load()
    if state:
        bundle.load_state_dict(state["model_state"])
        popt.load_state_dict(state["optimizer_states"]["primal"])
        dopt.load_state_dict(state["optimizer_states"]["dual"])
        start_outer = int(state["progress"]["outer_done"])
        rho = float(state["extra"]["rho"])
        prev_violation = float(state["extra"]["prev_violation"])
        primal_steps = int(state["extra"]["primal_steps"])
        dual_steps = int(state["extra"]["dual_steps"])
    pred_fn = lambda b: pdl_decode(P, torch.sigmoid(bundle.primal(P.X[b])), b)
    for outer_idx in range(start_outer, outer):
        dual.eval()
        primal.train()
        for b in fixed_step_batches(P.idx["train"], batch_size, P.device, realized_inner_steps):
            z = torch.sigmoid(primal(P.X[b])); p = pdl_decode(P, z, b)
            v = inequality_violation(P, p, b)
            lam = dual(P.X[b]).clamp_min(0.0).detach()
            loss = P.op.gen_cost(p).mean() / cost_scale + (lam * v).sum(1).mean() + 0.5 * rho * (v ** 2).sum(1).mean()
            popt.zero_grad(); loss.backward(); popt.step()
            primal_steps += 1
        frozen = copy.deepcopy(dual).eval()
        primal.eval(); dual.train()
        max_v = 0.0
        for b in fixed_step_batches(P.idx["train"], batch_size, P.device, realized_inner_steps):
            with torch.no_grad():
                p = pdl_decode(P, torch.sigmoid(primal(P.X[b])), b)
                v = inequality_violation(P, p, b)
                old = frozen(P.X[b]).clamp_min(0.0)
                target = torch.clamp(old + rho * v, min=0.0)
                max_v = max(max_v, float(v.max().item()))
            # Regress the raw output to the nonnegative ALM target.  Applying
            # ReLU here would give the paper's zero-initialized final layer a
            # zero gradient and freeze the dual network at initialization.
            pred_lam = dual(P.X[b])
            dloss = F.mse_loss(pred_lam, target)
            dopt.zero_grad(); dloss.backward(); dopt.step()
            dual_steps += 1
        if max_v > 0.9 * prev_violation:
            rho = min(10.0 * rho, 1e6)
        prev_violation = max_v
        progress = {"outer_done": outer_idx + 1, "target_outer": outer}
        bundle.eval()
        with torch.no_grad():
            ckpt.update_best(bundle, validation_score(P, pred_fn, batch_size), progress)
        ckpt.save(
            model=bundle,
            optimizers={"primal": popt, "dual": dopt},
            progress=progress,
            extra={
                "rho": rho,
                "prev_violation": prev_violation,
                "primal_steps": primal_steps,
                "dual_steps": dual_steps,
            },
        )
        print(
            f"[progress] pdl outer={outer_idx + 1}/{outer} "
            f"primal_steps={primal_steps} dual_steps={dual_steps} "
            f"rho={rho:.3g} elapsed={ckpt.elapsed_sec():.1f}s",
            flush=True,
        )
    if not ckpt.best_path.is_file():
        bundle.eval()
        with torch.no_grad():
            ckpt.update_best(bundle, validation_score(P, pred_fn, batch_size), {"outer_done": outer, "target_outer": outer})
    ckpt.load_best(bundle)
    bundle.eval()
    bundle.training_metadata = {
        "optimizer_steps": primal_steps + dual_steps,
        "primal_optimizer_steps": primal_steps,
        "dual_optimizer_steps": dual_steps,
        "outer_iterations": outer,
        "inner_steps_per_network_per_outer": realized_inner_steps,
        "architecture": "four hidden ReLU layers for primal and dual networks",
        "optimizer": "Adam",
        "supervision": "self-supervised ALM-inspired primal/dual learning",
        "paper_hidden_width": int(1.5 * P.X.shape[1]),
        "realized_hidden_width": hidden,
        "selection": "minimum validation normalized RMSE + 10x normalized inequality violation",
        "best_validation_score": ckpt.best_score,
        "best_progress": ckpt.best_progress,
    }
    ckpt.save(
        model=bundle,
        optimizers={"primal": popt, "dual": dopt},
        progress={"outer_done": outer, "target_outer": outer},
        extra={
            "rho": rho,
            "prev_violation": prev_violation,
            "primal_steps": primal_steps,
            "dual_steps": dual_steps,
        },
        complete=True,
    )
    return bundle, pred_fn


def train_canos(
    P: Prepared,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    steps: int,
    requested_optimizer_steps: int,
    ckpt: TrainingCheckpoint,
    checkpoint_step_interval: int,
    validation_step_interval: int,
) -> tuple[nn.Module, Callable]:
    model = CANOSDC(P, hidden=hidden, steps=steps).to(P.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_steps = 0
    total_steps = (
        requested_optimizer_steps
        if requested_optimizer_steps > 0 else
        epochs * max(1, math.ceil(len(P.idx["train"]) / batch_size))
    )
    state = ckpt.load()
    if state:
        model.load_state_dict(state["model_state"])
        opt.load_state_dict(state["optimizer_states"]["optimizer"])
        optimizer_steps = int(state["progress"]["optimizer_steps"])
    batches: Iterable[torch.Tensor] = fixed_step_batches(
        P.idx["train"], batch_size, P.device, total_steps - optimizer_steps,
    )
    pred_fn = lambda b: equality_decode(P, model(P, b), b)
    model.train()
    for b in batches:
        z = model(P, b)
        p = equality_decode(P, z, b)
        supervised = F.mse_loss(z, scaled_pg(P, b))
        constraint = inequality_violation(P, p, b).mean()
        loss = supervised + 0.1 * constraint
        opt.zero_grad(); loss.backward(); opt.step()
        optimizer_steps += 1
        checkpoint_due = (
            optimizer_steps % checkpoint_step_interval == 0 or optimizer_steps == total_steps
        )
        validation_due = (
            optimizer_steps % validation_step_interval == 0 or optimizer_steps == total_steps
        )
        progress = {"optimizer_steps": optimizer_steps, "target_optimizer_steps": total_steps}
        if validation_due:
            model.eval()
            with torch.no_grad():
                ckpt.update_best(model, validation_score(P, pred_fn, batch_size), progress)
            model.train()
        if checkpoint_due or validation_due:
            ckpt.save(
                model=model,
                optimizers={"optimizer": opt},
                progress=progress,
                extra={},
            )
        if validation_due:
            print(
                f"[progress] canos steps={optimizer_steps}/{total_steps} "
                f"elapsed={ckpt.elapsed_sec():.1f}s",
                flush=True,
            )
    if not ckpt.best_path.is_file():
        model.eval()
        with torch.no_grad():
            ckpt.update_best(
                model,
                validation_score(P, pred_fn, batch_size),
                {"optimizer_steps": total_steps, "target_optimizer_steps": total_steps},
            )
    ckpt.load_best(model)
    model.eval()
    model.training_metadata = {
        "optimizer_steps": optimizer_steps,
        "architecture": "heterogeneous residual Interaction Network encoder/process/decoder",
        "optimizer": "Adam",
        "supervision": "dispatch labels + constraint augmentation (C=0.1)",
        "message_passing_steps": steps,
        "hidden_width": hidden,
        "requested_optimizer_steps": requested_optimizer_steps,
        "selection": "minimum validation normalized RMSE + 10x normalized inequality violation",
        "best_validation_score": ckpt.best_score,
        "best_progress": ckpt.best_progress,
    }
    ckpt.save(
        model=model,
        optimizers={"optimizer": opt},
        progress={"optimizer_steps": total_steps, "target_optimizer_steps": total_steps},
        extra={},
        complete=True,
    )
    return model, pred_fn


def write_per_sample(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    ap.add_argument("--protocol", required=True, choices=("clean", "mixed"))
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--deepopf-hidden-base", type=int, default=8,
                    help="DeepOPF base width h in its unchanged 4h-2h-h architecture")
    ap.add_argument("--canos-hidden", type=int, default=64)
    ap.add_argument("--canos-steps", type=int, default=12)
    ap.add_argument(
        "--canos-optimizer-steps",
        type=int,
        default=0,
        help="exact CANOS optimizer-step budget; 0 uses --epochs over the training split",
    )
    ap.add_argument(
        "--pdl-hidden-cap",
        type=int,
        default=1024,
        help=(
            "resource cap for the paper's 1.5*input hidden width; the realized "
            "width and uncapped paper width are recorded in every PDL result"
        ),
    )
    ap.add_argument("--pdl-outer-iters", type=int, default=20)
    ap.add_argument(
        "--pdl-inner-steps",
        type=int,
        default=0,
        help="optimizer steps for each primal and dual phase; 0 derives a smoke/resource budget from --epochs",
    )
    ap.add_argument(
        "--smoke-limit",
        action="store_true",
        help="execution test only: deterministically truncate train/val/test to 84/12/24 rows",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="resume a matching crash-safe training state when present",
    )
    ap.add_argument(
        "--validation-interval",
        type=int,
        default=50,
        help="epoch interval for validation selection and dense-method checkpoints",
    )
    ap.add_argument(
        "--checkpoint-step-interval",
        type=int,
        default=5000,
        help="CANOS optimizer-step interval for crash-safe checkpoints",
    )
    ap.add_argument(
        "--validation-step-interval",
        type=int,
        default=10000,
        help="CANOS optimizer-step interval for validation selection",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    case = os.environ.get("DEQ_CASE")
    if not case:
        raise RuntimeError("DEQ_CASE must select the network")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if (
        args.epochs < 1 or args.batch_size < 1 or args.pdl_hidden_cap < 64
        or args.pdl_outer_iters < 1 or args.pdl_inner_steps < 0
        or args.canos_optimizer_steps < 0 or args.validation_interval < 1
        or args.deepopf_hidden_base < 1
        or args.checkpoint_step_interval < 1 or args.validation_step_interval < 1
    ):
        raise ValueError("invalid epoch, batch, or PDL iteration configuration")
    seed_all(args.seed)
    P = prepare(args.data, args.device, args.protocol)
    if args.smoke_limit:
        limits = {"train": 84, "val": 12, "test": 24}
        P.idx = {name: values[:limits[name]] for name, values in P.idx.items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_sha = sha256_file(args.data)
    split_counts = {k: int(v.size) for k, v in P.idx.items()}
    print(f"[literature] case={case} protocol={args.protocol} counts={split_counts} device={P.device}")

    summary_rows = []
    for method in args.methods:
        stem = f"{method}_{args.protocol}_seed{args.seed}"
        result_path = args.out_dir / f"{stem}.json"
        if result_path.exists() and not args.force:
            old = json.loads(result_path.read_text(encoding="utf-8"))
            if old.get("dataset_sha256") == dataset_sha and old.get("protocol_version") == PROTOCOL_VERSION:
                print(f"[literature] skip valid {method}: {result_path}")
                summary_rows.append(old)
                continue
        seed_all(args.seed)
        signature = {
            "protocol_version": PROTOCOL_VERSION,
            "dataset_sha256": dataset_sha,
            "case": case,
            "protocol": args.protocol,
            "method": method,
            "seed": args.seed,
            "split_counts": split_counts,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "deepopf_hidden_base": args.deepopf_hidden_base,
            "canos_hidden": args.canos_hidden,
            "canos_steps": args.canos_steps,
            "canos_optimizer_steps": args.canos_optimizer_steps,
            "pdl_hidden_cap": args.pdl_hidden_cap,
            "pdl_outer_iters": args.pdl_outer_iters,
            "pdl_inner_steps": args.pdl_inner_steps,
            "validation_interval": args.validation_interval,
            "checkpoint_step_interval": args.checkpoint_step_interval,
            "validation_step_interval": args.validation_step_interval,
        }
        ckpt = TrainingCheckpoint(
            args.out_dir / f"{stem}_training_state.pt",
            args.out_dir / f"{stem}_best_validation.pt",
            signature,
            P.device,
            args.resume,
        )
        runners = {
            "deepopf": lambda: train_deepopf(
                P, args.epochs, args.batch_size, args.lr, args.deepopf_hidden_base, ckpt, args.validation_interval,
            ),
            "dc3": lambda: train_dc3(
                P, args.epochs, args.batch_size, args.lr, ckpt, args.validation_interval,
            ),
            "deeplde": lambda: train_deeplde(
                P, args.epochs, args.batch_size, args.lr, ckpt, args.validation_interval,
            ),
            "pdl": lambda: train_pdl(
                P, args.epochs, args.batch_size, args.lr, args.pdl_hidden_cap,
                args.pdl_outer_iters, args.pdl_inner_steps, ckpt,
            ),
            "canos": lambda: train_canos(
                P, args.epochs, args.batch_size, args.lr, args.canos_hidden,
                args.canos_steps, args.canos_optimizer_steps, ckpt,
                args.checkpoint_step_interval, args.validation_step_interval,
            ),
        }
        print(f"[literature] training {method} epochs={args.epochs}")
        model, pred_fn = runners[method]()
        train_sec = ckpt.elapsed_sec()
        with torch.no_grad() if method != "dc3" else torch.enable_grad():
            pred = predict_batched(P, pred_fn, P.idx["test"], args.batch_size)
        metrics, samples = score_predictions(P, pred.detach(), P.idx["test"])
        result = {
            "case": case,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "protocol": args.protocol,
            "protocol_version": PROTOCOL_VERSION,
            "dataset": str(args.data),
            "dataset_sha256": dataset_sha,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "split_counts": split_counts,
            "smoke_limit": bool(args.smoke_limit),
            "scenario_encoding": (
                "demand only; clean rows" if args.protocol == "clean" else
                ("explicit outaged-edge mask + generator availability" if method == "canos" else "demand + effective generator bounds + selected LODF column")
            ),
            "test_outage_identity_policy": "disjoint from training for mixed protocol",
            "flow_model_contract": FLOW_MODEL_CONTRACT,
            "equality_handling": (
                "differentiable all-generator headroom repair" if method == "pdl" else
                "single available slack-priority generator analytic completion"
            ),
            "training_uses_solver_dispatch_labels": method in {"deepopf", "canos"},
            "training_metadata": getattr(model, "training_metadata", {}),
            "train_time_sec": train_sec,
            "model_selection": {
                "split": "validation",
                "criterion": "normalized dispatch RMSE + 10x normalized inequality violation",
                "best_score": ckpt.best_score,
                "best_progress": ckpt.best_progress,
            },
            "metrics": metrics,
            "canos_config": (
                {
                    "hidden": args.canos_hidden,
                    "interaction_steps": args.canos_steps,
                    "requested_optimizer_steps": args.canos_optimizer_steps,
                    "paper_reference_optimizer_steps": "600000-800000",
                    "paper_reference_interaction_steps": "36/48/60",
                }
                if method == "canos" else None
            ),
            "pdl_config": (
                {
                    "paper_hidden_width": int(1.5 * P.X.shape[1]),
                    "realized_hidden_width": max(64, min(args.pdl_hidden_cap, int(1.5 * P.X.shape[1]))),
                    "hidden_cap": args.pdl_hidden_cap,
                    "resource_matched": int(1.5 * P.X.shape[1]) > args.pdl_hidden_cap,
                    "requested_outer_iterations": args.pdl_outer_iters,
                    "requested_inner_steps": args.pdl_inner_steps,
                }
                if method == "pdl" else None
            ),
        }
        torch.save({"model": model.state_dict(), "result": result}, args.out_dir / f"{stem}.pt")
        atomic_json(result_path, result)
        write_per_sample(args.out_dir / f"{stem}_per_sample.csv", samples)
        summary_rows.append(result)
        allm = metrics["all"]
        print(
            f"[literature] {method} raw_abs_gap={allm['cost_gap_abs_mean']:.6g} "
            f"raw_signed_gap={allm['cost_gap_raw_mean']:.6g} "
            f"viol={allm['viol_MW']:.6g}MW rmse={allm['rmse_MW']:.6g}MW"
        )
        if str(P.device).startswith("cuda"):
            torch.cuda.empty_cache()
    atomic_json(args.out_dir / f"summary_{args.protocol}_seed{args.seed}.json", {"rows": summary_rows})


if __name__ == "__main__":
    main()
