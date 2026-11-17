#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import inspect
import sys

import torch


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Network selection is read while the trainer/harness modules are imported, so
# consume this one option before importing the runner. Parsing it only in
# ``main`` can accidentally reuse a stale DEQ_CASE from the parent shell.
_bootstrap = argparse.ArgumentParser(add_help=False)
_bootstrap.add_argument("--case", default="case300")
_bootstrap_args, _ = _bootstrap.parse_known_args()
os.environ["DEQ_CASE"] = _bootstrap_args.case
import run_literature_baselines as R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="case300")
    ap.add_argument("--data", type=pathlib.Path)
    args = ap.parse_args()
    os.environ["DEQ_CASE"] = args.case
    data = args.data or (
        R.ROOT / "_local_mixed_contingency_smoke_v1" / "case300" /
        "dataset" / "dataset_mixed_N120_seed0.npz"
    )
    Pm = R.prepare(data, "cuda" if torch.cuda.is_available() else "cpu", "mixed")
    Pc = R.prepare(data, str(Pm.device), "clean")
    assert all(len(Pm.idx[name]) >= 4 for name in ("train", "val", "test"))
    assert all(Pc.contingency_type[R.tensor_index(v, Pc.device)].eq(0).all() for v in Pc.idx.values())
    assert "reserve" not in R.METHODS and "gnn" not in R.METHODS and "canos" in R.METHODS
    assert {"ref_bus", "branch_x", "branch_tap", "branch_shift_deg"}.issubset(Pm.static)

    for outage in (Pm.cont_line, Pm.cont_gen):
        sets = []
        for split in ("train", "val", "test"):
            values = outage[R.tensor_index(Pm.idx[split], Pm.device)]
            sets.append(set(values[values >= 0].detach().cpu().tolist()))
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    test = Pm.idx["test"]
    metrics, rows = R.score_predictions(Pm, Pm.pg[R.tensor_index(test, Pm.device)], test)
    assert metrics["all"]["cost_gap_abs_mean"] == 0.0
    assert metrics["all"]["cost_gap_raw_mean"] == 0.0
    assert metrics["all"]["rmse_MW"] == 0.0
    # Solver labels and float32 case data can differ by sub-kW tolerances at a
    # bound; require operational feasibility rather than bitwise equality.
    assert metrics["all"]["gen_bound_viol_MW"] <= 1e-3, metrics["all"]["gen_bound_viol_MW"]
    assert len(rows) == len(test)
    assert {r["scenario"] for r in rows}.issubset({"clean", "line", "generator"})
    assert all("cost_gap_raw" in r and "cost_gap_abs" in r for r in rows)

    # Equality-embedded and PDL-style decoders must both preserve balance.  The
    # former may expose a completed-generator bound violation, which is now a
    # reported inequality rather than being clipped away.
    b = R.tensor_index(test[:4], Pm.device)
    z = torch.full((b.numel(), Pm.ngen), 0.5, device=Pm.device)
    eq = R.equality_decode(Pm, z, b)
    pd = R.pdl_decode(Pm, z, b)
    eq_balance = torch.max(torch.abs(
        eq.double().sum(1) - Pm.d_eff[b].double().sum(1)
    )).item()
    pd_balance = torch.max(torch.abs(
        pd.double().sum(1) - Pm.d_eff[b].double().sum(1)
    )).item()
    assert eq_balance < 1e-2, eq_balance
    assert pd_balance < 5e-2, pd_balance
    assert R.inequality_violation(Pm, eq, b).shape[1] == Pm.nbranch + 2 * Pm.ngen

    # One topology-aware CANOS forward pass verifies dimensions and finite
    # scenario-conditioned graph execution without training.
    model = R.CANOSDC(Pm, hidden=8, steps=1).to(Pm.device).eval()
    b = R.tensor_index(test[:2], Pm.device)
    with torch.no_grad():
        z = model(Pm, b)
        p = R.decode(Pm, z, b)
    assert z.shape == (2, Pm.ngen)
    assert p.shape == (2, Pm.ngen)
    assert torch.isfinite(p).all()

    # CANOS inference must not depend on unrelated rows in the same batch.
    with torch.no_grad():
        z_single = model(Pm, b[:1])
    assert torch.allclose(z_single[0], z[0], atol=2e-5, rtol=2e-5)

    # The self-supervised implementations may use labels only in the final
    # scorer.  This source-level guard fails if dispatch-label helpers are
    # accidentally introduced into their training routines.
    for fn in (R.train_dc3, R.train_deeplde, R.train_pdl):
        source = inspect.getsource(fn)
        assert "scaled_pg(" not in source
        assert "P.pg" not in source
    print("[test-literature] PASS")


if __name__ == "__main__":
    main()
