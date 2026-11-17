"""
DC-OPF on an IEEE test case using pandapower (full network constraints).

Single-place settings
---------------------
Change ONLY the constants in the "USER SETTINGS" block below.
Do NOT pass overrides elsewhere (the script uses Config() with those defaults).

What you get
------------
- Full DC-OPF via pandapower.rundcopp (network constraints included).
- Heterogeneous convex quadratic generator costs -> non-equal dispatch.
- Consistent voltage limits to avoid vm_pu warnings (DC ignores |V| anyway).
- Line/trafo limits scaling:
    * scale < 1  => TIGHTEN limits (higher loading %)
    * scale > 1  => LOOSEN limits (lower loading %)

Run
---
  pip install pandapower
  python dc_opf_pandapower_case300.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict

import os
import numpy as np


# ============================================================
# USER SETTINGS
# ============================================================
# CASE_NAME / LINE_LIMIT_SCALE may be overridden per-process via env vars
# (DEQ_CASE, DEQ_LINE_SCALE) so multiple networks can run concurrently on a
# cluster without editing this file. When unset, defaults are unchanged
# (case9241pegase, scale 10.0) so existing behavior is preserved.
#
# Per-network line-limit scale calibrated so base DC-OPF is congested (some
# binding lines) yet robustly feasible under per-load demand in [0.9, 1.1]:
#   case300        -> 0.3  (native limits loose; tighten to create congestion)
#   case1354pegase -> 3.0  (native limits tight; loosen until feasible+congested)
#   case1888rte    -> 3.0
#   case9241pegase -> 10.0
_DEFAULT_LINE_SCALE = {
    "case300": 0.3,
    "case1354pegase": 3.0,
    "case2869pegase": 10.0,   # feasible only with loose limits (like case9241); low congestion
    "case1888rte": 3.0,       # NOTE: RTE sgens enter the OPF gen matrix uncosted -> not trainable as-is
    "case9241pegase": 10.0,
}

CASE_NAME = os.environ.get("DEQ_CASE", "case9241pegase")
VERBOSE = False              # True prints long PYPOWER summary
LINE_LIMIT_SCALE = float(
    os.environ.get("DEQ_LINE_SCALE", _DEFAULT_LINE_SCALE.get(CASE_NAME, 1.0))
)                            # <1 tightens, >1 loosens
LOAD_SCALE = 1.0             # increase to raise flows (e.g. 1.5 or 2.0)
GEN_MARGIN_MW = 5.0          # keep solution away from gen bounds by this margin
VM_BOUNDS = (0.9, 1.1)       # consistent bus |V| bounds (DC-OPF keeps |V|≈1)
SEED = 7                     # random seed for heterogeneous quadratic costs
# ============================================================


@dataclass
class Config:
    case_name: str = CASE_NAME
    verbose: bool = VERBOSE
    line_limit_scale: float = LINE_LIMIT_SCALE
    load_scale: float = LOAD_SCALE
    gen_margin_mw: float = GEN_MARGIN_MW
    vm_bounds: Tuple[float, float] = VM_BOUNDS
    seed: int = SEED


def load_ieee_case(case_name: str):
    import pandapower.networks as pn
    cases = {"case14": pn.case14, "case30": pn.case30, "case57": pn.case57, "case118": pn.case118, "case300": pn.case300, "case1354pegase": pn.case1354pegase, "case1888rte": pn.case1888rte, "case2869pegase": pn.case2869pegase, "case9241pegase": pn.case9241pegase}
    if case_name not in cases:
        raise ValueError(f"Unknown case_name={case_name}. Choose from {list(cases.keys())}.")
    return cases[case_name]()


def ensure_controllable(net):
    if len(net.gen):
        net.gen.loc[:, "controllable"] = True
    if len(net.ext_grid):
        net.ext_grid.loc[:, "controllable"] = True


def set_voltage_limits(net, vmin: float, vmax: float):
    # Make bus limits consistent with generators/ext_grid setpoints
    if "min_vm_pu" in net.bus.columns:
        net.bus.loc[:, "min_vm_pu"] = float(vmin)
    if "max_vm_pu" in net.bus.columns:
        net.bus.loc[:, "max_vm_pu"] = float(vmax)

    # DC-OPF doesn't optimize |V|, but pandapower does consistency checks
    if len(net.gen) and "vm_pu" in net.gen.columns:
        net.gen.loc[:, "vm_pu"] = 1.0
    if len(net.ext_grid) and "vm_pu" in net.ext_grid.columns:
        net.ext_grid.loc[:, "vm_pu"] = 1.0


def scale_loads(net, load_scale: float):
    if abs(load_scale - 1.0) > 1e-12 and len(net.load):
        net.load.loc[:, "p_mw"] *= float(load_scale)
        if "q_mvar" in net.load.columns:
            net.load.loc[:, "q_mvar"] *= float(load_scale)


def scale_line_and_trafo_limits(net, factor: float):
    """
    factor < 1 tightens limits (increases loading%)
    factor > 1 loosens limits (decreases loading%)
    """
    factor = float(factor)

    if len(net.line):
        if "max_i_ka" in net.line.columns:
            # some entries can be NaN; only scale finite values
            mask = np.isfinite(net.line["max_i_ka"].astype(float).values)
            net.line.loc[mask, "max_i_ka"] = net.line.loc[mask, "max_i_ka"].astype(float) * factor
        if "max_loading_percent" in net.line.columns:
            net.line.loc[:, "max_loading_percent"] = net.line["max_loading_percent"].astype(float) * factor

    if len(net.trafo):
        if "max_loading_percent" in net.trafo.columns:
            net.trafo.loc[:, "max_loading_percent"] = net.trafo["max_loading_percent"].astype(float) * factor
        # sn_mva is not a "limit" in the same sense, but pandapower may use it for loading; keep it unchanged by default.


def controllables(net) -> List[Tuple[str, int]]:
    els: List[Tuple[str, int]] = []
    for idx in net.ext_grid.index.tolist():
        els.append(("ext_grid", int(idx)))
    for idx in net.gen.index.tolist():
        els.append(("gen", int(idx)))
    if not els:
        raise RuntimeError("No controllable generators found (ext_grid/gen).")
    return els


def add_heterogeneous_quadratic_costs_and_bounds(net, gen_margin_mw: float, seed: int):
    """
    Costs: sum_i a_i * p_i^2 (convex), with different a_i -> non-equal dispatch.

    We also create wide [pmin,pmax] around the expected uncongested dispatch
    so the solution stays away from generator bounds.
    """
    import pandapower as pp

    # wipe any existing costs
    if hasattr(net, "poly_cost") and len(net.poly_cost):
        net.poly_cost.drop(net.poly_cost.index, inplace=True)
    if hasattr(net, "pwl_cost") and len(net.pwl_cost):
        net.pwl_cost.drop(net.pwl_cost.index, inplace=True)

    els = controllables(net)
    n = len(els)
    load = float(net.load.p_mw.sum()) if len(net.load) else 0.0

    rng = np.random.default_rng(seed)
    a = 1.0 + 0.5 * rng.random(n) + 0.4 * np.arange(n) / max(1, n - 1)  # mild spread, all > 0

    inva = 1.0 / a
    p_hat = load * inva / inva.sum()  # ED solution if no congestion

    # bounds: wide enough to avoid binding (and leave at least gen_margin_mw slack)
    slack = max(50.0, 0.25 * load)  # MW
    for i, (et, idx) in enumerate(els):
        pmin = max(0.0, float(p_hat[i] - slack))
        pmax = float(p_hat[i] + slack)

        if et == "gen":
            net.gen.at[idx, "min_p_mw"] = pmin
            net.gen.at[idx, "max_p_mw"] = pmax
        else:
            net.ext_grid.at[idx, "min_p_mw"] = pmin
            net.ext_grid.at[idx, "max_p_mw"] = pmax

        pp.create_poly_cost(
            net,
            element=idx,
            et=et,
            cp2_eur_per_mw2=float(a[i]),
            cp1_eur_per_mw=0.0,
            cp0_eur=0.0,
        )

    return {els[i]: {"a": float(a[i]), "p_hat": float(p_hat[i])} for i in range(n)}


def run_dc_opf(net, verbose: bool):
    import pandapower as pp
    if hasattr(pp, "rundcopp"):
        pp.rundcopp(net, verbose=verbose)
        return "rundcopp"
    raise RuntimeError("pandapower is missing rundcopp; please update: pip install -U pandapower")


def check_generator_margins(net, gen_margin_mw: float) -> float:
    margins = []
    if len(net.gen):
        for idx in net.gen.index:
            p = float(net.res_gen.at[idx, "p_mw"])
            pmin = float(net.gen.at[idx, "min_p_mw"])
            pmax = float(net.gen.at[idx, "max_p_mw"])
            margins += [p - pmin, pmax - p]
    if len(net.ext_grid):
        for idx in net.ext_grid.index:
            p = float(net.res_ext_grid.at[idx, "p_mw"])
            pmin = float(net.ext_grid.at[idx, "min_p_mw"])
            pmax = float(net.ext_grid.at[idx, "max_p_mw"])
            margins += [p - pmin, pmax - p]

    min_margin = float(min(margins)) if margins else float("inf")
    if min_margin < float(gen_margin_mw):
        raise AssertionError(f"Generator solution too close to a bound. min_gen_margin={min_margin:.3f} MW")
    return min_margin


# -----------------------------
# NEW: cost printing helpers
# -----------------------------
def _get_p_mw(net, et: str, element: int):
    if et == "gen" and len(net.res_gen) and element in net.res_gen.index:
        return float(net.res_gen.at[element, "p_mw"])
    if et == "ext_grid" and len(net.res_ext_grid) and element in net.res_ext_grid.index:
        return float(net.res_ext_grid.at[element, "p_mw"])
    return None


def compute_active_poly_cost(net):
    """
    Returns:
      total_cost_eur: float
      per_element: dict[(et, element)] -> dict with p_mw, cp0, cp1, cp2, cost_eur

    Notes:
      - Sums over all rows in net.poly_cost.
      - Uses active power only (cp2*p^2 + cp1*p + cp0).
      - This matches how costs are created in this script (cp2 only, cp1/cp0 are 0).
    """
    total = 0.0
    per = {}

    if not hasattr(net, "poly_cost") or len(net.poly_cost) == 0:
        return 0.0, per

    for _, row in net.poly_cost.iterrows():
        et = str(row["et"])
        element = int(row["element"])
        p = _get_p_mw(net, et, element)
        if p is None:
            continue

        cp0 = float(row.get("cp0_eur", 0.0))
        cp1 = float(row.get("cp1_eur_per_mw", 0.0))
        cp2 = float(row.get("cp2_eur_per_mw2", 0.0))

        cost = cp2 * (p ** 2) + cp1 * p + cp0
        total += cost

        key = (et, element)
        if key not in per:
            per[key] = {"p_mw": p, "cp0": 0.0, "cp1": 0.0, "cp2": 0.0, "cost_eur": 0.0}
        per[key]["p_mw"] = p
        per[key]["cp0"] += cp0
        per[key]["cp1"] += cp1
        per[key]["cp2"] += cp2
        per[key]["cost_eur"] += cost

    return float(total), per


def main(cfg: Config):
    net = load_ieee_case(cfg.case_name)

    # apply edits
    scale_loads(net, cfg.load_scale)
    ensure_controllable(net)
    set_voltage_limits(net, *cfg.vm_bounds)
    scale_line_and_trafo_limits(net, cfg.line_limit_scale)

    meta = add_heterogeneous_quadratic_costs_and_bounds(net, cfg.gen_margin_mw, seed=cfg.seed)

    engine = run_dc_opf(net, verbose=cfg.verbose)
    min_gen_margin = check_generator_margins(net, cfg.gen_margin_mw)

    # ---- NEW: objective cost print ----
    pp_res_cost = getattr(net, "res_cost", None)
    try:
        pp_res_cost_val = float(pp_res_cost) if pp_res_cost is not None else None
    except Exception:
        pp_res_cost_val = None

    total_cost, per_cost = compute_active_poly_cost(net)

    print("\n--- Objective cost ---")
    if pp_res_cost_val is not None and np.isfinite(pp_res_cost_val):
        print(f"pandapower net.res_cost (if available): {pp_res_cost_val:.6f} EUR")
    print(f"Manual cost from net.poly_cost (active power terms): {total_cost:.6f} EUR")

    if per_cost:
        print("\n--- Cost breakdown by element ---")
        items = sorted(per_cost.items(), key=lambda kv: kv[1]["cost_eur"], reverse=True)
        for (et, idx), d in items:
            print(
                f"{et}[{idx}] : p={d['p_mw']:.4f} MW | "
                f"cp2={d['cp2']:.6g}, cp1={d['cp1']:.6g}, cp0={d['cp0']:.6g} | "
                f"cost={d['cost_eur']:.6f} EUR"
            )

    print("\n=== DC-OPF solved (single-config) ===")
    print(f"Engine: {engine}")
    print(f"Case: {cfg.case_name}")
    print(f"Total load (MW): {net.load.p_mw.sum():.3f}")
    print(f"Line limit scale: {cfg.line_limit_scale}   (<1 tightens, >1 loosens)")
    print(f"Min gen margin (MW): {min_gen_margin:.3f}")

    print("\n--- Costs (a_i) and expected uncongested dispatch p_hat ---")
    for (et, idx), info in meta.items():
        print(f"{et}[{idx}] : a={info['a']:.4f}   p_hat={info['p_hat']:.2f} MW")

    print("\n--- Actual dispatch (MW) ---")
    if len(net.res_ext_grid):
        print("ext_grid:")
        print(net.res_ext_grid[["p_mw"]])
    if len(net.res_gen):
        print("gen:")
        print(net.res_gen[["p_mw"]])

    if len(net.res_line) and "loading_percent" in net.res_line.columns:
        j = int(net.res_line.loading_percent.values.argmax())
        print("\n--- Most loaded line ---")
        print(net.res_line.loc[[j], ["p_from_mw", "p_to_mw", "loading_percent"]])

    # sanity check: show a few effective branch MW limits from the internal ppc if available
    if hasattr(net, "_ppc") and isinstance(net._ppc, dict) and "branch" in net._ppc:
        try:
            from pandapower.pypower.idx_brch import RATE_A
            rate_a = net._ppc["branch"][:, RATE_A]
            print("\n--- Internal branch RATE_A (MW) stats (after scaling) ---")
            print(f"min={np.nanmin(rate_a):.6g}, median={np.nanmedian(rate_a):.6g}, max={np.nanmax(rate_a):.6g}")
        except Exception:
            pass

    return net


if __name__ == "__main__":
    # Only one place to change settings: the USER SETTINGS block above.
    cfg = Config()
    main(cfg)
