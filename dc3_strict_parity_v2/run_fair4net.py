#!/usr/bin/env python3
"""Run the official DC3 engine through the documented Fair4Net DC adapter."""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from common import UPSTREAM_COMMIT, import_upstream, seed_everything
from fair4net_adapter import Fair4NetDC3Data, Fair4NetNNSolver, load_fair4net_helpers


def train_validation_selected(method, data, config: dict, out: pathlib.Path):
    """Use DC3 formulas without opening held-out test rows during training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = Fair4NetNNSolver(data, config).to(device)
    optimizer = optim.Adam(solver.parameters(), lr=config["lr"])
    train_loader = DataLoader(TensorDataset(data.trainX), batch_size=config["batchSize"], shuffle=True)
    valid_loader = DataLoader(TensorDataset(data.validX), batch_size=len(data.validX))
    best_score, best_epoch = float("inf"), None
    history = []
    for epoch in range(int(config["epochs"])):
        validation = {}
        solver.eval()
        for (x_valid,) in valid_loader:
            method.eval_net(data, x_valid.to(device), solver, config, "valid", validation)
        score = float(np.mean(validation["valid_eval"]))
        if score < best_score:
            best_score, best_epoch = score, epoch + 1
            torch.save(solver.state_dict(), out / "solver_net.dict")

        solver.train()
        losses = []
        for (x_train,) in train_loader:
            x_train = x_train.to(device)
            optimizer.zero_grad()
            corrected = method.grad_steps(data, x_train, solver(x_train), config)
            loss = method.total_loss(data, x_train, corrected, config)
            loss.sum().backward()
            optimizer.step()
            losses.extend(loss.detach().cpu().numpy().tolist())
        history.append({"epoch": epoch + 1, "validation_eval": score, "train_loss": float(np.mean(losses))})
        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == config["epochs"]:
            print(f"[progress] dc3 epoch={epoch + 1}/{config['epochs']} validation_eval={score:.7g} best_epoch={best_epoch}", flush=True)

    solver.load_state_dict(torch.load(out / "solver_net.dict", map_location=device))
    selection = {
        "status": "PASS", "test_opened_during_training": False,
        "selection_metric": "official_DC3_valid_eval", "best_epoch": best_epoch,
        "best_validation_eval": best_score, "history": history,
    }
    (out / "validation_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return solver, selection


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--case", default="case300")
    ap.add_argument("--protocol", choices=["clean", "mixed"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--hidden-size", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    method, _, defaults_module = import_upstream()
    # baseline_harness_fair4net constructs its immutable network operator from
    # DEQ_CASE at import time. Set it before importing the shared helper so the
    # operator and dataset dimensions must describe the same network.
    os.environ["DEQ_CASE"] = args.case
    helpers = load_fair4net_helpers()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prepared = helpers.prepare(args.data, device, args.protocol)
    data = Fair4NetDC3Data(prepared, helpers)

    config = defaults_module.method_default_args("acopf57")
    # Keep the upstream ACOPF branch semantics (sigmoid partial variables and
    # completion enabled) while the adapter supplies DC equations.
    config["probType"] = "acopf57"
    config["epochs"] = args.epochs
    config["hiddenSize"] = args.hidden_size
    config["batchSize"] = args.batch_size
    config["lr"] = args.lr
    # The official training loop instantiates its module-level NNSolver.  This
    # single substitution changes only the AC problem interface to the DC one.
    method.NNSolver = Fair4NetNNSolver
    _, selection = train_validation_selected(method, data, config, args.out)
    payload = {
        "status": "PASS",
        "track": "fair4net_dc_via_official_dc3_engine",
        "upstream_commit": UPSTREAM_COMMIT,
        "dataset": str(args.data.resolve()),
        "protocol": args.protocol,
        "seed": args.seed,
        "epochs": args.epochs,
        "effective_config": {"hiddenSize": args.hidden_size, "batchSize": args.batch_size, "lr": args.lr},
        "unavoidable_changes": [
            "ACOPF57 problem object replaced by audited Fair4Net DC-OPF adapter",
            "AC nonlinear equality completion replaced by analytic DC power-balance completion",
            "AC inequalities replaced by scenario-specific DC line and generator limits",
            "sample row id carried as hidden metadata and never passed to the neural network",
        ],
        "unchanged": [
            "two 200-unit Linear/BatchNorm/ReLU/Dropout hidden layers",
            "Kaiming initialization and sigmoid partial-variable output",
            "Adam learning rate 1e-3, batch size 200, 1000 epochs by default",
            "objective plus weight-10 equality/inequality soft loss",
            "five differentiable partial-correction steps in training and testing",
            "correction learning rate 1e-4, tolerance 1e-4, momentum 0.5",
            "official total_loss, grad_steps, grad_steps_all and eval_net functions",
        ],
        "validation_selection": {
            "best_epoch": selection["best_epoch"],
            "best_validation_eval": selection["best_validation_eval"],
        },
        "test_opened_during_training": False,
        "intentional_protocol_change": (
            "The upstream train_net scheduler evaluates testX each epoch. This benchmark "
            "selects only by valid_eval, then opens clean test rows once in the posthoc audit."
        ),
    }
    (args.out / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
