"""DC-OPF problem adapter consumed by the *official* DC3 training engine.

Only the problem interface is different from the authors' ACOPF57 experiment.
The neural architecture, Adam training loop, soft loss, equality completion,
five differentiable correction steps, and test correction loop are imported
from ``vendor/DC3/method.py``.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn


PROJECT = pathlib.Path(__file__).resolve().parents[1]
LITERATURE_RUNNER = PROJECT / "literature_baseline_rebuild" / "run_literature_baselines.py"


def load_fair4net_helpers():
    """Load the already-audited dataset/operator implementation by file path."""
    spec = importlib.util.spec_from_file_location("fair4net_literature_runner", LITERATURE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Fair4NetDC3Data:
    """Expose Fair4Net DC-OPF through the interface expected by upstream DC3.

    A final metadata column stores the immutable dataset row number.  The
    network wrapper below removes that column before prediction; it exists only
    so completion and correction can retrieve each row's outage operator and
    generator limits without changing the authors' training-loop signature.
    """

    def __init__(self, prepared, helpers):
        self.P = prepared
        self.H = helpers
        self._device = prepared.device
        self._xdim = int(prepared.X.shape[1])
        self._ydim = prepared.ngen
        self._neq = 1
        self._nknowns = 0
        self._cost_scale = helpers.unsupervised_cost_scale(prepared).double()

        def rows(index: np.ndarray) -> torch.Tensor:
            ids = torch.as_tensor(index, dtype=torch.float64, device=self._device)[:, None]
            return torch.cat([prepared.X[index].double(), ids], dim=1)

        self._trainX = rows(prepared.idx["train"])
        self._validX = rows(prepared.idx["val"])
        self._testX = rows(prepared.idx["test"])

    @property
    def xdim(self): return self._xdim
    @property
    def ydim(self): return self._ydim
    @property
    def neq(self): return self._neq
    @property
    def nknowns(self): return self._nknowns
    @property
    def trainX(self): return self._trainX
    @property
    def validX(self): return self._validX
    @property
    def testX(self): return self._testX

    def _idx(self, X: torch.Tensor) -> torch.Tensor:
        return X[:, -1].round().long()

    def complete_partial(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """Complete one generator analytically to enforce power balance.

        ``Z`` contains exactly ``ngen - 1`` bounded generator fractions, just
        as upstream ACOPF predicts only the variables not fixed by equalities.
        The completion generator may vary when a contingency removes the
        preferred slack generator.
        """
        idx = self._idx(X)
        pmin = self.P.pmin[idx].double()
        pmax = self.P.pmax_eff[idx].double()
        complete = self.P.completion_gen[idx]
        rows = torch.arange(idx.numel(), device=X.device)
        free_mask = torch.ones_like(pmin, dtype=torch.bool)
        free_mask[rows, complete] = False
        Y = torch.zeros_like(pmin)
        Y[free_mask] = pmin[free_mask] + Z.reshape(-1) * (pmax[free_mask] - pmin[free_mask])
        demand = self.P.d_eff[idx].double().sum(1)
        Y[rows, complete] = demand - Y.sum(1)
        return Y

    def process_output(self, X: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("strict parity requires equality completion")

    def obj_fn(self, Y: torch.Tensor) -> torch.Tensor:
        # DC3 minimizes the physical generation objective without dispatch
        # labels.  Scaling is fixed and positive, so it does not change optima.
        return self.P.op.gen_cost(Y.float()).double() / self._cost_scale

    def eq_resid(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        idx = self._idx(X)
        demand = self.P.d_eff[idx].double().sum(1)
        return (Y.sum(1) - demand)[:, None]

    def ineq_dist(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        idx = self._idx(X)
        return self.H.inequality_violation(self.P, Y.float(), idx).double()

    def ineq_grad(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        outer_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            work = Y if outer_graph else Y.detach().requires_grad_(True)
            loss = self.ineq_dist(X, work).square().sum()
            grad = torch.autograd.grad(loss, work, create_graph=outer_graph)[0]
        return grad if outer_graph else grad.detach()

    def eq_grad(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        outer_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            work = Y if outer_graph else Y.detach().requires_grad_(True)
            loss = self.eq_resid(X, work).square().sum()
            grad = torch.autograd.grad(loss, work, create_graph=outer_graph)[0]
        return grad if outer_graph else grad.detach()

    def ineq_partial_grad(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Gradient projected onto the exact power-balance manifold.

        If a free generator changes by ``d``, the completed generator changes
        by ``-sum(d)``.  Therefore each free-variable derivative is its direct
        derivative minus the derivative of the completion generator.
        """
        idx = self._idx(X)
        direct = self.ineq_grad(X, Y)
        complete = self.P.completion_gen[idx]
        rows = torch.arange(idx.numel(), device=Y.device)
        completion_grad = direct[rows, complete][:, None]
        one_hot = torch.nn.functional.one_hot(complete, num_classes=Y.shape[1]).to(Y.dtype)
        free_grad = (direct - completion_grad) * (1.0 - one_hot)
        tangent = free_grad + one_hot * (-free_grad.sum(1, keepdim=True))
        return tangent


class Fair4NetNNSolver(nn.Module):
    """The authors' NNSolver architecture with metadata hidden from the net."""

    def __init__(self, data: Fair4NetDC3Data, args: dict):
        super().__init__()
        hidden = args["hiddenSize"]
        # This is the exact upstream sequence: Linear-BN-ReLU-Dropout twice,
        # Kaiming initialization, and an ngen-1 output completed analytically.
        self.net = nn.Sequential(
            nn.Linear(data.xdim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, data.ydim - data.neq),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
        self.data = data

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        features = X[:, :-1]
        bounded_partial = torch.sigmoid(self.net(features))
        return self.data.complete_partial(X, bounded_partial)
