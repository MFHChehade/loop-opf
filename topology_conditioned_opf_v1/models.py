"""Topology-conditioned neural dispatch models for scenario-wise DC-OPF.

These are clean-room DC adaptations of two published ideas:
DeepOPF-FT's load-plus-admittance input and Falconer and Mones' graph-per-sample
OPF regression. They are not reproductions of the authors' AC experiments.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def line_status(outage: Tensor, nbranch: int) -> Tensor:
    """Return one base-ordered status vector per sample, with -1 meaning intact."""
    if outage.ndim != 1 or bool(((outage < -1) | (outage >= nbranch)).any()):
        raise ValueError("invalid line-outage indices")
    status = torch.ones((len(outage), nbranch), device=outage.device)
    active = outage >= 0
    rows = torch.arange(len(outage), device=outage.device)[active]
    status[rows, outage[active]] = 0.0
    return status


def bounded_balance(raw: Tensor, demand: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """Bounded, balanced dispatch with a differentiable active-set gradient."""
    if raw.shape != lower.shape or raw.shape != upper.shape:
        raise ValueError("dispatch and scenario bounds must have the same shape")
    total = demand.sum(dim=1, keepdim=True)
    if bool((total < lower.sum(1, keepdim=True) - 1e-3).any()) or bool(
        (total > upper.sum(1, keepdim=True) + 1e-3).any()
    ):
        raise ValueError("scenario aggregate generator bounds exclude demand")
    left = (lower - raw).amin(dim=1, keepdim=True)
    right = (upper - raw).amax(dim=1, keepdim=True)
    with torch.no_grad():
        for _ in range(48):
            mid = 0.5 * (left + right)
            produced = (raw + mid).clamp(min=lower, max=upper).sum(dim=1, keepdim=True)
            left = torch.where(produced < total, mid, left)
            right = torch.where(produced >= total, mid, right)
    lam = 0.5 * (left + right)
    candidate = (raw + lam).clamp(min=lower, max=upper)
    free = ((candidate > lower + 1e-5) & (candidate < upper - 1e-5)).to(raw.dtype)
    free = free.detach()
    correction = (total - candidate.sum(dim=1, keepdim=True)) / free.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)
    return candidate + free * correction


class AdmittanceConditionedNet(nn.Module):
    """Dense model with active admittances and outage-response features."""

    def __init__(self, nbus: int, ngen: int, susceptance: Tensor,
                 width: int = 256):
        super().__init__()
        self.nbranch = len(susceptance)
        strength = susceptance.float()
        scale = strength.abs().median().clamp_min(1e-9)
        self.register_buffer("base_admittance", (strength / scale).clamp(-20.0, 20.0))
        self.net = nn.Sequential(
            nn.Linear(nbus + ngen + 2 * self.nbranch, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width // 2), nn.ReLU(),
            nn.Linear(width // 2, ngen), nn.Sigmoid(),
        )

    def forward(self, demand: Tensor, capacity: Tensor, outage: Tensor,
                lodf_feature: Tensor) -> Tensor:
        if lodf_feature.shape != (len(outage), self.nbranch):
            raise ValueError("LODF feature shape must match the batch and branch count")
        active_admittance = line_status(outage, self.nbranch) * self.base_admittance[None, :]
        x = torch.cat((demand, capacity, active_admittance, lodf_feature), dim=1)
        return self.net(x)


class DynamicTopologyGNN(nn.Module):
    """Graph-per-sample message passing with globally informed generator heads."""

    def __init__(self, nbus: int, bus_of_gen: Tensor, branch_from: Tensor,
                 branch_to: Tensor, susceptance: Tensor, width: int = 64,
                 layers: int = 3):
        super().__init__()
        if len(branch_from) != len(branch_to) or len(branch_from) != len(susceptance):
            raise ValueError("branch arrays must align in base internal order")
        self.nbus = nbus
        self.nbranch = len(branch_from)
        self.register_buffer("bus_of_gen", bus_of_gen.long())
        self.register_buffer("branch_from", branch_from.long())
        self.register_buffer("branch_to", branch_to.long())
        strength = susceptance.abs().float().clamp_min(1e-9)
        strength = (strength / strength.median()).clamp(0.05, 20.0)
        self.register_buffer("strength", strength)
        self.encoder = nn.Linear(2, width)
        self.convolutions = nn.ModuleList(nn.Linear(2 * width, width) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(2 * width + 2, width), nn.ReLU(),
                                  nn.Linear(width, 1), nn.Sigmoid())

    def adjacency_weights(self, outage: Tensor) -> Tensor:
        return line_status(outage, self.nbranch) * self.strength[None, :]

    def _messages(self, h: Tensor, weights: Tensor) -> Tensor:
        f, t = self.branch_from, self.branch_to
        batch, nbus, hidden = h.shape
        degree = h.new_ones((batch, nbus))
        degree.index_add_(1, f, weights)
        degree.index_add_(1, t, weights)
        normalized = weights * degree[:, f].rsqrt() * degree[:, t].rsqrt()
        result = h.clone() / degree[:, :, None]
        result.index_add_(1, f, h[:, t, :] * normalized[:, :, None])
        result.index_add_(1, t, h[:, f, :] * normalized[:, :, None])
        return result

    def forward(self, demand: Tensor, capacity: Tensor, outage: Tensor) -> Tensor:
        if demand.shape[1] != self.nbus or capacity.shape[1] != len(self.bus_of_gen):
            raise ValueError("node or generator dimension mismatch")
        bus_capacity = demand.new_zeros(demand.shape)
        bus_capacity.index_add_(1, self.bus_of_gen, capacity)
        node = torch.stack((demand, bus_capacity), dim=-1)
        h = F.relu(self.encoder(node))
        weights = self.adjacency_weights(outage)
        for layer in self.convolutions:
            h = F.relu(layer(torch.cat((h, self._messages(h, weights)), dim=-1)))
        local = h[:, self.bus_of_gen, :]
        global_context = h.mean(dim=1, keepdim=True).expand(-1, local.shape[1], -1)
        gen_features = torch.stack((capacity, (capacity > 0).float()), dim=-1)
        return self.head(torch.cat((local, global_context, gen_features), dim=-1)).squeeze(-1)
