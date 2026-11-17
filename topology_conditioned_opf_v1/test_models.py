"""Small checks that the topology input is used and gradients are finite."""

import torch

from models import AdmittanceConditionedNet, DynamicTopologyGNN, bounded_balance, line_status


def test_line_status():
    status = line_status(torch.tensor([-1, 1]), 3)
    assert status.tolist() == [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]


def test_dynamic_graph():
    model = DynamicTopologyGNN(
        4, torch.tensor([0, 3]), torch.tensor([0, 1, 2]),
        torch.tensor([1, 2, 3]), torch.tensor([1.0, 2.0, 3.0]),
        width=8, layers=2,
    )
    outage = torch.tensor([-1, 1])
    weights = model.adjacency_weights(outage)
    assert weights[0, 1] > 0 and weights[1, 1] == 0
    demand = torch.tensor([[0.5, 0.2, 0.1, 0.3]]).repeat(2, 1)
    capacity = torch.tensor([[1.0, 1.0]]).repeat(2, 1)
    output = model(demand, capacity, outage)
    assert output.shape == (2, 2) and torch.isfinite(output).all()
    output.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_dense_topology_input():
    model = AdmittanceConditionedNet(4, 2, torch.tensor([1.0, 2.0, 3.0]), width=16)
    x = torch.ones(2, 4)
    cap = torch.ones(2, 2)
    output = model(x, cap, torch.tensor([-1, 1]), torch.zeros(2, 3))
    assert output.shape == (2, 2) and torch.isfinite(output).all()
    assert not torch.equal(output[0], output[1])


def test_bounded_balance():
    raw = torch.tensor([[2.0, 3.0, 4.0]], requires_grad=True)
    demand = torch.tensor([[2.0, 2.0, 3.0]])
    lower = torch.zeros_like(raw)
    upper = torch.full_like(raw, 5.0)
    result = bounded_balance(raw, demand, lower, upper)
    assert abs(float(result.sum()) - 7.0) < 1e-5
    assert torch.all(result >= lower) and torch.all(result <= upper)
    gradient = torch.autograd.grad(result[0, 0], raw)[0][0]
    assert torch.allclose(gradient, torch.tensor([2 / 3, -1 / 3, -1 / 3]), atol=1e-4)


if __name__ == "__main__":
    test_line_status()
    test_dynamic_graph()
    test_dense_topology_input()
    test_bounded_balance()
    print("PASS: topology-conditioned model tests")
