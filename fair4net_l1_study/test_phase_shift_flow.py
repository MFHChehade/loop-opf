#!/usr/bin/env python3
"""Focused regression tests for the phase-shift-aware DC flow contract."""
from __future__ import annotations

import unittest

import numpy as np
import torch

import deq_trainer_fair4net as trainer


def make_operator(flow_offset: torch.Tensor) -> trainer.PDHGOperatorLearnable:
    static = {
        "nbus": 2,
        "ngen": 2,
        "nbranch": 2,
        "bus_of_gen": torch.tensor([0, 1], dtype=torch.long),
        "PTDF": torch.tensor([[1.0, -1.0], [-0.25, 0.25]]),
        "gs_bus": torch.tensor([1.0, 0.5]),
        "flow_offset": flow_offset,
        "fmax": torch.tensor([7.0, 4.0]),
        "pmin": torch.zeros(2),
        "pmax": torch.full((2,), 20.0),
        "c2": torch.ones(2),
        "c1": torch.zeros(2),
    }
    steps = trainer.LearnableSteps(nK=2.0, s_max=1.0)
    return trainer.PDHGOperatorLearnable(static, steps)


class PhaseShiftFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.p = torch.tensor([[8.0, 4.0]])
        self.d = torch.tensor([[5.0, 7.0]])

    def test_complete_flows_and_violations_include_affine_offset(self) -> None:
        op = make_operator(torch.tensor([2.0, -3.0]))
        expected_flow = torch.tensor([[7.5, -4.375]])
        expected_viol = torch.tensor([[0.5, 0.375]])
        torch.testing.assert_close(op.branch_flows(self.p, self.d), expected_flow)
        torch.testing.assert_close(op.flow_violation_vec(self.p, self.d), expected_viol)

    def test_pdhg_rhs_has_correct_offset_signs(self) -> None:
        op = make_operator(torch.tensor([2.0, -3.0]))
        f_gen = op.ptdf_apply(op.gen_to_bus_inj(self.p))
        flow_term = torch.cat([f_gen, -f_gen], dim=1)
        residual = flow_term - op.flow_constraint_rhs(self.d)
        f = op.branch_flows(self.p, self.d)
        expected = torch.cat(
            [f - op.fmax[None, :], -f - op.fmax[None, :]], dim=1
        )
        torch.testing.assert_close(residual, expected)

    def test_zero_offset_preserves_previous_flow_formula(self) -> None:
        op = make_operator(torch.zeros(2))
        inj = op.gen_to_bus_inj(self.p) - op.effective_demand(self.d)
        torch.testing.assert_close(op.branch_flows(self.p, self.d), op.ptdf_apply(inj))

    def test_makebdc_call_signatures_produce_same_mw_offset(self) -> None:
        bus = np.zeros((2, 1))
        branch = np.zeros((1, 1))
        ptdf = np.asarray([[0.5, -0.5]], dtype=np.float64)

        def standalone(base_mva, _bus, _branch):
            self.assertEqual(base_mva, 100.0)
            return None, None, np.asarray([0.1, -0.1]), np.asarray([0.2])

        def pandapower(_bus, _branch):
            return None, None, np.asarray([0.1, -0.1]), np.asarray([0.2]), None

        expected = np.asarray([10.0], dtype=np.float32)
        np.testing.assert_allclose(
            trainer._compute_phase_shift_flow_offset(standalone, 100.0, bus, branch, ptdf),
            expected,
        )
        np.testing.assert_allclose(
            trainer._compute_phase_shift_flow_offset(pandapower, 100.0, bus, branch, ptdf),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
