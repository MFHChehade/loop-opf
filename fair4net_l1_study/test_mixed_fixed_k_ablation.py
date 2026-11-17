import unittest
import torch

from mixed_fixed_k_ablation import aggregate_timing, build_budget_rows, rollout_fixed_k


class AddOne(torch.nn.Module):
    ngen = 1
    nbranch = 1
    def forward(self, z, d, gen, line):
        return z + 1


class Tests(unittest.TestCase):
    def test_rollout_is_exactly_k_calls(self):
        d = torch.zeros(2, 3)
        idx = torch.full((2,), -1, dtype=torch.long)
        z = rollout_fixed_k(AddOne(), d, idx, idx, 7)
        self.assertTrue(torch.equal(z, torch.full((2, 4), 7.0)))

    def test_aggregation_and_conservative_budget(self):
        rows = []
        for k, values in ((1, (1.0, 1.2, 1.1)), (2, (2.2, 2.0, 2.1))):
            for repeat, value in enumerate(values):
                rows.append({"case": "case300", "category": "pooled", "k": k,
                             "batch_size": 1, "timing_kind": "model_only",
                             "repeat": repeat, "latency_ms": value,
                             "peak_cuda_mem_alloc_bytes": 10})
        agg = aggregate_timing(rows)
        budget = build_budget_rows(agg, [1.5, 3.0])
        self.assertEqual([r["selected_k"] for r in budget], [1, 2])


if __name__ == "__main__":
    unittest.main()
