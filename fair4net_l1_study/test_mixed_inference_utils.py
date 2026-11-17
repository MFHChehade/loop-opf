import unittest

from mixed_inference_utils import aggregate_timing, category_name, parse_positive_ints


class MixedInferenceUtilsTests(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(category_name(-1, -1), "clean")
        self.assertEqual(category_name(4, -1), "line")
        self.assertEqual(category_name(-1, 7), "generator")
        with self.assertRaises(ValueError):
            category_name(4, 7)

    def test_parse_positive_ints(self):
        self.assertEqual(parse_positive_ints("32,1,8,8"), [1, 8, 32])
        with self.assertRaises(ValueError):
            parse_positive_ints("0,8")

    def test_aggregate(self):
        rows = [
            {"case": "case300", "method": "deq", "category": "clean", "batch_size": 2,
             "timing_kind": "model_only", "latency_ms": value, "peak_cuda_mem_alloc_bytes": 100 + i}
            for i, value in enumerate((2.0, 4.0, 6.0))
        ]
        out = aggregate_timing(rows)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["latency_p50_ms"], 4.0)
        self.assertAlmostEqual(out[0]["throughput_samples_per_sec"], 500.0)
        self.assertEqual(out[0]["peak_cuda_mem_alloc_bytes"], 102)


if __name__ == "__main__":
    unittest.main()
