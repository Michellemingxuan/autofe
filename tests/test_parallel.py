import unittest

from validation.parallel import parallel_map


def _double(value):
    return value * 2


class ParallelMapProgressTests(unittest.TestCase):
    def test_sequential_progress_logs_each_completed_item(self):
        with self.assertLogs("validation.parallel", level="INFO") as captured:
            results = parallel_map(
                _double, [1, 2, 3], n_jobs=1, desc="variant", log_progress=True
            )

        self.assertEqual(results, [2, 4, 6])
        self.assertIn("variant: 1/3 completed", captured.output[0])
        self.assertIn("variant: 3/3 completed", captured.output[-1])

    def test_parallel_progress_preserves_input_order(self):
        with self.assertLogs("validation.parallel", level="INFO") as captured:
            results = parallel_map(
                _double, [3, 1, 2], n_jobs=2, backend="threading",
                desc="variant", log_progress=True,
            )

        self.assertEqual(results, [6, 2, 4])
        progress = [line for line in captured.output if "completed" in line]
        self.assertEqual(len(progress), 3)
        self.assertIn("variant: 3/3 completed", progress[-1])


if __name__ == "__main__":
    unittest.main()