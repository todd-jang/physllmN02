import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from norm_metrics import norm_change_metrics


class NormChangeMetricsTests(unittest.TestCase):
    def test_expansion_returns_relative_and_log_changes(self):
        metrics = norm_change_metrics(4.0, 6.0)

        self.assertAlmostEqual(metrics["output_input_norm_ratio"], 1.5)
        self.assertAlmostEqual(metrics["relative_norm_change"], 0.5)
        self.assertAlmostEqual(metrics["log_norm_ratio"], math.log(1.5))

    def test_contraction_returns_negative_relative_and_log_changes(self):
        metrics = norm_change_metrics(4.0, 2.0)

        self.assertAlmostEqual(metrics["relative_norm_change"], -0.5)
        self.assertAlmostEqual(metrics["log_norm_ratio"], math.log(0.5))

    def test_rejects_nonpositive_or_nonfinite_norms(self):
        for in_norm, out_norm in ((0.0, 1.0), (1.0, 0.0), (math.inf, 1.0), (1.0, math.nan)):
            with self.subTest(in_norm=in_norm, out_norm=out_norm):
                with self.assertRaises(ValueError):
                    norm_change_metrics(in_norm, out_norm)


if __name__ == "__main__":
    unittest.main()
