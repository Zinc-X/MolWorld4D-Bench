import numpy as np
import unittest

from complexmd_eval.metrics import autocorrelation, safe_corr


class MetricTests(unittest.TestCase):
    def test_autocorrelation_starts_at_one(self) -> None:
        values = np.sin(np.linspace(0, 8, 100))
        acf = autocorrelation(values, 10)
        self.assertTrue(np.isclose(acf[0], 1.0))
        self.assertEqual(len(acf), 11)

    def test_safe_corr_constant_is_nan(self) -> None:
        self.assertTrue(np.isnan(safe_corr(np.ones(5), np.arange(5.0))))


if __name__ == "__main__":
    unittest.main()
