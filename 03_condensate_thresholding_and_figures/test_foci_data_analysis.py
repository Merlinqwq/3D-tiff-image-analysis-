import unittest

import numpy as np
from scipy import stats

from foci_data_analysis import calculate_one, significance_label


class FociAnalysisTests(unittest.TestCase):
    def test_equal_variance_unpaired_t_test(self):
        values = {
            "normoxia": np.array([1.0, 2.0, 3.0, 4.0]),
            "hypoxia": np.array([2.0, 4.0, 6.0, 8.0]),
        }
        result = calculate_one("X", "focus_volume_um3", values, "individual focus")
        expected = stats.ttest_ind(values["normoxia"], values["hypoxia"], equal_var=True)
        self.assertAlmostEqual(result["t_statistic"], float(expected.statistic))
        self.assertAlmostEqual(result["p_value"], float(expected.pvalue))
        self.assertEqual(result["degrees_of_freedom"], 6)

    def test_significance_labels(self):
        self.assertEqual(significance_label(0.2), "ns")
        self.assertEqual(significance_label(0.04), "*")
        self.assertEqual(significance_label(0.004), "**")
        self.assertEqual(significance_label(0.0004), "***")
        self.assertEqual(significance_label(0.00004), "****")


if __name__ == "__main__":
    unittest.main()
