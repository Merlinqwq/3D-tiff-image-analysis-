import unittest

import numpy as np
import pandas as pd
from scipy import stats

from intensity_data_analysis import calculate_statistics


class StatisticsTests(unittest.TestCase):
    def test_unpaired_equal_variance_t_test_uses_individual_nuclei(self):
        data = pd.DataFrame(
            {
                "acquisition_date": ["2026-01-01"] * 4 + ["2026-01-02"] * 4,
                "condition": ["normoxia"] * 2 + ["hypoxia"] * 2
                + ["normoxia"] * 2 + ["hypoxia"] * 2,
                "target_integrated_intensity": [1.0, 2.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0],
            }
        )
        summary, _ = calculate_statistics(data, ["target_integrated_intensity"])
        normoxia = np.array([1.0, 2.0, 2.0, 3.0])
        hypoxia = np.array([4.0, 5.0, 6.0, 7.0])
        expected = stats.ttest_ind(normoxia, hypoxia, equal_var=True)
        row = summary.iloc[0]
        self.assertEqual(row["statistical_unit"], "individual nucleus")
        self.assertEqual(row["test"], "unpaired two-sided t-test; equal variance")
        self.assertEqual(row["degrees_of_freedom"], 6)
        self.assertAlmostEqual(row["t_statistic"], expected.statistic)
        self.assertAlmostEqual(row["p_value"], expected.pvalue)


if __name__ == "__main__":
    unittest.main()
