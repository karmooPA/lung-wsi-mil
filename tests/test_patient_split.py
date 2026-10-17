"""Patient grouping regressions using a deterministic mixed-label cohort."""

import unittest

import pandas as pd

from src.data.splits import (
    REFERENCE_SEEDS,
    create_all_reference_splits,
    create_patient_aware_split,
    validate_patient_split,
)


class PatientSplitTests(unittest.TestCase):
    @staticmethod
    def metadata():
        return pd.DataFrame([
            {"slide_id": f"patient_{patient:02d}_{label}",
             "patient_id": f"patient_{patient:02d}", "label": label}
            for patient in range(40)
            for label in ("Normal", "Tumor")
        ])

    def test_reference_splits_are_reproducible_complete_and_patient_disjoint(self):
        metadata = self.metadata()
        first = create_all_reference_splits(metadata)
        second = create_all_reference_splits(metadata)
        self.assertEqual(list(first), REFERENCE_SEEDS)
        expected = set(metadata["slide_id"])
        for seed in REFERENCE_SEEDS:
            with self.subTest(seed=seed):
                train, test = first[seed]["train"], first[seed]["test"]
                self.assertTrue(set(train["patient_id"]).isdisjoint(test["patient_id"]))
                self.assertEqual(set(train["slide_id"]) | set(test["slide_id"]), expected)
                self.assertEqual(len(train) + len(test), len(metadata))
                self.assertEqual(set(train["label"]), {"Normal", "Tumor"})
                self.assertEqual(set(test["label"]), {"Normal", "Tumor"})
                pd.testing.assert_frame_equal(train, second[seed]["train"])
                pd.testing.assert_frame_equal(test, second[seed]["test"])

    def test_missing_required_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            create_patient_aware_split(self.metadata().drop(columns="patient_id"), 2)

    def test_patient_leakage_is_rejected_even_with_distinct_slides(self):
        metadata = self.metadata()
        with self.assertRaisesRegex(ValueError, "Patient leakage"):
            validate_patient_split(metadata, metadata.iloc[::2], metadata.iloc[1::2])

    def test_incomplete_coverage_is_rejected(self):
        metadata = self.metadata()
        train, test = create_patient_aware_split(metadata, 2)
        with self.assertRaisesRegex(ValueError, "cover the dataset"):
            validate_patient_split(metadata, train.iloc[1:], test)


if __name__ == "__main__":
    unittest.main()
