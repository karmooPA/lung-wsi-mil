"""Five-fold coverage and leakage checks independent of real WSI files."""

import unittest
from collections import Counter

import pandas as pd

from lung_wsi_mil.data.cross_validation import (
    N_FOLDS,
    create_patient_aware_folds,
    validate_fold_split,
)
from lung_wsi_mil.data.splits import REFERENCE_SEEDS, create_patient_aware_split


class CrossValidationTests(unittest.TestCase):
    @staticmethod
    def metadata():
        return pd.DataFrame(
            [
                {
                    "slide_id": f"case_{patient:02d}_{label}",
                    "patient_id": f"case_{patient:02d}",
                    "label": label,
                }
                for patient in range(40)
                for label in ("Normal", "Tumor")
            ]
        )

    def test_every_slide_and_patient_is_validation_once_without_test_leakage(self):
        metadata = self.metadata()
        for seed in REFERENCE_SEEDS:
            with self.subTest(seed=seed):
                train, test = create_patient_aware_split(metadata, seed)
                folds = create_patient_aware_folds(train)
                repeated = create_patient_aware_folds(train)
                self.assertEqual([fold["fold"] for fold in folds], list(range(1, N_FOLDS + 1)))
                validation_slides, validation_patients = [], []
                for fold, same_fold in zip(folds, repeated):
                    fit, val = fold["train"], fold["val"]
                    self.assertTrue(set(fit["patient_id"]).isdisjoint(val["patient_id"]))
                    self.assertTrue(set(val["patient_id"]).isdisjoint(test["patient_id"]))
                    self.assertTrue(set(fit["patient_id"]).isdisjoint(test["patient_id"]))
                    self.assertEqual(
                        set(fit["slide_id"]) | set(val["slide_id"]), set(train["slide_id"])
                    )
                    self.assertEqual(len(fit) + len(val), len(train))
                    validation_slides.extend(val["slide_id"])
                    validation_patients.extend(val["patient_id"].unique())
                    pd.testing.assert_frame_equal(fit, same_fold["train"])
                    pd.testing.assert_frame_equal(val, same_fold["val"])
                self.assertEqual(Counter(validation_slides), Counter(train["slide_id"]))
                self.assertEqual(
                    Counter(validation_patients), Counter(train["patient_id"].unique())
                )

    def test_patient_overlap_is_rejected(self):
        metadata = self.metadata()
        with self.assertRaisesRegex(ValueError, "Patient leakage"):
            validate_fold_split(metadata, metadata.iloc[::2], metadata.iloc[1::2])

    def test_incomplete_fold_coverage_is_rejected(self):
        metadata = self.metadata()
        fold = create_patient_aware_folds(metadata)[0]
        with self.assertRaisesRegex(ValueError, "cover the full"):
            validate_fold_split(metadata, fold["train"].iloc[1:], fold["val"])


if __name__ == "__main__":
    unittest.main()
