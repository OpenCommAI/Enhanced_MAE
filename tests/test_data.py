from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from enhanced_mae.data import canonical_shape, to_channel_first, validate_dataset


class DataContractTests(unittest.TestCase):
    def test_supported_layouts(self) -> None:
        complex_array = np.zeros((2, 64, 256), dtype=np.complex64)
        channel_first = np.zeros((2, 2, 64, 256), dtype=np.float32)
        channel_last = np.zeros((2, 64, 256, 2), dtype=np.float32)
        expected = (2, 2, 64, 256)

        self.assertEqual(canonical_shape(complex_array, "complex"), expected)
        self.assertEqual(canonical_shape(channel_first, "first"), expected)
        self.assertEqual(canonical_shape(channel_last, "last"), expected)
        self.assertEqual(to_channel_first(channel_last).shape, expected)

    def test_complete_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "train_inputs.npy", np.zeros((3, 64, 256), np.complex64))
            np.save(root / "train_labels.npy", np.zeros((3, 2, 64, 32), np.float32))
            np.save(root / "test_inputs.npy", np.zeros((2, 64, 256, 2), np.float32))
            np.save(root / "test_labels.npy", np.zeros((2, 64, 32), np.complex64))

            report = validate_dataset(root)

        self.assertEqual(report.train_samples, 3)
        self.assertEqual(report.test_samples, 2)

    def test_mismatched_samples_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "train_inputs.npy", np.zeros((3, 64, 256), np.complex64))
            np.save(root / "train_labels.npy", np.zeros((2, 64, 32), np.complex64))
            np.save(root / "test_inputs.npy", np.zeros((2, 64, 256), np.complex64))
            np.save(root / "test_labels.npy", np.zeros((2, 64, 32), np.complex64))

            with self.assertRaisesRegex(ValueError, "input samples"):
                validate_dataset(root)


if __name__ == "__main__":
    unittest.main()
