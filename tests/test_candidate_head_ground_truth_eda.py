import unittest

import torch

from eda.candidate_head_ground_truth_eda import WeightedCentroid


class CandidateHeadGroundTruthEDATests(unittest.TestCase):
    def test_streaming_centroid_transfers_separated_classes(self) -> None:
        model = WeightedCentroid(2, 2)
        train = torch.tensor([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        labels = torch.tensor([0, 0, 1, 1])
        model.update_train(train, labels, torch.ones(4))
        model.finish()
        model.update_holdout(train, labels, torch.ones(4))
        self.assertEqual(model.record()["holdout_accuracy"], 1.0)

    def test_weighted_accuracy_uses_supplied_pair_energy(self) -> None:
        model = WeightedCentroid(1, 2)
        model.update_train(
            torch.tensor([[-2.0], [-1.0], [1.0], [2.0]]),
            torch.tensor([0, 0, 1, 1]),
            torch.ones(4),
        )
        model.finish()
        model.update_holdout(
            torch.tensor([[-2.0], [2.0]]),
            torch.tensor([1, 1]),
            torch.tensor([0.1, 0.9]),
        )
        self.assertAlmostEqual(model.record()["holdout_accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()
