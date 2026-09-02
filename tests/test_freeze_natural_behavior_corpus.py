import unittest

from eda.freeze_natural_behavior_corpus import (
    _corpus_split,
    _counts,
    _source_counts,
)


class FreezeNaturalBehaviorCorpusTests(unittest.TestCase):
    def test_split_names_are_train_and_holdout(self) -> None:
        self.assertEqual(_corpus_split("discovery"), "train")
        self.assertEqual(_corpus_split("confirmation"), "holdout")

    def test_counts_keep_family_and_split_explicit(self) -> None:
        rows = [
            {"family": "word", "split": "train", "source_id": "a"},
            {"family": "word", "split": "train", "source_id": "a"},
            {"family": "word", "split": "holdout", "source_id": "b"},
        ]
        self.assertEqual(
            _counts(rows), {"word|holdout": 1, "word|train": 2}
        )
        self.assertEqual(
            _source_counts(rows), {"word|holdout": 1, "word|train": 1}
        )


if __name__ == "__main__":
    unittest.main()
