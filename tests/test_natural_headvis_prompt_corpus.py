import unittest

from eda.natural_headvis_prompt_corpus import (
    Candidate,
    _select_candidates,
    classify_domain,
    stable_split,
    strong_event,
)


class NaturalHeadVisCorpusTests(unittest.TestCase):
    def test_domain_rules_cover_distinct_natural_surfaces(self) -> None:
        self.assertEqual(classify_domain("def f(x):\n    return x\n" + "{};" * 4), "code")
        self.assertEqual(classify_domain("Human: hello\nAssistant: hi"), "dialogue")
        self.assertEqual(classify_domain("Élève français déjà arrivé." * 4), "multilingual")
        self.assertEqual(classify_domain("ordinary prose paragraph"), "prose")
        self.assertEqual(
            classify_domain("\n".join(f"field{i}: value" for i in range(10))),
            "structured",
        )

    def test_source_split_is_stable_and_binary(self) -> None:
        self.assertEqual(stable_split("42"), stable_split("42"))
        self.assertIn(stable_split("42"), {"train", "holdout"})

    def test_candidate_limits_apply_per_family_domain_split(self) -> None:
        rows = [
            Candidate(str(index), "prose", "train", "word_split", index + 2, ((1, 2),), "x")
            for index in range(8)
        ]
        rows += [
            Candidate(str(100 + index), "code", "holdout", "word_split", index + 2, ((1, 2),), "x")
            for index in range(6)
        ]
        selected = _select_candidates(rows, train_per_stratum=3, holdout_per_stratum=2)
        self.assertEqual(len(selected), 5)
        self.assertEqual(sum(row.split == "train" for row in selected), 3)
        self.assertEqual(sum(row.split == "holdout" for row in selected), 2)

    def test_strong_gate_requires_all_three_conditions(self) -> None:
        good = {
            "best_target_token_rank": 1,
            "target_attention_mass": 0.2,
            "target_pair_write_energy_fraction": 0.2,
        }
        self.assertTrue(strong_event(good))
        for key, value in (
            ("best_target_token_rank", 2),
            ("target_attention_mass", 0.01),
            ("target_pair_write_energy_fraction", 0.01),
        ):
            bad = dict(good)
            bad[key] = value
            self.assertFalse(strong_event(bad))


if __name__ == "__main__":
    unittest.main()
