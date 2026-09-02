import unittest

from eda.headvis_natural_mechanism_audit import (
    _route_category,
    l1_rule_events,
    source_split,
)
from eda.natural_headvis_prompt_corpus import TokenizedSource


def source(tokens, ids=None):
    text = "".join(tokens)
    offsets = []
    cursor = 0
    for token in tokens:
        offsets.append((cursor, cursor + len(token)))
        cursor += len(token)
    return TokenizedSource(
        source_id="1",
        text=text,
        input_ids=tuple(range(len(tokens))) if ids is None else tuple(ids),
        offsets=tuple(offsets),
        tokens=tuple(tokens),
        domain="prose",
    )


class NaturalMechanismAuditTests(unittest.TestCase):
    def test_source_split_is_stable(self) -> None:
        self.assertEqual(source_split("42"), source_split("42"))
        self.assertIn(source_split("42"), {"discovery", "confirmation"})

    def test_route_taxonomy_identifies_strict_induction(self) -> None:
        item = source(("<bos>", " A", " B", " A", " B", " x"), (0, 1, 2, 1, 2, 3))
        self.assertEqual(_route_category(item, 3, 2), "strict_induction")

    def test_route_taxonomy_identifies_within_word(self) -> None:
        item = source(("<bos>", " electro", "magnet", "ism", " "))
        self.assertEqual(_route_category(item, 3, 1), "within_alpha_surface")

    def test_route_taxonomy_identifies_tokenization_tolerant_copy(self) -> None:
        item = source(("<bos>", " Or", " x", "or", "Or", " x"), (0, 1, 2, 3, 4, 2))
        self.assertEqual(_route_category(item, 3, 1), "tokenization_tolerant_copy")

    def test_route_taxonomy_keeps_bos_and_self_explicit(self) -> None:
        item = source(("<bos>", " a", " b"))
        self.assertEqual(_route_category(item, 1, 0), "bos")
        self.assertEqual(_route_category(item, 1, 1), "self")

    def test_newline_boundary_allows_multiple_prior_newlines(self) -> None:
        item = source(("<bos>", "\n", " first", "\n", " second", "\n", " end"))
        events = [event for event in l1_rule_events(item) if event.family == "newline_boundary"]
        final = next(event for event in events if event.query == 5)
        self.assertEqual(final.targets, ((1, 2), (3, 4)))


if __name__ == "__main__":
    unittest.main()
