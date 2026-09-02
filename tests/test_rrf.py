from __future__ import annotations

"""Reciprocal Rank Fusion — детерминированный алгоритм, легко проверить.

Формула: score(doc) = Σ_lists weight_list × 1/(k + rank_in_list).
Инвариант: чем выше документ в каждом списке и чем больше вес его списка,
тем выше в результате.
"""

import unittest

from app.search.hybrid import reciprocal_rank_fusion


def _hit(doc_id: str) -> dict:
    return {"_id": doc_id, "_source": {}}


class RRFFusionTests(unittest.TestCase):

    def test_single_list_preserves_order(self):
        hits = [_hit("a"), _hit("b"), _hit("c")]
        fused = reciprocal_rank_fusion([hits])
        self.assertEqual([h["_id"] for h in fused], ["a", "b", "c"])

    def test_two_lists_boost_common_docs(self):
        """Документ, попавший в обе ветки, должен обойти документ,
        попавший в одну — при равных рангах."""
        list_a = [_hit("common"), _hit("only_a")]
        list_b = [_hit("common"), _hit("only_b")]
        fused = reciprocal_rank_fusion([list_a, list_b])
        self.assertEqual(fused[0]["_id"], "common")

    def test_weights_shift_ranking(self):
        """Ветвь с большим весом тянет свои документы наверх."""
        primary = [_hit("p1"), _hit("p2")]
        secondary = [_hit("s1"), _hit("s2")]
        # Без весов первое место может быть чьё угодно (по _id алфавитно
        # неявно). С весом 5:1 primary-документы должны быть выше.
        fused = reciprocal_rank_fusion(
            [primary, secondary], weights=[5.0, 1.0])
        top_two = {h["_id"] for h in fused[:2]}
        self.assertEqual(top_two, {"p1", "p2"})

    def test_rrf_score_attached(self):
        fused = reciprocal_rank_fusion([[_hit("x")]])
        self.assertIn("_rrf_score", fused[0])
        self.assertGreater(fused[0]["_rrf_score"], 0.0)

    def test_empty_lists(self):
        self.assertEqual(reciprocal_rank_fusion([]), [])
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])


if __name__ == "__main__":
    unittest.main()
