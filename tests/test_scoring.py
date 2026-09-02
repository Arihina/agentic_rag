from __future__ import annotations

"""cosine ↔ _score. Формула прямой копии из ingestion; тест защищает от
случайного расхождения формул на двух сторонах — что означало бы, что
пользовательский score_threshold значит разное в agentic_rag и в
самопроверке ingestion."""

import unittest

from app.search.scoring import cosine_from_score, score_from_cosine


class ScoringFormulaTests(unittest.TestCase):

    def test_perfect_cosine_gives_score_one(self):
        """Один и тот же вектор сам себе даёт cos=1 → _score=1."""
        self.assertAlmostEqual(score_from_cosine(1.0), 1.0)

    def test_orthogonal_gives_half(self):
        self.assertAlmostEqual(score_from_cosine(0.0), 0.5)

    def test_anti_parallel_gives_third(self):
        self.assertAlmostEqual(score_from_cosine(-1.0), 1.0 / 3.0)

    def test_round_trip_identity(self):
        """cosine → score → cosine должно давать исходное значение."""
        for cos in [-0.5, -0.1, 0.0, 0.25, 0.5, 0.75, 0.9, 0.99]:
            with self.subTest(cos=cos):
                self.assertAlmostEqual(
                    cosine_from_score(score_from_cosine(cos)), cos, places=10)

    def test_monotonic_in_cosine(self):
        """Больший косинус — больший score. Иначе min_score-фильтр
        перестанет вести себя как «отсеять всё ниже такого-то cos»."""
        cosines = [-0.5, 0.0, 0.5, 0.99]
        scores = [score_from_cosine(c) for c in cosines]
        self.assertEqual(scores, sorted(scores))


if __name__ == "__main__":
    unittest.main()
