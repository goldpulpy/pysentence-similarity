"""Test sentence similarity module."""
import unittest

import numpy as np
from pysentence_similarity import Model, compute_score


class TestModel(unittest.TestCase):
    """Test sentence similarity model."""
    @classmethod
    def setUpClass(cls) -> None:
        """Set up resources that are shared across tests."""
        cls.model = Model("all-MiniLM-L6-v2", dtype="fp16")

    def test_similarity_score(self) -> None:
        """Test similarity score between sentences."""
        source_embedding = self.model.encode("This is a test.")
        target_embedding = self.model.encode("This is another test.")

        score = compute_score(source_embedding, [target_embedding])

        self.assertIsInstance(score, list)
        self.assertEqual(len(score), 1)
        self.assertGreaterEqual(score[0], -1)
        self.assertLessEqual(score[0], 1)

    def test_similarity_score_invalid_input(self) -> None:
        """Test similarity score raises error on invalid inputs."""
        with self.assertRaises(ValueError):
            compute_score(123, [np.array([0.5, 0.1])])

    def test_similarity_score_rounding(self) -> None:
        """Test similarity score with different rounding values."""
        source_embedding = self.model.encode("This is a test.")
        target_embedding = self.model.encode("This is another test.")

        for rounding in range(0, 11):
            score = compute_score(
                source_embedding, [target_embedding],
                rounding=rounding
            )
            self.assertIsInstance(score, list)
            self.assertEqual(len(score), 1)

    def test_similarity_score_multiple_embeddings(self) -> None:
        """Test similarity score with multiple embeddings."""
        embeddings = self.model.encode(
            ["This is a test.",
             "This is another test."]
        )

        score = compute_score(embeddings, embeddings)

        self.assertIsInstance(score, list)
        self.assertEqual(len(score), 2)
        self.assertGreaterEqual(score[0][0], -1)
        self.assertLessEqual(score[0][0], 1)


if __name__ == "__main__":
    unittest.main()
