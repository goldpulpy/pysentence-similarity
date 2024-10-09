"""Test cases for the compute functions module."""
import unittest
import numpy as np
from pysentence_similarity.compute import (
    cosine,
    euclidean,
    manhattan,
    jaccard,
    pearson,
    minkowski,
    hamming,
    kl_divergence,
    chebyshev,
    bregman
)


class TestСompute(unittest.TestCase):
    """Test cases for the embedding metrics."""

    def setUp(self) -> None:
        """Set up some common embeddings for testing."""
        self.embedding_1 = np.array([1, 2, 3])
        self.embedding_2 = np.array([4, 5, 6])
        self.embedding_3 = np.array([1, 0, 0])
        self.embedding_prob_1 = np.array([0.2, 0.5, 0.3])
        self.embedding_prob_2 = np.array([0.1, 0.7, 0.2])

    def test_cosine(self) -> None:
        """Test cosine similarity between two embedding vectors."""
        result = cosine(self.embedding_1, self.embedding_2)
        self.assertAlmostEqual(result, 0.974631846, places=6)

    def test_euclidean(self) -> None:
        """Test Euclidean distance between two embedding vectors."""
        result = euclidean(self.embedding_1, self.embedding_2)
        self.assertAlmostEqual(result, 5.196152422, places=6)

    def test_manhattan(self) -> None:
        """Test Manhattan distance between two embedding vectors."""
        result = manhattan(self.embedding_1, self.embedding_2)
        self.assertEqual(result, 9)

    def test_jaccard(self) -> None:
        """Test Jaccard similarity between two embedding vectors."""
        result = jaccard(self.embedding_1, self.embedding_2)
        self.assertAlmostEqual(result, 0.4, places=6)

    def test_pearson(self) -> None:
        """Test Pearson correlation between two embedding vectors."""
        result = pearson(self.embedding_1, self.embedding_2)
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_minkowski(self) -> None:
        """Test Minkowski distance between two embedding vectors."""
        result = minkowski(self.embedding_1, self.embedding_2, p=3)
        self.assertAlmostEqual(result, 4.32674871, places=6)

    def test_hamming(self) -> None:
        """Test Hamming distance between two embedding vectors."""
        result = hamming(self.embedding_1, self.embedding_2)
        self.assertEqual(result, 1.0)

    def test_kl_divergence(self) -> None:
        """Test KL divergence between two embedding vectors."""
        result = kl_divergence(self.embedding_prob_1, self.embedding_prob_2)
        self.assertAlmostEqual(result, 0.0920328, places=6)

    def test_chebyshev(self) -> None:
        """Test Chebyshev distance between two embedding vectors."""
        result = chebyshev(self.embedding_1, self.embedding_2)
        self.assertEqual(result, 3)

    def test_bregman(self) -> None:
        """Test Bregman divergence between two embedding vectors."""
        result = bregman(self.embedding_1, self.embedding_2)
        self.assertAlmostEqual(result, 27, places=6)


if __name__ == '__main__':
    unittest.main()
