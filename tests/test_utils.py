"""Test sentence similarity module."""
import unittest

import numpy as np
from pysentence_similarity import Model, Storage
from pysentence_similarity.utils import compute_score, search_similar


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

    def test_search_similar(self) -> None:
        """Test search_similar returns the correct similar sentences."""
        query_embedding = self.model.encode("This is a test.")
        sentences = [
            "This is another test.",
            "This is a test.",
            "This is yet another test."
        ]
        embeddings = self.model.encode(sentences)

        results = search_similar(
            query_embedding=query_embedding,
            sentences=sentences,
            embeddings=embeddings,
            top_k=1
        )

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], ("This is a test.", 1.0))

    def test_search_similar_no_sentences(self) -> None:
        """Test search_similar with no sentences raises error."""
        query_embedding = self.model.encode("This is a test.")
        embeddings = np.zeros((0, 0))
        with self.assertRaises(ValueError) as context:
            search_similar(
                query_embedding=query_embedding,
                sentences=[],
                embeddings=embeddings
            )
        self.assertEqual(str(context.exception), "No sentences provided.")

    def test_search_similar_empty_embeddings(self) -> None:
        """Test search_similar with no sentences raises error."""
        query_embedding = self.model.encode("This is a test.")
        with self.assertRaises(ValueError) as context:
            search_similar(
                query_embedding=query_embedding,
                sentences=["This is a test."],
                embeddings=[]
            )
        self.assertEqual(str(context.exception), "No embeddings provided.")

    def test_search_similar_empty_storage(self) -> None:
        """Test search_similar with empty storage raises error."""
        query_embedding = self.model.encode("This is a test.")
        empty_storage = Storage()
        with self.assertRaises(ValueError):
            search_similar(
                query_embedding=query_embedding,
                storage=empty_storage
            )


if __name__ == "__main__":
    unittest.main()
