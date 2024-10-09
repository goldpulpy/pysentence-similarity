"""Test cases for the Storage class."""
import unittest
import os
import numpy as np
from pysentence_similarity._storage import Storage, InvalidDataError


class TestStorage(unittest.TestCase):
    """Test cases for the Storage class."""

    def setUp(self) -> None:
        """Set up test data before each test."""
        self.sentences = [
            "This is a test sentence.",
            "This is another sentence.",
        ]
        self.embeddings = [np.random.rand(3), np.random.rand(3),]
        self.storage = Storage(
            sentences=self.sentences,
            embeddings=self.embeddings
        )
        self.test_filename = "test_storage.h5"

    def tearDown(self) -> None:
        """Remove test file after each test."""
        if os.path.exists(self.test_filename):
            os.remove(self.test_filename)

    def test_initialization_valid(self) -> None:
        """Test initialization with valid data."""
        self.assertEqual(len(self.storage), 2)
        self.assertEqual(self.storage._sentences, self.sentences)

    def test_initialization_invalid_sentences(self) -> None:
        """Test initialization with invalid sentences."""
        with self.assertRaises(InvalidDataError):
            Storage(sentences="Not a list", embeddings=self.embeddings)

    def test_initialization_invalid_embeddings(self) -> None:
        """Test initialization with invalid embeddings."""
        with self.assertRaises(InvalidDataError):
            Storage(sentences=self.sentences, embeddings="Not a list")

    def test_save_and_load(self) -> None:
        """Test saving and loading data."""
        self.storage.save(self.test_filename)
        loaded_storage = Storage.load(self.test_filename)
        self.assertEqual(len(loaded_storage), 2)
        self.assertEqual(loaded_storage._sentences, self.sentences)

    def test_save_invalid_data(self) -> None:
        """Test saving with invalid data."""
        self.storage._sentences.append(123)
        with self.assertRaises(InvalidDataError):
            self.storage.save(self.test_filename)

    def test_add_sentences_and_embeddings(self) -> None:
        """Test adding valid sentences and embeddings."""
        new_sentence = "This is a new sentence."
        new_embedding = np.random.rand(3)
        self.storage.add(new_sentence, new_embedding)

        self.assertEqual(len(self.storage), 3)
        self.assertEqual(self.storage._sentences[-1], new_sentence)

    def test_add_and_save(self) -> None:
        """Test adding and saving."""
        new_sentence = "This is another new sentence."
        new_embedding = np.random.rand(3)
        self.storage.add(
            new_sentence,
            new_embedding,
            save=True,
            filename=self.test_filename
        )

        loaded_storage = Storage.load(self.test_filename)
        self.assertEqual(len(loaded_storage), 3)
        self.assertEqual(loaded_storage._sentences[-1], new_sentence)

    def test_index_out_of_range(self) -> None:
        """Test index out of range."""
        with self.assertRaises(IndexError):
            self.storage[10]

    def test_remove_by_index_valid(self) -> None:
        """Test removing a sentence and embedding by valid index."""
        self.storage.remove_by_index(1)
        expected_sentences = ["This is a test sentence."]

        self.assertEqual(self.storage.get_sentences(), expected_sentences)

    def test_remove_by_index_out_of_range(self) -> None:
        """Test removing a sentence and embedding by out-of-range index."""
        with self.assertRaises(IndexError):
            self.storage.remove_by_index(5)

    def test_remove_by_index_boundary(self) -> None:
        """Test removing the first and last elements."""
        self.storage.remove_by_index(0)
        expected_sentences_after_first = ["This is another sentence."]
        self.assertEqual(
            self.storage.get_sentences(),
            expected_sentences_after_first
        )

    def test_remove_by_sentence_valid(self) -> None:
        """Test removing a sentence and embedding by valid sentence."""
        self.storage.remove_by_sentence("This is a test sentence.")
        expected_sentences = ["This is another sentence."]
        self.assertEqual(self.storage.get_sentences(), expected_sentences)

    def test_remove_by_sentence_not_found(self) -> None:
        """Test removing a sentence that does not exist."""
        with self.assertRaises(ValueError):
            self.storage.remove_by_sentence("non-existent sentence")


if __name__ == "__main__":
    unittest.main()
