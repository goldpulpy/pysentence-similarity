"""Test sentence similarity module."""
import unittest
from unittest.mock import patch, mock_open, MagicMock

import numpy as np
from pysentence_similarity import Model


class TestModel(unittest.TestCase):
    """Test sentence similarity model."""
    @classmethod
    def setUpClass(cls) -> None:
        """Set up resources that are shared across tests."""
        cls.model = Model("all-MiniLM-L6-v2", dtype="fp16")

    def test_initialization(self) -> None:
        """Test that SentenceSimilarity is initialized properly."""
        self.assertIsInstance(self.model, Model)

    def test_encode(self) -> None:
        """Test single sentence embedding conversion."""
        sentence = "This is a test sentence."
        embedding = self.model.encode(sentence)

        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape[1], 384)

    def test_encode_invalid_input(self) -> None:
        """Test to_embedding raises error on invalid input."""
        with self.assertRaises(ValueError):
            self.model.encode(12345)

    def test_encode_sentences(self) -> None:
        """Test multiple sentence embedding conversion."""
        sentences = ["This is a test.", "Another sentence."]
        embeddings = self.model.encode(sentences)

        self.assertEqual(len(embeddings), 2)
        self.assertTrue(all(isinstance(emb, np.ndarray) for emb in embeddings))

    def test_encode_empty_input(self) -> None:
        """Test to_embeddings raises error on empty input."""
        with self.assertRaises(ValueError):
            self.model.encode([])

    def test_load_model_invalid_dtype(self) -> None:
        """Test that load_model raises error on invalid dtype."""
        self.model.dtype = "invalid_dtype"
        with self.assertRaises(ValueError):
            self.model._load_model()

    @patch('requests.get')
    def test_download_file_success(self, mock_get):
        """Test that download_file works as expected."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content = MagicMock(return_value=[b'data'])
        mock_response.headers = {'content-length': '4'}
        mock_get.return_value = mock_response

        with patch('builtins.open', mock_open()) as mock_file:
            self.model._download_file(
                "http://mock-url.com",
                "/mock/save/path",
                "Mock Description"
            )
            mock_file.assert_called_once_with("/mock/save/path", 'wb')

    @patch('requests.get')
    def test_download_file_fail(self, mock_get) -> None:
        """Тест: ошибка при загрузке файла."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with self.assertRaises(Exception):
            self.model._download_file(
                "http://mock-url.com",
                "/mock/save/path",
                "Mock Description"
            )

    def test_get_providers_cpu(self) -> None:
        """Test that get_providers returns CPUExecutionProvider."""
        self.model.device = 'cpu'
        self.assertEqual(self.model._get_providers(), ['CPUExecutionProvider'])

    def test_get_providers_cuda(self) -> None:
        """Test that get_providers returns CUDAExecutionProvider."""
        self.model.device = 'cuda'
        self.assertEqual(
            self.model._get_providers(),
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

    def test_get_providers_invalid(self) -> None:
        """Test that get_providers raises error on invalid device."""
        self.model.device = 'invalid'
        with self.assertRaises(ValueError):
            self.model._get_providers()


if __name__ == "__main__":
    unittest.main()
