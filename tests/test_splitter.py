"""Test cases for the Splitter class."""
import json
import unittest
from unittest import mock
from unittest.mock import patch, mock_open

import requests
from pysentence_similarity import Splitter


class TestSplitter(unittest.TestCase):
    """Test cases for the Splitter class."""

    def setUp(self) -> None:
        """Set up the test case by creating an instance of Splitter."""
        self.splitter = Splitter(
            markers_to_split=[".", "?", "!"],
            preserve_markers=True
        )

    def test_split_from_text(self) -> None:
        """Test splitting text into sentences."""
        text = "Hello world! How are you? I'm fine."
        expected = ["Hello world!", "How are you?", "I'm fine."]
        result = self.splitter.split_from_text(text)
        self.assertEqual(result, expected)

    def test_split_from_text_empty(self) -> None:
        """Test splitting an empty string."""
        text = ""
        result = self.splitter.split_from_text(text)
        self.assertEqual(result, [])

    def test_split_from_text_whitespace(self) -> None:
        """Test splitting a string with only whitespaces."""
        text = "    "
        result = self.splitter.split_from_text(text)
        self.assertEqual(result, [])

    def test_split_from_file(self) -> None:
        """Test splitting text from a file."""
        text = "Hello. This is a test file.\nHow are you?"
        expected = ["Hello.", "This is a test file.", "How are you?"]

        with patch("builtins.open", mock_open(read_data=text)) as mock_file:
            result = self.splitter.split_from_file("fake_file.txt")
            mock_file.assert_called_once_with(
                "fake_file.txt", "r", encoding="utf-8")
            self.assertEqual(result, expected)

    @patch("requests.get")
    def test_split_from_url(self, mock_get) -> None:
        """Test splitting text from a URL with a mocked HTTP request."""
        # Mock response content with HTML
        mock_html = (
            "<html><body>Hello world. How are you?<br>"
            "I'm fine.</body></html>"
        )
        mock_response = mock_get.return_value
        mock_response.content = mock_html.encode('utf-8')
        mock_response.raise_for_status = unittest.mock.Mock()

        expected = ["Hello world.", "How are you?", "I'm fine."]
        result = self.splitter.split_from_url("http://example.com")

        mock_get.assert_called_once_with("http://example.com", timeout=10)
        self.assertEqual(result, expected)

    @patch("requests.get")
    def test_split_from_url_http_error(self, mock_get) -> None:
        """Test handling an HTTP error in split_from_url."""
        mock_get.side_effect = requests.exceptions.HTTPError("404 Not Found")

        with self.assertRaises(requests.exceptions.HTTPError):
            self.splitter.split_from_url("http://example.com")

    def test_split_from_url_invalid_url(self) -> None:
        """Test handling invalid URL type in split_from_url."""
        with self.assertRaises(TypeError):
            self.splitter.split_from_url(12345)

    def test_split_from_url_invalid_timeout(self) -> None:
        """Test handling invalid timeout type in split_from_url."""
        with self.assertRaises(TypeError):
            self.splitter.split_from_url("http://example.com", timeout="five")

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='text1,text2\nHello World!,"This is a test."\n'
    )
    def test_split_from_csv_valid(self, mock_file) -> None:
        """Test valid CSV input with multiple columns."""
        result = self.splitter.split_from_csv(
            "fake_path.csv", ["text1", "text2"])
        expected_result = ["Hello World!", "This is a test."]
        self.assertEqual(result, expected_result)
        mock_file.assert_called_once_with(
            "fake_path.csv", 'r', encoding='utf-8')

    @patch("builtins.open", new_callable=mock_open)
    def test_split_from_csv_empty_file(self, mock_file) -> None:
        """Test handling of an empty CSV file."""
        mock_file.return_value.read.side_effect = ''
        with self.assertRaises(ValueError):
            self.splitter.split_from_csv("fake_path.csv", ["text1", "text2"])

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='text1,text2\nHello World!,"This is a test."\n'
    )
    def test_split_from_csv_missing_column(self, mock_file) -> None:
        """Test handling of a missing column in the CSV."""
        with self.assertRaises(ValueError):
            self.splitter.split_from_csv(
                "fake_path.csv", ["text1", "missing_column"])

    def test_split_from_csv_invalid_file_path(self) -> None:
        """Test handling of an invalid file path."""
        with self.assertRaises(FileNotFoundError):
            self.splitter.split_from_csv("invalid_path.csv", ["text1"])

    def test_split_from_csv_invalid_column_names(self) -> None:
        """Test handling of invalid column names argument."""
        with self.assertRaises(TypeError):
            self.splitter.split_from_csv("fake_path.csv", "text1")

    def test_split_from_csv_non_string_column_names(self) -> None:
        """Test handling of non-string column names."""
        with self.assertRaises(TypeError):
            self.splitter.split_from_csv("fake_path.csv", [123])

    @mock.patch(
        'builtins.open',
        new_callable=mock_open,
        read_data=json.dumps({
            "key1": "This is the first sentence. This is the second sentence.",
            "key2": "Another sentence here."
        }))
    def test_split_from_json_valid(self, mock_file) -> None:
        """Test splitting sentences from a valid JSON file."""
        result = self.splitter.split_from_json(
            "dummy_path.json", ["key1", "key2"])
        expected = [
            "This is the first sentence.",
            "This is the second sentence.",
            "Another sentence here."
        ]
        self.assertEqual(result, expected)

    @mock.patch('builtins.open', new_callable=mock_open, read_data=json.dumps({
        "key1": "",
        "key2": "   "
    }))
    def test_split_from_json_empty_keys(self, mock_file) -> None:
        """Test handling of empty strings in keys."""
        result = self.splitter.split_from_json(
            "dummy_path.json", ["key1", "key2"])
        self.assertEqual(result, [])

    @mock.patch('builtins.open', new_callable=mock_open, read_data=json.dumps({
        "key1": "This is a test sentence."
    }))
    def test_split_from_json_missing_key(self, mock_file) -> None:
        """Test handling of a missing key in the JSON."""
        result = self.splitter.split_from_json(
            "dummy_path.json", ["key1", "missing_key"])
        expected = ["This is a test sentence."]
        self.assertEqual(result, expected)

    @mock.patch('builtins.open', side_effect=FileNotFoundError)
    def test_split_from_json_file_not_found(self, mock_file) -> None:
        """Test handling of a file not found error."""
        with self.assertRaises(FileNotFoundError):
            self.splitter.split_from_json("dummy_path.json", ["key1"])

    @mock.patch(
        'builtins.open', new_callable=mock_open, read_data='not a json'
    )
    def test_split_from_json_json_decode_error(self, mock_file) -> None:
        """Test handling of JSON decode error."""
        with self.assertRaises(json.JSONDecodeError):
            self.splitter.split_from_json("dummy_path.json", ["key1"])

    def test_split_from_json_invalid_file_path_type(self) -> None:
        """Test passing an invalid file path type."""
        with self.assertRaises(TypeError):
            self.splitter.split_from_json(123, ["key1"])

    def test_split_from_json_invalid_keys_type(self) -> None:
        """Test passing an invalid keys type."""
        with self.assertRaises(TypeError):
            self.splitter.split_from_json("dummy_path.json", "key1")


if __name__ == "__main__":
    unittest.main()
