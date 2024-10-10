"""Splitter module."""
import re
import csv
import json
import logging
from typing import List, Any, Union

import requests
from bs4 import BeautifulSoup

# Logging configuration
logger = logging.getLogger("pysentence-similarity:splitter")


class Splitter:
    """
    A class to split text into sentences.
    """

    def __init__(
        self,
        markers_to_split: Union[str, List[str]] = "\n",
        preserve_markers: bool = False,
    ) -> None:
        """
        Initializes the Splitter object, which is used to split a given text 
        based on specific characters or markers. This class allows flexible 
        splitting based on one or more characters and provides the option to 
        preserve these markers in the split result.

        :param markers_to_split: A string or list of characters (e.g., 
        punctuation marks) used to split the text. Default is a newline 
        character.
        :type markers_to_split: Union[str, List[str]]
        :param preserve_markers: A boolean indicating whether to include the 
        split markers in the resulting text. Default is False.
        :type preserve_markers: bool
        """
        if isinstance(markers_to_split, str):
            markers_to_split = [markers_to_split]

        if not isinstance(markers_to_split, list):
            logger.error("Split list must be a list or tuple.")
            raise ValueError("Split list must be a list or tuple.")

        self.markers_to_split = markers_to_split
        self.preserve_markers = preserve_markers
        logger.info("Splitter initialized.")

    def split_from_text(
        self,
        text: str,
    ) -> List[str]:
        """
        Splits the given text into sentences based on specified punctuation and 
        newlines.

        This method uses regular expressions to identify splitting points in 
        the input text. It can preserve split markers (such as punctuation) 
        based on the `preserve_markers` attribute set during initialization.


        :param text: The input text to split.
        :type text: str
        :return: A list of sentences.
        :rtype: List[str]
        """
        try:
            if not isinstance(text, str):
                raise TypeError("Expected a string as input.")

            if not text.strip():
                logger.warning("Empty string provided for splitting.")
                return []

            regex_pattern = '|'.join(map(re.escape, self.markers_to_split))
            if self.preserve_markers:
                parts = re.split(f'({regex_pattern})', text)
                sentences = [
                    ''.join(
                        part for part in parts[i:i + 2]
                    ).strip() for i in range(0, len(parts)-1, 2)
                ]
            else:
                sentences = re.split(regex_pattern, text)

            sentences = [
                sentence.strip() for sentence in sentences if sentence.strip()
            ]
            return sentences

        except Exception as err:
            logger.error(
                "An error occurred while splitting the text: %s", err
            )
            raise

    def split_from_file(
        self,
        file_path: str,
    ) -> List[str]:
        """
        Splits the contents of a text file into sentences based on specified 
        punctuation and newlines.

        This method reads the entire content of the specified text file and 
        utilizes the `split_from_text` method to split the content into 
        sentences. It expects the file to be encoded in UTF-8.

        :param file_path: The path to the file to split.
        :type file_path: str
        :return: A list of sentences.
        :rtype: List[str]
        """
        try:
            if not isinstance(file_path, str):
                raise TypeError("Expected a string as input.")

            if not file_path.strip():
                logger.warning("Empty string provided for splitting.")
                return []
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return self.split_from_text(text)

        except Exception as err:
            logger.error(
                "An error occurred while splitting the file: %s", err
            )
            raise

    def split_from_url(
        self,
        url: str,
        timeout: int = 10
    ) -> List[str]:
        """
        Fetches the content from a URL, removes HTML tags, and splits the
        cleaned text into sentences.

        This method retrieves the content from the provided URL, removes all
        HTML tags, and splits the remaining plain text into sentences based on 
        the specified split markers.

        :param url: The URL of the webpage to split.
        :type url: str
        :param timeout: The number of seconds to wait for the request to
        complete. Default is 10.
        :type timeout: int
        :return: A list of sentences.
        :rtype: List[str]
        """
        try:
            if not isinstance(url, str):
                raise TypeError("Expected a string as input.")

            if not isinstance(timeout, int):
                raise TypeError("Expected an integer as input.")

            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            # Use BeautifulSoup to clean the HTML
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text()
            return self.split_from_text(text)

        except requests.exceptions.RequestException as req_err:
            logger.error("An HTTP error occurred: %s", req_err)
            raise
        except Exception as err:
            logger.error("An error occurred while processing the URL: %s", err)
            raise

    def split_from_csv(
        self,
        file_path: str,
        column_names: List[str]
    ) -> List[str]:
        """
        Reads a CSV file and splits the text from specified columns into
        sentences.

        This method reads the contents of a CSV file, extracts text from the 
        specified columns, and then splits the text into sentences based on 
        the markers defined in the `Splitter` object. It can handle multiple 
        columns and combines the results into a single list of sentences.

        :param file_path: The path to the CSV file to read.
        :type file_path: str
        :param column_names: A list of column names to extract text from.
        :type column_names: List[str]
        :return: A list of sentences extracted from the specified columns.
        :rtype: List[str]
        """
        try:
            if not isinstance(file_path, str):
                raise TypeError("Expected a string as input for file_path.")

            if not isinstance(column_names, list) or not all(
                isinstance(col, str) for col in column_names
            ):
                raise TypeError(
                    "Expected a list of strings as input for column_names."
                )

            if not file_path.strip():
                logger.warning("Empty string provided for file_path.")
                return []

            sentences = []
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)

                if not reader.fieldnames:
                    logger.error("No fieldnames found in the CSV file.")
                    raise ValueError(
                        "The CSV file is empty or has no headers."
                    )

                missing_columns = [
                    col for col in column_names if col not in reader.fieldnames
                ]
                if missing_columns:
                    logger.error(
                        "Columns %s do not exist in the CSV file.",
                        missing_columns
                    )
                    raise ValueError(
                        f"Columns {missing_columns} do not exist."
                    )

                for row in reader:
                    for column_name in column_names:
                        text = row[column_name].strip()
                        if text:
                            sentences.extend(self.split_from_text(text))

            return sentences

        except FileNotFoundError:
            logger.error("The specified file was not found: %s", file_path)
            raise
        except ValueError as val_err:
            logger.error("Value error: %s", val_err)
            raise
        except Exception as err:
            logger.error(
                "An error occurred while processing the CSV file: %s", err
            )
            raise

    def split_from_json(self, file_path: str, keys: List[str]) -> List[str]:
        """
        Reads a JSON file and splits text from specified keys into sentences.

        This method processes a JSON file by extracting text values from 
        specified keys. The extracted text is then split into sentences based 
        on the markers defined in the `Splitter` object. It can handle nested 
        JSON structures and recursively extract values from deeply nested 
        objects.

        :param file_path: The path to the JSON file to read.
        :type file_path: str
        :param keys: A list of keys to extract text from.
        :type keys: List[str]
        :return: A list of sentences extracted from the specified keys.
        :rtype: List[str]
        """
        try:
            if not isinstance(file_path, str):
                raise TypeError("Expected a string as input for file_path.")

            if not isinstance(keys, list) or not all(
                isinstance(key, str) for key in keys
            ):
                raise TypeError(
                    "Expected a list of strings as input for keys.")

            if not file_path.strip():
                logger.warning("Empty string provided for file_path.")
                return []

            sentences = []
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                self._extract_json(data, keys, sentences)
            return sentences

        except FileNotFoundError:
            logger.error("The specified file was not found: %s", file_path)
            raise
        except json.JSONDecodeError:
            logger.error("Error decoding JSON from the file: %s", file_path)
            raise
        except Exception as err:
            logger.error(
                "An error occurred while processing the JSON file: %s", err)
            raise

    def _extract_json(
        self,
        data: Any,
        keys: List[str],
        sentences: List[str]
    ) -> None:
        """
        Recursively extracts sentences from a nested JSON structure.

        :param data: The JSON data to process (can be a dict, list, etc.).
        :type data: Any
        :param keys: A list of keys to extract text from.
        :type keys: List[str]
        :param sentences: A list to collect extracted sentences.
        :type sentences: List[str]
        """
        if isinstance(data, dict):
            for key in keys:
                if key in data:
                    text = data[key].strip()
                    if text:
                        sentences.extend(self.split_from_text(text))

            for value in data.values():
                self._extract_json(value, keys, sentences)

        elif isinstance(data, list):
            for item in data:
                self._extract_json(item, keys, sentences)

    def __str__(self) -> str:
        """
        Returns a string representation of the Splitter object.

        :return: A string representation of the Splitter object.
        :rtype: str
        """
        return (
            f"Splitter(markers_to_split={self.markers_to_split}, "
            f"preserve_markers={self.preserve_markers})"
        )

    def __repr__(self) -> str:
        """
        Returns a string representation of the Splitter object.

        :return: A string representation of the Splitter object.
        :rtype: str
        """
        return self.__str__()

    def __copy__(self):
        """Create a shallow copy of the SentenceSimilarity object."""
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.__dict__.update(self.__dict__)
        return new_instance
