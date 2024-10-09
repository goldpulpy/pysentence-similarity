"""Class to store embeddings in memory."""
import logging
from typing import List, Optional, Union

import h5py
import numpy as np

# Set up logging
logger = logging.getLogger("pysentence-similarity:storage")


class InvalidDataError(Exception):
    """Custom exception for invalid data."""
    pass


class Storage:
    """Storage class."""

    def __init__(
        self,
        sentences: Optional[List[str]] = None,
        embeddings: Optional[List[np.ndarray]] = None,
    ) -> None:
        """
        Initialize the storage class.

        :param sentences: List of sentences.
        :type sentences: List[str], optional
        :param embeddings: List of embeddings.
        :type embeddings: List[np.ndarray], optional
        :return: None
        """
        self._sentences = sentences if sentences is not None else []
        self._embeddings = embeddings if embeddings is not None else []

        if sentences is not None and embeddings is not None:
            self._validate_data()

    def save(self, filename: str) -> None:
        """
        Save the embeddings and sentences to a file.

        Save the embeddings and sentences to a file.

        :param filename: The name of the file to save the embeddings to.
        :type filename: str
        :raises OSError: If there is an error saving the embeddings.
        :raises InvalidDataError: If data is not valid for saving.
        :return: None
        """
        self._validate_data()
        try:
            with h5py.File(filename, 'w') as file:
                embeddings_array = np.stack(self._embeddings)
                file.create_dataset('embeddings', data=embeddings_array)
                dt = h5py.string_dtype(encoding='utf-8')
                file.create_dataset(
                    'sentences', data=self._sentences, dtype=dt)
                logger.info("Data saved to %s", filename)
        except OSError as err:
            logger.error("Error saving data: %s", err)
            raise

    @staticmethod
    def load(filename: str) -> "Storage":
        """
        Factory method to load the embeddings and sentences from a file and
        return a new Storage instance.

        :param filename: The name of the file to load the embeddings from.
        :type filename: str
        :return: A new instance of Storage class populated with the loaded 
        data.
        :rtype: Storage
        :raises OSError: If there is an error loading the embeddings.
        """
        try:
            with h5py.File(filename, 'r') as file:
                embeddings = [
                    np.array(embedding)
                    for embedding in file['embeddings'][:]
                ]
                sentences = [
                    sentence.decode('utf-8')
                    for sentence in file['sentences'][:]
                ]
                logger.info("Data loaded from %s", filename)
            return Storage(sentences=sentences, embeddings=embeddings)
        except OSError as err:
            logger.error("Error loading data: %s", err)
            raise

    def add(
        self,
        sentences: Union[str, List[str]],
        embeddings: Union[np.ndarray, List[np.ndarray]],
        save: bool = False,
        filename: str = None
    ) -> None:
        """
        Add a new sentences and embeddings to the storage.

        :param sentence: The sentence to add.
        :type sentence: Union[str, List[str]]
        :param embedding: The embedding to add.
        :type embedding: Union[np.ndarray, List[np.ndarray]]
        :param save: Whether to save the embeddings and sentences to a file.
        :type save: bool, optional
        :param filename: The name of the file to save the embeddings to.
        :type filename: str, optional
        :return: None
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        if isinstance(embeddings, np.ndarray):
            embeddings = [embeddings]

        if len(embeddings) != len(sentences):
            logger.error("Number of sentences and embeddings must match.")
            raise InvalidDataError(
                "Number of sentences and embeddings must be equal."
            )

        if save and filename is None:
            logger.error("Filename must be provided if save is True.")
            raise InvalidDataError(
                "Filename must be provided if save is True."
            )

        for sentence, embedding in zip(sentences, embeddings):
            self._sentences.append(sentence)
            self._embeddings.append(embedding)

        if save:
            self.save(filename)

    def remove_by_index(self, index: int) -> None:
        """
        Remove the sentence and embedding at the specified index.

        :param index: Index of the item to remove.
        :type index: int
        :raises IndexError: If the index is out of bounds.
        :return: None
        """
        try:
            removed_sentence = self._sentences.pop(index)
            self._embeddings.pop(index)
            logger.info("Removed sentence: %s", removed_sentence)
        except IndexError as err:
            logger.error("Index out of range: %s", err)
            raise

    def remove_by_sentence(self, sentence: str) -> None:
        """
        Remove the sentence and its corresponding embedding by sentence.

        :param sentence: The sentence to remove.
        :type sentence: str
        :raises ValueError: If the sentence is not found in the storage.
        :return: None
        """
        try:
            index = self._sentences.index(sentence)
            self.remove_by_index(index)
        except ValueError as err:
            logger.error("Sentence not found: %s", err)
            raise

    def get_sentences(self) -> List[str]:
        """
        Get the list of sentences.

        :return: The list of sentences.
        :rtype: List[str]
        """
        return self._sentences

    def get_embeddings(self) -> List[np.ndarray]:
        """
        Get the list of embeddings.

        :return: The list of embeddings.
        :rtype: List[np.ndarray]
        """
        return self._embeddings

    def _validate_data(self) -> None:
        """Validate data."""
        if isinstance(self._sentences, str):
            self._sentences = [self._sentences]

        if isinstance(self._embeddings, np.ndarray):
            self._embeddings = [self._embeddings]

        if not isinstance(self._sentences, list):
            logger.error("Sentences must be a list.")
            raise InvalidDataError("Sentences must be a list of strings.")

        if not all(isinstance(sentence, str) for sentence in self._sentences):
            logger.error("All sentences must be of type str.")
            raise InvalidDataError("All sentences must be of type str.")

        if not isinstance(self._embeddings, list):
            logger.error("Embeddings must be a list.")
            raise InvalidDataError(
                "Embeddings must be a list of numpy arrays.")

        if not all(
            isinstance(embedding, np.ndarray) for embedding in self._embeddings
        ):
            logger.error("All embeddings must be numpy arrays.")
            raise InvalidDataError("All embeddings must be numpy arrays.")

        if len(self._embeddings) != len(self._sentences):
            logger.error("Number of sentences and embeddings must match.")
            raise InvalidDataError(
                "Number of sentences and embeddings must be equal."
            )

    def __str__(self) -> str:
        """Return a string representation of the Model object."""
        return (f"Storage(sentences={len(self._sentences)}, "
                f"embeddings={len(self._embeddings)})")

    def __repr__(self) -> str:
        """Return a string representation of the Model object."""
        return self.__str__()

    def __copy__(self):
        """Create a shallow copy of the Model object."""
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.__dict__.update(self.__dict__)
        return new_instance

    def __len__(self) -> int:
        """Return the number of sentences."""
        return len(self._sentences)

    def __getitem__(self, index: int) -> List[Union[str, np.ndarray]]:
        """
        Get the sentence and embedding at the specified index.

        :param index: Index of the item to retrieve.
        :return: A list containing the sentence and its corresponding 
        embedding.
        :raises IndexError: If the index is out of bounds.
        """
        try:
            return [self._sentences[index], self._embeddings[index]]
        except IndexError as e:
            logger.error("Index out of range: %s", e)
            raise
