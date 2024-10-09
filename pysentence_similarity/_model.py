"""Model obj module."""

import os
import time
import logging
from typing import List, Union, Callable

import onnxruntime as ort
import numpy as np
import requests
from tokenizers import Tokenizer
from tqdm import tqdm
from platformdirs import user_cache_dir

from .pooling import mean_pooling
from ._support_models import _support_models
from ._version import __title__

# Set up logging
logger = logging.getLogger("pysentence-similarity:model")


class Model:
    """Class for computing sentence similarity model."""

    _repo_url = (
        "https://huggingface.co/goldpulpy/pysentence-similarity/resolve/main/"
    )

    def __init__(
        self,
        model: str,
        dtype: str = 'fp32',
        cache_dir: str = None,
        device: str = 'cpu'
    ) -> None:
        """
        Initialize the sentence similarity task model.

        :param model: The name of the model to be used.
        :type model: str
        :param dtype: The dtype of the model ('fp32', 'fp16', 'int8').
        :type dtype: str
        :param cache_dir: Directory to cache the model and tokenizer.
        :type cache_dir: str
        :param device: Device to use for inference ('cuda', 'cpu').
        :type device: str
        :raises ValueError: If the model or tokenizer cannot be loaded.
        """
        self.model = model
        self.dtype = dtype.lower()
        self.cache_dir = cache_dir or user_cache_dir(__title__)
        self._model_dir = os.path.join(self.cache_dir, self.model)
        os.makedirs(self._model_dir, exist_ok=True)
        self.device = device

        try:
            self._providers = self._get_providers()
            self._tokenizer = self._load_tokenizer()
            self._session = self._load_model()
        except Exception as err:
            logger.error("Error initializing model: %s", err)
            raise

    @staticmethod
    def get_supported_models() -> List[str]:
        """Get a list of supported models.

        :return: List of supported models.
        :rtype: List[str]
        """
        return _support_models

    def encode(
        self,
        sentences: Union[str, List[str]],
        pooling_function: Callable = mean_pooling,
        progress_bar: bool = False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Convert a single sentence to an embedding vector.

        :param sentences: Sentence or list of sentences to convert.
        :type sentences: Union[str, List[str]]
        :param pooling_function: Function to pool the sentence embeddings, 
        defaults to mean,
        :type pooling_function: Callable | None
        :param progress_bar: Whether to show a progress bar.
        :type progress_bar: bool
        :return: Embedding vector for the sentence.
        :rtype: Union[np.ndarray, List[np.ndarray]]
        """
        if not isinstance(sentences, (str, list)):
            raise ValueError(
                "Input must be a string or a list representing a sentence."
            )

        if not sentences:
            raise ValueError("Input cannot be an empty string or list.")

        if isinstance(sentences, list):
            return self._encode_sentences(
                sentences,
                pooling_function=pooling_function,
                progress_bar=progress_bar
            )

        try:
            encoded_input = self._tokenizer.encode(sentences)
            inputs = {
                'input_ids': [encoded_input.ids],
                'attention_mask': [encoded_input.attention_mask]
            }
            output = self._session.run(None, inputs)
            sentence_embedding = pooling_function(
                output[0],
                [encoded_input.attention_mask]
            )
            return sentence_embedding
        except Exception as err:
            logger.error(
                "Error getting embedding for sentence: %s, error: %s",
                sentences,
                err
            )
            raise

    def _get_providers(self) -> List[str]:
        """Get the list of providers to use for inference.

        :return: List of providers to use for inference.
        :rtype: List[str]
        :raises ValueError: If the device is invalid.
        """
        provider_mapping = {
            'cpu': ['CPUExecutionProvider'],
            'cuda': ['CUDAExecutionProvider', 'CPUExecutionProvider']
        }

        providers = provider_mapping.get(self.device)

        if providers is None:
            raise ValueError(
                "Invalid device. Must be 'cpu' or 'cuda'."
            )

        return providers

    def _load_tokenizer(self) -> Tokenizer:
        """Load the tokenizer from cache or download it if not available.

        :raises FileNotFoundError: If the tokenizer cannot be found.
        :return: Loaded Tokenizer instance.
        :rtype: Tokenizer
        """
        tokenizer_path = os.path.join(self._model_dir, 'tokenizer.json')

        if os.path.isfile(tokenizer_path):
            return Tokenizer.from_file(tokenizer_path)

        tokenizer_url = f"{self._repo_url}{self.model}/tokenizer.json"

        try:
            self._download_file(
                tokenizer_url,
                tokenizer_path,
                f"Downloading tokenizer for {self.model}"
            )
            return Tokenizer.from_file(tokenizer_path)
        except Exception as err:
            logger.error("Error loading tokenizer: %s", err)
            raise FileNotFoundError("Tokenizer file not found in repo.")

    def _load_model(self) -> ort.InferenceSession:
        """Load the model from cache or download it if not available.

        :raises FileNotFoundError: If the model cannot be found.
        :return: Loaded ONNX InferenceSession.
        :rtype: ort.InferenceSession
        """

        if self.model not in _support_models:
            raise ValueError(
                f"Model '{self.model}' not supported. Must be one of "
                f"{_support_models}."
            )

        valid_dtypes = {'fp32', 'fp16', 'int8'}
        if self.dtype not in valid_dtypes:
            raise ValueError(
                f"Invalid dtype '{self.dtype}'. Must be one of "
                f"{valid_dtypes}."
            )

        model_path = os.path.join(
            self._model_dir, f"model_{self.dtype}.onnx"
        )

        # Check if model already exists
        if os.path.isfile(model_path):

            return ort.InferenceSession(
                model_path,
                providers=self._providers
            )

        model_url = (
            f"{self._repo_url}{self.model}/model_{self.dtype}.onnx"
        )

        try:
            self._download_file(
                model_url,
                model_path,
                f"Downloading model {self.model} ({self.dtype.upper()})"
            )
        except Exception as err:
            logger.error("Error loading model: %s", err)
            raise FileNotFoundError("Model file not found in repo.")
        try:
            return ort.InferenceSession(
                model_path,
                providers=self._providers
            )
        except Exception as err:
            logger.error("Error loading model: %s", err)
            raise

    def _download_file(self, url: str, save_path: str, desc: str) -> None:
        """Download a file from a URL with progress indication.

        :param url: URL of the file to download.
        :type url: str
        :param save_path: Path to save the downloaded file.
        :type save_path: str
        :param desc: Description of the download.
        :type desc: str
        :raises Exception: If there is an error during download.
        """
        response = requests.get(url, stream=True, timeout=30)
        total_size = int(response.headers.get('content-length', 0))
        logger.info("Starting download of %s (%d bytes)", url, total_size)

        if response.status_code != 200:
            logger.error("Failed to download file: %s", response.status_code)
            raise Exception("Failed to download file.")

        with open(save_path, 'wb') as file, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                progress_bar.update(len(data))

    def _encode_sentences(
        self,
        sentences: List[str],
        pooling_function: Callable,
        progress_bar: bool,
    ) -> List[np.ndarray]:
        """Convert a list of sentences to embedding vectors.

        :param sentences: List of sentences to convert.
        :type sentences: List[str]
        :param pooling_function: Function to pool the embedding vectors.
        :type pooling_function: Callable
        :param progress_bar: Whether to show a progress bar.
        :type progress_bar: bool
        :return: List of embedding vectors.
        :rtype: List[np.ndarray]
        """
        if not isinstance(sentences, list) or not all(
            isinstance(sentence, str) for sentence in sentences
        ):
            raise ValueError("Input must be a list of sentences.")

        if not isinstance(progress_bar, bool):
            raise ValueError("Progress bar must be a boolean.")

        if not sentences:
            raise ValueError("Input list cannot be empty.")

        try:
            start_time = time.time()
            sentence_embeddings = [
                self.encode(sentence, pooling_function) for sentence in tqdm(
                    sentences,
                    desc="Converting sentences to embeddings",
                    disable=not progress_bar
                )
            ]
            end_time = time.time()
            logger.info(
                "Took %s seconds to convert %s sentences to embeddings.",
                round(end_time - start_time, 2),
                len(sentences)
            )
        except Exception as err:
            logger.error(
                "Error converting sentence to embedding error: %s", err
            )
            raise
        return sentence_embeddings

    def __call__(self, *args, **kwargs) -> np.ndarray | List[np.ndarray]:
        """Call the Model object."""
        return self.encode(*args, **kwargs)

    def __str__(self) -> str:
        """Return a string representation of the Model object."""
        return (f"Model(model='{self.model}', "
                f"dtype='{self.dtype}', "
                f"cache_dir='{self.cache_dir}', "
                f"device='{self.device}')")

    def __repr__(self) -> str:
        """Return a string representation of the Model object."""
        return self.__str__()

    def __copy__(self):
        """Create a shallow copy of the Model object."""
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.__dict__.update(self.__dict__)
        return new_instance
