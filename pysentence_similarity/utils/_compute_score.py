"""Compute functions module."""
import logging
from typing import List, Union, Callable

import numpy as np
from tqdm import tqdm

from ..compute import cosine
from .._storage import Storage

# Set up logging
logger = logging.getLogger("pysentence-similarity:utils")


def _validate_inputs(
    source: Union[np.ndarray, List[np.ndarray]],
    embeddings: List[np.ndarray],
    rounding: int,
    progress_bar: bool
) -> None:
    """Validate the input parameters for the similarity_score function.

    :param source: Source embedding for comparison.
    :type source: Union[np.ndarray, List[np.ndarray]]
    :param embeddings: Embeddings to compare against.
    :type embeddings: List[np.ndarray]
    :param rounding: Number of decimal places to round the similarity scores.
    :type rounding: int
    :param progress_bar: Whether to show a progress bar.
    :type progress_bar: bool
    :raises ValueError: If inputs are not of the expected type or values.
    """
    if not isinstance(source, (np.ndarray, list)):
        logger.error("Source must be an np.ndarray or a list of np.ndarray.")
        raise ValueError(
            "Source must be an np.ndarray or a list of np.ndarray.")

    if not isinstance(embeddings, list) or not all(
        isinstance(e, np.ndarray) for e in embeddings
    ):
        logger.error("Embeddings must be a list of np.ndarray.")
        raise ValueError("Embeddings must be a list of np.ndarray.")

    if not isinstance(progress_bar, bool):
        logger.error("Progress bar must be a boolean.")
        raise ValueError("Progress bar must be a boolean.")

    if not isinstance(rounding, int) or not (0 <= rounding <= 10):
        logger.error("Rounding must be an integer between 0 and 10.")
        raise ValueError("Rounding must be an integer between 0 and 10.")

    if not embeddings:
        logger.error("Embeddings list cannot be empty.")
        raise ValueError("Embeddings list cannot be empty.")


def _compute_similarities(
    source: np.ndarray,
    embeddings: List[np.ndarray],
    compute_function: Callable,
    rounding: int,
    progress_bar: bool
) -> List[float]:
    """Compute similarity scores for a single source embedding.

    :param source: The source embedding.
    :type source: np.ndarray
    :param embeddings: List of embeddings to compare against.
    :type embeddings: List[np.ndarray]
    :param compute_function: Function to compute similarity scores.
    :type compute_function: Callable
    :param rounding: Number of decimal places to round the similarity scores.
    :type rounding: int
    :param progress_bar: Whether to show a progress bar.
    :type progress_bar: bool
    :return: List of similarity scores.
    :rtype: List[float]
    :raises ValueError: If inputs are not of the expected type.
    """
    try:
        similarities = [
            round(float(compute_function(source[0], embedding[0])), rounding)
            for embedding in tqdm(
                embeddings,
                desc="Computing similarity scores",
                disable=not progress_bar
            )
        ]
        return similarities
    except Exception as err:
        logger.error("Error computing similarity: %s", err)
        raise


def compute_score(
    source: Union[np.ndarray, List[np.ndarray]],
    embeddings: Union[np.ndarray, List[np.ndarray], Storage],
    compute_function: Callable = cosine,
    rounding: int = 2,
    progress_bar: bool = False
) -> List[float]:
    """Compute similarity scores between a source embedding and an array of 
    embeddings.

    This function calculates similarity scores between a given source 
    embedding (or a list of embeddings) and a set of embeddings using 
    a specified similarity computation function. It allows for 
    flexibility in the input types and provides options for rounding 
    the scores and displaying a progress bar.

    :param source: Source embedding for comparison.
    :type source: Union[np.ndarray, List[np.ndarray]]
    :param embeddings: Embeddings to compare against.
    :type embeddings: Union[np.ndarray, List[np.ndarray], Storage]
    :param compute_function: Function to compute similarity scores.
    :type compute_function: Callable
    :param rounding: Number of decimal places to round the similarity scores.
    :type rounding: int
    :param progress_bar: Whether to show a progress bar.
    :type progress_bar: bool
    :return: List of similarity scores.
    :rtype: List[float]
    :raises ValueError: If inputs are not of the expected type.
    """
    if isinstance(embeddings, np.ndarray):
        embeddings = [embeddings]

    if isinstance(embeddings, Storage):
        embeddings = embeddings.get_embeddings()

    _validate_inputs(source, embeddings if isinstance(
        embeddings, list
    ) else [embeddings], rounding, progress_bar)

    if isinstance(source, list):
        try:
            return [_compute_similarities(
                sentence,
                embeddings,
                compute_function,
                rounding,
                progress_bar
            ) for sentence in tqdm(
                source,
                desc="Source batch",
                disable=not progress_bar
            )]
        except Exception as err:
            logger.error(
                "Error computing similarities for source batch: %s", err)
            raise
    else:
        return _compute_similarities(
            source, embeddings,
            compute_function,
            rounding,
            progress_bar
        )
