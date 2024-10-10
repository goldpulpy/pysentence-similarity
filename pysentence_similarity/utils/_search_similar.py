"""Search similar module."""
import logging
from typing import List, Callable, Tuple, Optional
import numpy as np

from ..compute import cosine
from .._storage import Storage, InvalidDataError
from ._compute_score import compute_score

# Set up logging
logger = logging.getLogger("pysentence-similarity:utils")


def search_similar(
    query_embedding: np.ndarray,
    sentences: Optional[List[str]] = None,
    embeddings: Optional[List[np.ndarray]] = None,
    storage: Optional[Storage] = None,
    top_k: int = 5,
    compute_function: Callable = cosine,
    rounding: int = 2,
    progress_bar: bool = False,
    sort_order: str = 'desc'
) -> List[Tuple[str, float]]:
    """
    Search for similar sentences based on the provided query embedding.

    This function retrieves and computes similarity scores between a given
    query embedding and a set of candidate sentences (and their corresponding
    embeddings). It returns the top K most similar sentences based on the
    specified similarity metric.

    - If `storage` is provided, it will be used to retrieve both sentences and 
    embeddings, allowing the other parameters (`sentences` and `embeddings`) to 
    be omitted.
    - Similarity scores are calculated using the specified `compute_function`.
    - Results can be sorted in either ascending or descending order based 
    on the specified `sort_order`.

    :param query_embedding: The embedding of the query sentence.
    :type query_embedding: np.ndarray
    :param sentences: List of candidate sentences. Optional if `storage` is
    provided.
    :type sentences: Optional[List[str]]
    :param embeddings: List of embeddings corresponding to the sentences.
    Optional if `storage` is provided.
    :type embeddings: Optional[List[np.ndarray]]
    :param storage: An instance of `Storage` to retrieve sentences and 
    embeddings from.
    :type storage: Optional[Storage]
    :param top_k: The number of top similar sentences to return. Default is 5.
    :type top_k: int
    :param compute_function: Function used to compute similarity between 
    embeddings. Default is cosine similarity.
    :type compute_function: Callable
    :param rounding: The number of decimal places to round the similarity 
    scores to. Default is 2.
    :type rounding: int
    :param progress_bar: Whether to show a progress bar during score 
    computation. Default is False.
    :type progress_bar: bool
    :param sort_order: The order to sort results by similarity score. 'asc' 
    for ascending, 'desc' for descending. Default is 'desc'.
    :type sort_order: str
    :return: A list of tuples containing the top `top_k` similar sentences and 
    their similarity scores.
    :rtype: List[Tuple[str, float]]
    :raises ValueError: If both `sentences` and `embeddings` or `storage` are 
    not provided.
    :raises InvalidDataError: If there is an inconsistency in data (e.g., 
    different lengths of `sentences` and `embeddings`).
    """

    if storage is not None:
        try:
            sentences = storage.get_sentences()
            embeddings = storage.get_embeddings()
        except Exception as err:
            logger.error("Failed to retrieve data from storage: %s", err)
            raise InvalidDataError(
                "Failed to retrieve data from storage.") from err

    if not sentences:
        logger.error("No sentences provided.")
        raise ValueError("No sentences provided.")

    if not embeddings:
        logger.error("No embeddings provided.")
        raise ValueError("No embeddings provided.")

    if len(sentences) != len(embeddings):
        logger.error(
            "Mismatch between number of sentences (%d) and embeddings "
            "(%d).", len(sentences), len(embeddings))
        raise InvalidDataError(
            "Number of sentences and embeddings must match.")

    if sort_order not in ['asc', 'desc']:
        logger.error("Invalid sort order: %s", sort_order)
        raise ValueError("Invalid sort order, must be 'asc' or 'desc'.")

    try:
        scores = compute_score(
            source=query_embedding,
            embeddings=embeddings,
            compute_function=compute_function,
            rounding=rounding,
            progress_bar=progress_bar
        )
        if sort_order == 'asc':
            sorted_indices = np.argsort(scores)[:top_k]
        else:
            sorted_indices = np.argsort(scores)[-top_k:][::-1]

        # Return top_k sentences and their scores
        top_similar = [
            (sentences[i], round(scores[i], rounding))
            for i in sorted_indices
        ]
        return top_similar

    except Exception as err:
        logger.exception("An error occurred during similarity search: %s", err)
        raise RuntimeError(
            "An error occurred while searching for similar sentences."
        ) from err
