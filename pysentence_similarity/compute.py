"""Compute functions module."""
import logging
import numpy as np

# Set up logging
logger = logging.getLogger("pysentence-similarity:compute_function")


def cosine(
    embedding_1: np.ndarray,
    embedding_2: np.ndarray
) -> float:
    """Compute cosine similarity between two embedding vectors.

    :param embedding_1: First embedding vector.
    :type embedding_1: np.ndarray
    :param embedding_2: Second embedding vector.
    :type embedding_2: np.ndarray
    :return: Cosine similarity score.
    :rtype: float
    """
    try:
        embedding_1_norm = embedding_1 / np.linalg.norm(embedding_1)
        embedding_2_norm = embedding_2 / np.linalg.norm(embedding_2)
        similarity = np.dot(embedding_1_norm, embedding_2_norm)
        return similarity
    except Exception as err:
        logger.error("Error computing cosine similarity: %s", err)
        raise


def euclidean(
    embedding_1: np.ndarray,
    embedding_2: np.ndarray
) -> float:
    """Compute Euclidean distance between two embedding vectors.

    :param embedding_1: First embedding vector.
    :type embedding_1: np.ndarray
    :param embedding_2: Second embedding vector.
    :type embedding_2: np.ndarray
    :return: Euclidean distance.
    :rtype: float
    """
    try:
        distance = np.linalg.norm(embedding_1 - embedding_2)
        return distance
    except Exception as err:
        logger.error("Error computing Euclidean distance: %s", err)
        raise


def manhattan(
    embedding_1: np.ndarray,
    embedding_2: np.ndarray
) -> float:
    """Compute Manhattan distance between two embedding vectors.

    :param embedding_1: First embedding vector.
    :type embedding_1: np.ndarray
    :param embedding_2: Second embedding vector.
    :type embedding_2: np.ndarray
    :return: Manhattan distance.
    :rtype: float
    """
    try:
        distance = np.sum(np.abs(embedding_1 - embedding_2))
        return distance
    except Exception as err:
        logger.error("Error computing Manhattan distance: %s", err)
        raise


def jaccard(
    embedding_1: np.ndarray,
    embedding_2: np.ndarray
) -> float:
    """Compute Jaccard similarity between two embedding vectors.

    :param embedding_1: First embedding vector.
    :type embedding_1: np.ndarray
    :param embedding_2: Second embedding vector.
    :type embedding_2: np.ndarray
    :return: Jaccard similarity score.
    :rtype: float
    """
    try:
        intersection = np.minimum(embedding_1, embedding_2).sum()
        union = np.maximum(embedding_1, embedding_2).sum()
        similarity = intersection / union
        return similarity
    except Exception as err:
        logger.error("Error computing Jaccard similarity: %s", err)
        raise


def pearson(
    embedding_1: np.ndarray,
    embedding_2: np.ndarray
) -> float:
    """Compute Pearson correlation between two embedding vectors.

    :param embedding_1: First embedding vector.
    :type embedding_1: np.ndarray
    :param embedding_2: Second embedding vector.
    :type embedding_2: np.ndarray
    :return: Pearson correlation coefficient.
    :rtype: float
    """
    try:
        correlation = np.corrcoef(embedding_1, embedding_2)[0, 1]
        return correlation
    except Exception as err:
        logger.error("Error computing Pearson correlation: %s", err)
        raise


def minkowski(
    embedding_1: np.ndarray,
    embedding_2: np.ndarray,
    p: int = 3
) -> float:
    """Compute Minkowski distance between two embedding vectors.

    :param embedding_1: First embedding vector.
    :type embedding_1: np.ndarray
    :param embedding_2: Second embedding vector.
    :type embedding_2: np.ndarray
    :param p: Power parameter for Minkowski distance (default is 3).
    :type p: int
    :return: Minkowski distance.
    :rtype: float
    """
    try:
        distance = np.sum(np.abs(embedding_1 - embedding_2) ** p) ** (1 / p)
        return distance
    except Exception as err:
        logger.error("Error computing Minkowski distance: %s", err)
        raise


def hamming(
    embedding_1: np.ndarray,
    embedding_2: np.ndarray
) -> float:
    """Compute Hamming distance between two embedding vectors.

    :param embedding_1: First embedding vector.
    :type embedding_1: np.ndarray
    :param embedding_2: Second embedding vector.
    :type embedding_2: np.ndarray
    :return: Hamming distance.
    :rtype: float
    """
    try:
        if embedding_1.shape != embedding_2.shape:
            raise ValueError(
                "Embeddings must have the same length for Hamming distance."
            )
        distance = np.mean(embedding_1 != embedding_2)
        return distance
    except Exception as err:
        logger.error("Error computing Hamming distance: %s", err)
        raise


def kl_divergence(
    embedding_1: np.ndarray,
    embedding_2: np.ndarray
) -> float:
    """Compute Kullback-Leibler divergence between two probability 
    distributions.

    :param embedding_1: First probability distribution.
    :param embedding_2: Second probability distribution.
    :return: KL divergence.
    """
    try:
        embedding_1 = embedding_1 / np.sum(embedding_1)
        embedding_2 = embedding_2 / np.sum(embedding_2)

        # Avoid division by zero and log(0)
        embedding_1 = np.clip(embedding_1, 1e-10, 1)
        embedding_2 = np.clip(embedding_2, 1e-10, 1)
        divergence = np.sum(embedding_1 * np.log(embedding_1 / embedding_2))
        return divergence
    except Exception as err:
        logger.error("Error computing KL divergence: %s", err)
        raise


def chebyshev(
    embedding_1: np.ndarray,
    embedding_2: np.ndarray
) -> float:
    """Compute Chebyshev distance between two embedding vectors.

    :param embedding_1: First embedding vector.
    :type embedding_1: np.ndarray
    :param embedding_2: Second embedding vector.
    :type embedding_2: np.ndarray
    :return: Chebyshev distance.
    :rtype: float
    """
    try:
        distance = np.max(np.abs(embedding_1 - embedding_2))
        return distance
    except Exception as err:
        logger.error("Error computing Chebyshev distance: %s", err)
        raise


def bregman(
    embedding_1: np.ndarray,
    embedding_2: np.ndarray,
    f=np.square,
    grad_f=lambda x: 2 * x
) -> float:
    """Compute Bregman divergence between two embedding vectors using a convex
    function.

    :param embedding_1: First embedding vector.
    :param embedding_2: Second embedding vector.
    :param f: Convex function to compute divergence (default is square
    function).
    :param grad_f: Gradient of convex function. If not provided, defaults to 2
    * x for f=x^2.
    :return: Bregman divergence.
    """
    try:
        divergence = f(embedding_1).sum() - f(embedding_2).sum() - \
            np.dot(grad_f(embedding_2), (embedding_1 - embedding_2))
        return divergence
    except Exception as err:
        logger.error("Error computing Bregman divergence: %s", err)
        raise
