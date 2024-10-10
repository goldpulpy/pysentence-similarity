"""Compute functions module."""
import logging
import numpy as np

# Set up logging
logger = logging.getLogger("pysentence-similarity:compute_function")


def cosine(
    embedding_1: np.ndarray,
    embedding_2: np.ndarray
) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Cosine similarity is a measure of similarity between two non-zero vectors
    of an inner product space that measures the cosine of the angle between 
    them.
    It is defined as the dot product of the vectors divided by the product of 
    their magnitudes (norms). The value ranges from -1 to 1, where 1 indicates 
    that the vectors are identical, 0 indicates orthogonality, and -1 indicates 
    opposite directions.

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

    The Euclidean distance is a measure of the straight-line distance between 
    two points in Euclidean space. It is calculated as the square root of the 
    sum of the squared differences between corresponding elements of the 
    vectors.
    This distance metric is commonly used in various machine learning and
    data analysis tasks to quantify similarity or dissimilarity between data 
    points.

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

    The Manhattan distance, also known as L1 distance or city block distance, 
    measures the distance between two points in a grid-based system by 
    calculating the sum of the absolute differences of their coordinates. 
    It is defined as the sum of the absolute differences between corresponding 
    elements of the vectors. 

    This distance metric is useful in various machine learning applications and 
    optimization problems.

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

    The Jaccard similarity coefficient measures the similarity between two sets 
    by comparing the size of their intersection to the size of their union. 
    For two embedding vectors, the Jaccard similarity is calculated as the 
    sum of the minimum values (intersection) divided by the sum of the maximum 
    values (union) for corresponding elements of the vectors. This metric 
    is particularly useful in applications such as clustering and information 
    retrieval where the similarity between sets is of interest.

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

    The Pearson correlation coefficient measures the linear correlation 
    between two variables, ranging from -1 to 1. A coefficient of 1 indicates 
    a perfect positive linear relationship, 0 indicates no linear correlation, 
    and -1 indicates a perfect negative linear relationship. This metric is 
    commonly used in statistics to determine the strength and direction of a 
    linear relationship between two data sets.

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

    The Minkowski distance is a generalization of both the Euclidean and 
    Manhattan distances, defined as the p-th root of the sum of the absolute 
    differences of the coordinates raised to the p-th power. 
    The Minkowski distance becomes:
    - Euclidean distance when p = 2
    - Manhattan distance when p = 1

    The parameter p controls the 'order' of the distance metric. A higher value 
    of p emphasizes larger differences between dimensions.

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

    The Hamming distance measures the proportion of positions at which 
    the corresponding elements of two vectors are different. It is 
    commonly used for comparing binary strings or categorical data 
    and is defined as the number of differing elements divided by the 
    total number of elements. This distance metric is particularly 
    useful in error detection and correction codes.

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

    The Kullback-Leibler (KL) divergence is a measure of how one probability 
    distribution diverges from a second, expected probability distribution. 
    It quantifies the information lost when one distribution is used to 
    approximate another. The KL divergence is always non-negative and is 
    zero if and only if the two distributions are identical.

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

    The Chebyshev distance, also known as the maximum metric, 
    measures the maximum absolute difference between the components 
    of two vectors. It is defined as the greatest of the absolute 
    differences along any coordinate dimension. This distance 
    metric is particularly useful in scenarios where you want to 
    focus on the largest difference between dimensions.

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

    Bregman divergence is a generalization of various distance measures 
    based on a convex function. It quantifies the difference between 
    two points in terms of the convex function and its gradient. Bregman 
    divergence is non-negative and equals zero only when the two points 
    are the same. This metric is useful in various applications, 
    including optimization and information theory.

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
