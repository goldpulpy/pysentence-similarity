"""Module for pooling token embeddings."""
from typing import List
import numpy as np


def max_pooling(
    model_output: np.ndarray,
    attention_mask: List[int]
) -> np.ndarray:
    """
    Perform max pooling on token embeddings.

    :param model_output: Model output (token embeddings).
    :type model_output: np.ndarray
    :param attention_mask: Attention mask for the tokens.
    :type attention_mask: List[int]
    :return: Embedding vector for the entire sentence.
    :rtype: np.ndarray
    """
    token_embeddings = model_output
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    pooled_embedding = np.max(token_embeddings * input_mask_expanded, axis=1)
    return pooled_embedding


def mean_pooling(
    model_output: np.ndarray,
    attention_mask: List[int]
) -> np.ndarray:
    """
    Perform mean pooling on token embeddings.

    :param model_output: Model output (token embeddings).
    :type model_output: np.ndarray
    :param attention_mask: Attention mask for the tokens.
    :type attention_mask: List[int]
    :return: Embedding vector for the entire sentence.
    :rtype: np.ndarray
    """
    token_embeddings = model_output
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    pooled_embedding = (
        np.sum(token_embeddings * input_mask_expanded, axis=1) /
        np.clip(np.sum(input_mask_expanded, axis=1), 1e-9, None)
    )
    return pooled_embedding


def min_pooling(
    model_output: np.ndarray,
    attention_mask: List[int]
) -> np.ndarray:
    """
    Perform min pooling on token embeddings.

    :param model_output: Model output (token embeddings).
    :type model_output: np.ndarray
    :param attention_mask: Attention mask for the tokens.
    :type attention_mask: List[int]
    :return: Embedding vector for the entire sentence.
    :rtype: np.ndarray
    """
    token_embeddings = model_output
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    pooled_embedding = np.min(
        np.where(input_mask_expanded > 0, token_embeddings, np.inf), axis=1)
    return pooled_embedding
