"""Module for pooling token embeddings."""
from typing import List
import numpy as np


def max_pooling(
    model_output: np.ndarray,
    attention_mask: List[int]
) -> np.ndarray:
    """
    Perform max pooling on token embeddings, using an attention mask to ignore 
    padding tokens.

    This function takes in token embeddings (e.g., from a transformer model's 
    output) and an attention mask and applies a max pooling operation across 
    the token embeddings for each sentence. The attention mask ensures that 
    padding tokens (which have a mask value of 0) are ignored in the pooling 
    operation.

    Max pooling selects the maximum value across the embedding dimension for 
    each token, after multiplying the embeddings by the attention mask. This 
    results in a pooled embedding representing the entire input sentence.


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
    Perform mean pooling on token embeddings, using an attention mask to ignore 
    padding tokens.

    This function computes the mean (average) of the token embeddings for each 
    sentence, ignoring the padding tokens by using an attention mask. The 
    attention mask helps in weighting the valid tokens during pooling and 
    ensures that the padding tokens (marked as 0 in the mask) are excluded from 
    the average computation.

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
    Perform min pooling on token embeddings, using an attention mask to ignore 
    padding tokens.

    This function computes the minimum of the token embeddings for each 
    sentence, while ignoring padding tokens by utilizing an attention mask. The 
    attention mask ensures that tokens marked as padding (with a value of 0) 
    are not considered in the min pooling operation, effectively allowing the 
    computation to focus only on valid tokens.

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
