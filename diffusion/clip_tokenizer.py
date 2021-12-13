from clip_jax.simple_tokenizer import SimpleTokenizer as Tokenizer
import numpy as np
from typing import Union, List
import torch

tokenizer = Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = np.zeros((len(all_tokens), context_length), dtype=np.int32)
    max_len = context_length - 1
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            # raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :] = tokens[:max_len] + [eot_token]
        else:
            result[i, :len(tokens)] = tokens
    return result