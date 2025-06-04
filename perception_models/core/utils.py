import math
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn.functional as F


@dataclass
class InitArgs:
    use_gaussian: bool = True  # gaussian vs uniform
    coeff_std: Optional[float] = None  # std coeff multiplier
    no_init: bool = False


def get_init_fn(
    args: InitArgs, input_dim: int, init_depth: Optional[int]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Init functions.
    """
    if args.no_init:
        return lambda x: x

    # standard deviation
    std = 1 / math.sqrt(input_dim)
    std = std if args.coeff_std is None else (args.coeff_std * std)

    # rescale with depth
    if init_depth is not None:
        std = std / math.sqrt(2 * init_depth)

    # gaussian vs uniform
    if args.use_gaussian:
        return partial(
            torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
    else:
        bound = math.sqrt(3) * std  # ensure the standard deviation is `std`
        return partial(torch.nn.init.uniform_, a=-bound, b=bound)


def manual_scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    d_k = q.size(-1)
    scale = scale if scale is not None else 1.0 / (d_k ** 0.5)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply causal mask
    if is_causal:
        causal_mask = torch.triu(
            torch.ones_like(attn_scores), diagonal=1
        ).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

    # Apply attention mask
    if attn_mask is not None:
        # attn_mask shape must be broadcastable to attn_scores
        attn_scores = attn_scores.masked_fill(attn_mask.bool(), float('-inf'))

    # Softmax with numerical stability
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_probs = attn_probs.masked_fill(torch.isnan(attn_probs), 0.0)  # prevent nan from propagating

    if dropout_p > 0.0 and torch.is_grad_enabled():
        attn_probs = F.dropout(attn_probs, p=dropout_p)

    return torch.matmul(attn_probs, v)
