from pathlib import Path
from typing import Union

import numpy as np
import scipy
import torch
import torch.nn.functional as F
import torchaudio


def center_trim(to_trim: torch.Tensor, target: torch.Tensor, dim=-1):
    """
    Trims a tensor to match the length of another, removing equally from both sides.

    Args:
        to_trim: the tensor to trim
        target: the tensor whose length to match
        dim: the dimension in which to trim

    Returns:
        The trimmed to_trim tensor
    """
    return to_trim.narrow(dim, (to_trim.shape[dim] - target.shape[dim]) // 2, target.shape[dim])


def pad(
    to_pad: Union[np.ndarray, torch.Tensor], target_length: int
) -> Union[np.ndarray, torch.Tensor]:
    delta = int(target_length - to_pad.shape[-1])
    padding_left = max(0, delta) // 2
    padding_right = delta - padding_left

    if isinstance(to_pad, np.ndarray):
        return np.pad(to_pad, (padding_left, padding_right))
    elif isinstance(to_pad, torch.Tensor):
        return F.pad(to_pad, (padding_left, padding_right))


def hp_filter(to_filter: torch.Tensor) -> torch.Tensor:
    """Apply highpass filter"""
    with open("data/hp_filter_coeffs.txt") as f:
        coeffs = [float(coeff) for coeff in f.read().strip().split(",")]
    return torch.Tensor(scipy.signal.lfilter(coeffs, 1, to_filter))


def save_as_audio(x, out_path: str):
    assert len(x.shape) == 2
    p = Path(out_path)
    p.parent.mkdir(exist_ok=True)
    for i in range(x.shape[0]):
        torchaudio.save(p.with_stem(f"{p.stem}_{i}"), x[i].unsqueeze(dim=0), sample_rate=int(8e3))
