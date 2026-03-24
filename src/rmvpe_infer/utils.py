"""F0 decoding utilities for RMVPE inference."""

import numpy as np
import librosa
import torch
from .constants import N_CLASS, CONST


def to_local_average_cents(salience, center=None, thred=0.03):
    """Find the weighted average cents near the argmax bin."""
    if not hasattr(to_local_average_cents, 'cents_mapping'):
        to_local_average_cents.cents_mapping = (
                20 * np.arange(N_CLASS) + CONST)

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * to_local_average_cents.cents_mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum if np.max(salience) > thred else 0
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :], None, thred) for i in
                         range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")


def to_viterbi_cents(salience, thred=0.03):
    """Viterbi decoding for cents estimation."""
    if not hasattr(to_viterbi_cents, 'transition'):
        xx, yy = np.meshgrid(range(N_CLASS), range(N_CLASS))
        transition = np.maximum(30 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        to_viterbi_cents.transition = transition

    prob = salience.T
    prob = prob / prob.sum(axis=0)

    path = librosa.sequence.viterbi(prob, to_viterbi_cents.transition).astype(np.int64)

    return np.array([to_local_average_cents(salience[i, :], path[i], thred) for i in
                     range(len(path))])


def to_local_average_f0(hidden, center=None, thred=0.03):
    """Convert model hidden output to F0 values using local average."""
    idx = torch.arange(N_CLASS, device=hidden.device)[None, None, :]
    idx_cents = idx * 20 + CONST
    if center is None:
        center = torch.argmax(hidden, dim=2, keepdim=True)
    start = torch.clip(center - 4, min=0)
    end = torch.clip(center + 5, max=N_CLASS)
    idx_mask = (idx >= start) & (idx < end)
    weights = hidden * idx_mask
    product_sum = torch.sum(weights * idx_cents, dim=2)
    weight_sum = torch.sum(weights, dim=2)
    cents = product_sum / (weight_sum + (weight_sum == 0))
    f0 = 10 * 2 ** (cents / 1200)
    uv = hidden.max(dim=2)[0] < thred
    f0 = f0 * ~uv
    return f0.squeeze(0).cpu().numpy()


def to_viterbi_f0(hidden, thred=0.03):
    """Convert model hidden output to F0 values using Viterbi decoding."""
    if not hasattr(to_viterbi_cents, 'transition'):
        xx, yy = np.meshgrid(range(N_CLASS), range(N_CLASS))
        transition = np.maximum(30 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        to_viterbi_cents.transition = transition

    prob = hidden.squeeze(0).cpu().numpy()
    prob = prob.T
    prob = prob / prob.sum(axis=0)

    path = librosa.sequence.viterbi(prob, to_viterbi_cents.transition).astype(np.int64)
    center = torch.from_numpy(path).unsqueeze(0).unsqueeze(-1).to(hidden.device)

    return to_local_average_f0(hidden, center=center, thred=thred)
