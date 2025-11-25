"""Eigenvalue analysis utilities."""

from __future__ import annotations

import numpy as np
from scipy import linalg


def solve_eigen(M: np.ndarray, K: np.ndarray, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Solve the generalized eigen problem |K - ω^2 M| = 0."""
    eigvals, eigvecs = linalg.eigh(K, M)
    eigvals = np.maximum(eigvals, 0.0)
    natural_freqs = np.sqrt(eigvals)
    if normalize:
        for i in range(eigvecs.shape[1]):
            mode = eigvecs[:, i]
            modal_mass = mode.T @ M @ mode
            eigvecs[:, i] = mode / np.sqrt(modal_mass)
    return natural_freqs, eigvecs


