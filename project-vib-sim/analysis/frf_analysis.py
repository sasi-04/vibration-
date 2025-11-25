"""Frequency response functions for assembled systems."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_frf(
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    omega: np.ndarray,
    dof_in: int = 0,
    dof_out: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FRF magnitude and phase between two DOFs."""
    omega = np.asarray(omega, dtype=float)
    n = M.shape[0]
    force_vec = np.zeros((n, 1))
    force_vec[dof_in, 0] = 1.0
    responses = []
    for w in omega:
        dyn = K - (w**2) * M + 1j * w * C
        x = np.linalg.solve(dyn, force_vec)
        responses.append(x[dof_out, 0])
    responses = np.array(responses)
    return np.abs(responses), np.angle(responses, deg=True)





