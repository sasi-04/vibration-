"""Time-domain simulations for multi-DOF systems."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Union

import numpy as np
from scipy.integrate import solve_ivp

ForceDict = Dict[Union[int, str], Callable[[float], float]]


def sine_force(amplitude: float, frequency: float) -> Callable[[float], float]:
    return lambda t: amplitude * np.sin(2 * np.pi * frequency * t)


def impulse_force(magnitude: float, t_impulse: float = 0.0) -> Callable[[float], float]:
    return lambda t: magnitude if abs(t - t_impulse) < 1e-6 else 0.0


def random_force(seed: Optional[int], level: float) -> Callable[[float], float]:
    rng = np.random.default_rng(seed)

    def _force(t: float) -> float:
        return level * rng.standard_normal()

    return _force


def simulate_forced_response(
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    t_span: tuple[float, float],
    loads: ForceDict,
    t_eval: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    v0: Optional[np.ndarray] = None,
) -> solve_ivp:
    """Simulate the forced response with arbitrary input loads per DOF."""
    n = M.shape[0]
    x0 = np.zeros(n) if x0 is None else x0
    v0 = np.zeros(n) if v0 is None else v0

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        x = y[:n]
        v = y[n:]
        f = np.zeros(n)
        for idx, fn in loads.items():
            if idx < n:  # Ensure index is valid
                f[idx] += fn(t)
        acc = np.linalg.solve(M, f - C @ v - K @ x)
        return np.concatenate([v, acc])

    y0 = np.concatenate([x0, v0]).astype(float)
    return solve_ivp(rhs, t_span, y0, t_eval=t_eval, rtol=1e-6, atol=1e-8, method='RK45')




