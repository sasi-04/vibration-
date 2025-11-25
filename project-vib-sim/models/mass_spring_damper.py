"""Mass-spring-damper component definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp


ForceFunction = Callable[[float], float]


@dataclass
class MassSpringDamper:
    """Represents a single-degree-of-freedom mass–spring–damper element."""

    mass: float
    stiffness: float
    damping: float
    name: str = "msd"
    force_fn: Optional[ForceFunction] = None
    metadata: dict = field(default_factory=dict)

    def natural_frequency(self) -> float:
        """Return the undamped natural frequency (rad/s)."""
        if self.mass <= 0 or self.stiffness <= 0:
            raise ValueError("Mass and stiffness must be positive to compute ω_n.")
        return np.sqrt(self.stiffness / self.mass)

    def damping_ratio(self) -> float:
        """Return the damping ratio ζ."""
        if self.mass <= 0 or self.stiffness <= 0:
            raise ValueError("Mass and stiffness must be positive to compute ζ.")
        return self.damping / (2.0 * np.sqrt(self.mass * self.stiffness))

    def state_space_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the state-space matrices (A, B, C, D)."""
        m, c, k = self.mass, self.damping, self.stiffness
        A = np.array([[0.0, 1.0], [-k / m, -c / m]])
        B = np.array([[0.0], [1.0 / m]])
        C = np.eye(2)
        D = np.zeros((2, 1))
        return A, B, C, D

    def frequency_response(self, omega: Sequence[float]) -> np.ndarray:
        """Return the scalar frequency response H(jω)."""
        omega = np.asarray(omega, dtype=float)
        m, c, k = self.mass, self.damping, self.stiffness
        return 1.0 / (k - m * omega**2 + 1j * omega * c)

    def equivalent_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the element mass, damping, and stiffness matrices."""
        return (
            np.array([[self.mass]]),
            np.array([[self.damping]]),
            np.array([[self.stiffness]]),
        )

    def simulate(
        self,
        t_span: Tuple[float, float],
        t_eval: Optional[Sequence[float]] = None,
        x0: float = 0.0,
        v0: float = 0.0,
        force_fn: Optional[ForceFunction] = None,
        rtol: float = 1e-7,
        atol: float = 1e-9,
    ) -> solve_ivp:
        """Simulate the time response using solve_ivp."""

        m, c, k = self.mass, self.damping, self.stiffness
        force = force_fn or self.force_fn or (lambda _: 0.0)

        def ode(t: float, y: np.ndarray) -> np.ndarray:
            x, v = y
            f = force(t)
            xdd = (f - c * v - k * x) / m
            return np.array([v, xdd], dtype=float)

        return solve_ivp(
            ode,
            t_span=t_span,
            y0=np.array([x0, v0], dtype=float),
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
        )


def build_msd_array(
    masses: Sequence[float],
    stiffnesses: Sequence[float],
    dampings: Sequence[float],
    names: Optional[Sequence[str]] = None,
) -> List[MassSpringDamper]:
    """Convenience helper for creating multiple MSD components at once."""

    if not (len(masses) == len(stiffnesses) == len(dampings)):
        raise ValueError("Mass, stiffness, and damping arrays must have equal length.")

    result = []
    for idx, (m, k, c) in enumerate(zip(masses, stiffnesses, dampings)):
        result.append(
            MassSpringDamper(
                mass=m,
                stiffness=k,
                damping=c,
                name=names[idx] if names else f"msd_{idx}",
            )
        )
    return result





