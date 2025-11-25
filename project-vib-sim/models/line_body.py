"""Line (1D beam) element model for axial vibrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class LineBody:
    """Represents a 1D axial element with distributed properties."""

    area: float
    length: float
    youngs_modulus: float
    density: float
    name: str = "line"

    def stiffness(self) -> float:
        """Return axial stiffness EA/L."""
        if self.length <= 0:
            raise ValueError("Length must be positive.")
        return self.youngs_modulus * self.area / self.length

    def mass(self) -> float:
        """Return lumped mass ρAL."""
        return self.density * self.area * self.length

    def element_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return 2x2 stiffness and mass matrices."""
        k = self.stiffness()
        m = self.mass()
        k_mat = k * np.array([[1, -1], [-1, 1]], dtype=float)
        m_mat = (m / 6) * np.array([[2, 1], [1, 2]], dtype=float)
        return m_mat, k_mat





