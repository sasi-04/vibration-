"""Global system assembly for vibrating components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
from scipy import sparse

from models.mass_spring_damper import MassSpringDamper
from models.line_body import LineBody


@dataclass
class SystemAssembly:
    """Assembles multi-component mass-spring-damper-line systems."""

    node_count: int
    masses: List[MassSpringDamper] = field(default_factory=list)
    line_bodies: List[LineBody] = field(default_factory=list)
    mass_map: List[int] = field(default_factory=list)
    line_body_map: List[Sequence[int]] = field(default_factory=list)

    def add_component(self, msd: MassSpringDamper, node_id: int) -> None:
        """Add a lumped component at global DOF index node_id."""
        if node_id >= self.node_count:
            raise ValueError("node_id out of range.")
        self.masses.append(msd)
        self.mass_map.append(node_id)

    def add_line_body(self, element: LineBody, node_i: int, node_j: int) -> None:
        """Add a connecting line-body between two nodes."""
        if node_i >= self.node_count or node_j >= self.node_count:
            raise ValueError("Node index out of range.")
        self.line_bodies.append(element)
        self.line_body_map.append((node_i, node_j))

    def finalize_system(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assemble and return global M, C, K matrices."""
        dof = self.node_count
        M = np.zeros((dof, dof))
        C = np.zeros((dof, dof))
        K = np.zeros((dof, dof))

        for msd, idx in zip(self.masses, self.mass_map):
            M[idx, idx] += msd.mass
            C[idx, idx] += msd.damping
            K[idx, idx] += msd.stiffness

        for element, (i, j) in zip(self.line_bodies, self.line_body_map):
            m_mat, k_mat = element.element_matrices()
            for local_a, global_a in enumerate((i, j)):
                for local_b, global_b in enumerate((i, j)):
                    M[global_a, global_b] += m_mat[local_a, local_b]
                    K[global_a, global_b] += k_mat[local_a, local_b]

        return M, C, K

    @staticmethod
    def assemble_sparse(matrix: np.ndarray) -> sparse.csr_matrix:
        """Return sparse CSR representation."""
        return sparse.csr_matrix(matrix)





