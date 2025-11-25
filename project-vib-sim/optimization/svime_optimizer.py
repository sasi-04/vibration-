"""Smart Vibration Interaction Mitigation Engine (SVIME)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from analysis.frf_analysis import compute_frf


@dataclass
class SVIMEOptimizer:
    """Automatically tunes isolation stiffness/damping to reduce coupling."""

    M: np.ndarray
    C: np.ndarray
    K: np.ndarray
    adjustable_dofs: Sequence[int]
    omega_grid: np.ndarray

    def identify_coupling(self) -> Tuple[int, int, float]:
        """Return the DOF pair with strongest FRF magnitude."""
        n = self.M.shape[0]
        best = (0, 0, -np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                mag, _ = compute_frf(self.M, self.C, self.K, self.omega_grid, i, j)
                peak = np.max(mag)
                if peak > best[2]:
                    best = (i, j, peak)
        return best

    def optimize(self) -> dict:
        """Run differential evolution to minimize coupling peaks."""
        if not self.adjustable_dofs:
            raise ValueError("SVIME requires adjustable DOFs.")
        pair_i, pair_j, baseline = self.identify_coupling()
        base_C_diag = np.diag(self.C).copy()
        base_K_diag = np.diag(self.K).copy()

        def objective(params: np.ndarray) -> float:
            trial_C = self.C.copy()
            trial_K = self.K.copy()
            for dof, k_scale, c_scale in zip(
                self.adjustable_dofs,
                params[: len(self.adjustable_dofs)],
                params[len(self.adjustable_dofs) :],
            ):
                trial_K[dof, dof] = base_K_diag[dof] * k_scale
                trial_C[dof, dof] = base_C_diag[dof] * c_scale
            mag, _ = compute_frf(self.M, trial_C, trial_K, self.omega_grid, pair_i, pair_j)
            return np.max(mag)

        bounds = [(0.5, 2.0)] * len(self.adjustable_dofs) + [(0.5, 3.0)] * len(
            self.adjustable_dofs
        )
        result = differential_evolution(objective, bounds, polish=True, maxiter=20, tol=1e-3)

        best_params = result.x
        opt_C = self.C.copy()
        opt_K = self.K.copy()
        for dof, k_scale, c_scale in zip(
            self.adjustable_dofs,
            best_params[: len(self.adjustable_dofs)],
            best_params[len(self.adjustable_dofs) :],
        ):
            opt_K[dof, dof] = base_K_diag[dof] * k_scale
            opt_C[dof, dof] = base_C_diag[dof] * c_scale

        mag, _ = compute_frf(self.M, opt_C, opt_K, self.omega_grid, pair_i, pair_j)
        return {
            "pair": (pair_i, pair_j),
            "baseline_peak": baseline,
            "optimized_peak": float(np.max(mag)),
            "improvement_pct": float((baseline - np.max(mag)) / baseline * 100.0),
            "best_params": best_params,
            "optimized_C": opt_C,
            "optimized_K": opt_K,
            "result": result,
        }





