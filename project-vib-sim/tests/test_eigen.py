"""Unit tests for eigen analysis."""

import numpy as np
import pytest

from analysis.eigen_analysis import solve_eigen


def test_eigen_simple_2dof():
    """Test eigen solver on simple 2-DOF system."""
    # Simple 2-DOF system: two uncoupled oscillators
    M = np.array([[1.0, 0.0], [0.0, 1.0]])
    K = np.array([[100.0, 0.0], [0.0, 200.0]])
    
    nat_freqs, modes = solve_eigen(M, K)
    
    # Expected frequencies: sqrt(100) and sqrt(200)
    expected_freqs = np.array([np.sqrt(100.0), np.sqrt(200.0)])
    
    assert len(nat_freqs) == 2
    assert np.allclose(np.sort(nat_freqs), np.sort(expected_freqs), rtol=1e-6)
    assert modes.shape == (2, 2)


def test_eigen_coupled_system():
    """Test eigen solver on coupled 2-DOF system."""
    # Coupled system
    M = np.array([[1.0, 0.0], [0.0, 1.0]])
    K = np.array([[100.0, -10.0], [-10.0, 100.0]])
    
    nat_freqs, modes = solve_eigen(M, K)
    
    assert len(nat_freqs) == 2
    assert np.all(nat_freqs > 0)  # All frequencies should be positive
    assert modes.shape == (2, 2)
    
    # Check orthogonality: modes^T * M * modes should be diagonal
    M_modal = modes.T @ M @ modes
    assert np.allclose(M_modal, np.diag(np.diag(M_modal)), rtol=1e-6)


def test_eigen_3dof_system():
    """Test eigen solver on 3-DOF system."""
    # 3-DOF system
    M = np.eye(3)
    K = np.array([
        [200.0, -50.0, 0.0],
        [-50.0, 200.0, -50.0],
        [0.0, -50.0, 200.0]
    ])
    
    nat_freqs, modes = solve_eigen(M, K)
    
    assert len(nat_freqs) == 3
    assert np.all(nat_freqs > 0)
    assert modes.shape == (3, 3)
    
    # Frequencies should be in ascending order
    assert np.all(np.diff(nat_freqs) >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


