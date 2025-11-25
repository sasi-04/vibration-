"""Unit tests for FRF analysis."""

import numpy as np
import pytest

from analysis.frf_analysis import compute_frf


def test_frf_simple_1dof():
    """Test FRF on simple 1-DOF system."""
    M = np.array([[1.0]])
    C = np.array([[10.0]])
    K = np.array([[100.0]])
    
    omega = np.array([0.0, 10.0, 100.0])
    mag, phase = compute_frf(M, C, K, omega, dof_in=0, dof_out=0)
    
    assert mag.shape == omega.shape
    assert phase.shape == omega.shape
    
    # At DC, magnitude should be 1/k
    assert np.isclose(mag[0], 1.0 / 100.0, rtol=1e-3)


def test_frf_2dof_system():
    """Test FRF on 2-DOF system."""
    M = np.eye(2)
    C = np.array([[10.0, 0.0], [0.0, 10.0]])
    K = np.array([[100.0, 0.0], [0.0, 200.0]])
    
    omega = np.linspace(1.0, 50.0, 100)
    mag, phase = compute_frf(M, C, K, omega, dof_in=0, dof_out=1)
    
    assert mag.shape == omega.shape
    assert phase.shape == omega.shape
    assert np.all(mag >= 0)  # Magnitude should be non-negative


def test_frf_peak_detection():
    """Test that FRF shows peaks near natural frequencies."""
    M = np.eye(2)
    C = np.array([[5.0, 0.0], [0.0, 5.0]])
    K = np.array([[100.0, -10.0], [-10.0, 100.0]])
    
    # Compute natural frequencies
    from analysis.eigen_analysis import solve_eigen
    nat_freqs, _ = solve_eigen(M, K)
    
    # Compute FRF over wide range
    omega = np.linspace(0.1, 20.0, 1000)
    mag, _ = compute_frf(M, C, K, omega, dof_in=0, dof_out=1)
    
    # Should have peaks near natural frequencies
    peak_indices = []
    for i in range(1, len(mag) - 1):
        if mag[i] > mag[i-1] and mag[i] > mag[i+1] and mag[i] > np.max(mag) * 0.1:
            peak_indices.append(i)
    
    # Should find at least one peak
    assert len(peak_indices) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


