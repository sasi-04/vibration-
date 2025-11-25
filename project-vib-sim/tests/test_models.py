"""Unit tests for model components."""

import numpy as np
import pytest

from models.mass_spring_damper import MassSpringDamper
from models.line_body import LineBody


def test_mass_spring_damper_basic():
    """Test basic MassSpringDamper properties."""
    m, k, c = 10.0, 1e5, 100.0
    msd = MassSpringDamper(m, k, c, name="test")
    
    assert msd.mass == m
    assert msd.stiffness == k
    assert msd.damping == c
    assert msd.name == "test"


def test_mass_spring_damper_natural_frequency():
    """Test natural frequency calculation."""
    m, k = 10.0, 1e5
    msd = MassSpringDamper(m, k, 100.0)
    
    omega_n = msd.natural_frequency()
    expected = np.sqrt(k / m)
    
    assert np.isclose(omega_n, expected, rtol=1e-6)


def test_mass_spring_damper_damping_ratio():
    """Test damping ratio calculation."""
    m, k, c = 10.0, 1e5, 200.0
    msd = MassSpringDamper(m, k, c)
    
    zeta = msd.damping_ratio()
    expected = c / (2 * np.sqrt(m * k))
    
    assert np.isclose(zeta, expected, rtol=1e-6)


def test_mass_spring_damper_frf():
    """Test frequency response function."""
    m, k, c = 10.0, 1e5, 100.0
    msd = MassSpringDamper(m, k, c)
    
    frequencies = np.array([0.0, 10.0, 100.0])
    H = msd.frf(frequencies)
    
    assert H.shape == frequencies.shape
    assert np.iscomplexobj(H)
    
    # At DC (ω=0), H should be 1/k
    assert np.isclose(np.abs(H[0]), 1.0 / k, rtol=1e-6)


def test_line_body_stiffness():
    """Test beam element stiffness calculation."""
    A, L, E = 0.01, 0.5, 2e11
    beam = LineBody(A, L, E, 7800.0)
    
    Ke = beam.stiffness_matrix()
    expected_k = E * A / L
    
    assert Ke.shape == (2, 2)
    assert np.isclose(Ke[0, 0], expected_k, rtol=1e-6)
    assert np.isclose(Ke[1, 1], expected_k, rtol=1e-6)
    assert np.isclose(Ke[0, 1], -expected_k, rtol=1e-6)
    assert np.isclose(Ke[1, 0], -expected_k, rtol=1e-6)


def test_line_body_mass():
    """Test beam element mass calculation."""
    A, L, rho = 0.01, 0.5, 7800.0
    beam = LineBody(A, L, 2e11, rho)
    
    Me = beam.mass_matrix()
    expected_m = rho * A * L / 2  # Lumped mass per node
    
    assert Me.shape == (2, 2)
    assert np.isclose(Me[0, 0], expected_m, rtol=1e-6)
    assert np.isclose(Me[1, 1], expected_m, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


