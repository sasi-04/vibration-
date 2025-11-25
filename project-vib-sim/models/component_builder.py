"""Helper functions to create component models from parameters."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from models.mass_spring_damper import MassSpringDamper
from models.line_body import LineBody


def create_component_from_params(params: Dict, name: str = "Component") -> MassSpringDamper:
    """Create a MassSpringDamper component from parameter dictionary.
    
    Args:
        params: Dictionary with keys 'mass', 'stiffness', 'damping'
        name: Component name
        
    Returns:
        MassSpringDamper instance
    """
    m = params.get('mass', 50.0)
    k = params.get('stiffness', 1e5)
    c = params.get('damping', 100.0)
    
    return MassSpringDamper(m, k, c, name=name)


def create_line_body_from_params(params: Dict) -> LineBody:
    """Create a LineBody element from parameter dictionary.
    
    Args:
        params: Dictionary with keys 'area', 'length', 'youngs_modulus', 'density'
        
    Returns:
        LineBody instance
    """
    area = params.get('area', 0.01)
    length = params.get('length', 0.5)
    E = params.get('youngs_modulus', 2e11)
    rho = params.get('density', 7800.0)
    
    return LineBody(area, length, E, rho)


def select_interface_dofs(component_dofs: List[int], interface_nodes: List[int]) -> List[int]:
    """Select interface DOFs for component connection.
    
    Args:
        component_dofs: List of component DOF indices
        interface_nodes: List of interface node indices
        
    Returns:
        List of selected interface DOF indices
    """
    return [dof for dof in component_dofs if dof in interface_nodes]


def create_reduced_component(component: MassSpringDamper, 
                           interface_dofs: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reduced-order model matrices for a component.
    
    Args:
        component: MassSpringDamper component
        interface_dofs: List of interface DOF indices to retain
        
    Returns:
        Tuple of (M_reduced, C_reduced, K_reduced)
    """
    # Get full matrices (1 DOF for simple MSD)
    M_full = np.array([[component.mass]])
    C_full = np.array([[component.damping]])
    K_full = np.array([[component.stiffness]])
    
    # For now, return full matrices (reduction would require Guyan reduction or similar)
    # This is a placeholder for future enhancement
    return M_full, C_full, K_full


def build_two_motor_system(motor1_params: Dict, motor2_params: Dict,
                          frame_params: Dict, beam_params: Dict) -> Dict:
    """Build a complete two-motor system configuration.
    
    Args:
        motor1_params: Parameters for first motor
        motor2_params: Parameters for second motor
        frame_params: Parameters for frame/enclosure
        beam_params: Parameters for beam elements
        
    Returns:
        Dictionary with component instances and configuration
    """
    motor1 = create_component_from_params(motor1_params, name="Motor1")
    motor2 = create_component_from_params(motor2_params, name="Motor2")
    frame = create_component_from_params(frame_params, name="Frame")
    beam = create_line_body_from_params(beam_params)
    
    return {
        'motor1': motor1,
        'motor2': motor2,
        'frame': frame,
        'beam': beam,
        'interface_dofs': [0, 1, 2],  # Motor1, Motor2, Frame
    }
