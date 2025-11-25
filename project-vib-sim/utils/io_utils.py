"""I/O utilities for loading/saving models and reading diagram assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def load_params_yaml(filepath: str | Path) -> Dict[str, Any]:
    """Load component and enclosure parameters from YAML file.
    
    Args:
        filepath: Path to YAML file
        
    Returns:
        Dictionary with parameters
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {filepath}")
    
    with open(path, 'r') as f:
        params = yaml.safe_load(f)
    return params


def save_params_yaml(params: Dict[str, Any], filepath: str | Path) -> None:
    """Save parameters to YAML file.
    
    Args:
        params: Dictionary with parameters
        filepath: Path to save YAML file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)


def save_matrices(filepath: str | Path, M: np.ndarray, C: np.ndarray, K: np.ndarray,
                  metadata: Dict[str, Any] | None = None) -> None:
    """Save assembled matrices to NPZ file.
    
    Args:
        filepath: Path to save NPZ file
        M: Mass matrix
        C: Damping matrix
        K: Stiffness matrix
        metadata: Optional metadata dictionary
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {'M': M, 'C': C, 'K': K}
    if metadata:
        save_dict['metadata'] = metadata
    
    np.savez(path, **save_dict)


def load_matrices(filepath: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any] | None]:
    """Load assembled matrices from NPZ file.
    
    Args:
        filepath: Path to NPZ file
        
    Returns:
        Tuple of (M, C, K, metadata)
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {filepath}")
    
    data = np.load(path, allow_pickle=True)
    M = data['M']
    C = data['C']
    K = data['K']
    metadata = dict(data['metadata'].item()) if 'metadata' in data else None
    
    return M, C, K, metadata


def load_diagram_image(filepath: str | Path) -> bytes | None:
    """Load diagram image file.
    
    Args:
        filepath: Path to image file
        
    Returns:
        Image bytes or None if file doesn't exist
    """
    path = Path(filepath)
    if not path.exists():
        return None
    
    with open(path, 'rb') as f:
        return f.read()


def get_diagram_path() -> Path:
    """Get path to technical description diagram.
    
    Returns:
        Path object (may not exist)
    """
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).parent.parent / "data" / 'A_flowchart_in_the_image_titled_"Technical_Descrip.png',
        Path(__file__).parent.parent / "assets" / 'A_flowchart_in_the_image_titled_"Technical_Descrip.png',
        Path("/mnt/data/A_flowchart_in_the_image_titled_\"Technical_Descrip.png"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Return first path as default (may not exist)
    return possible_paths[0]


