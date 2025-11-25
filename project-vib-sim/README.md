# Model-Based Design for Two Isolated Vibrating Systems

This project simulates multiple vibrating components sharing an industrial enclosure. Each component is modeled with mass–spring–damper dynamics, connected by axial line-body elements, assembled into global matrices, and analyzed in time and frequency domains. The **Smart Vibration Interaction Mitigation Engine (SVIME)** automatically tunes isolation hardware to reduce coupling.

## Architecture Overview

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ models/      │ -> │ assembly/    │ -> │ analysis/    │ -> │ optimization/│
│ MSD + beams  │    │ global M,C,K │    │ eigen, FRF   │    │ SVIME engine │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
        │                                                      │
        └────────────── utils/plot_utils.py ───────────────────┘
                           │
                      app/streamlit_app.py
```

## Key Formulas

- Component motion: `m x¨ + c ẋ + k x = F(t)`
- Natural frequency: `ωₙ = √(k/m)`
- Damping ratio: `ζ = c / (2 √(mk))`
- Line-body stiffness/mass: `k = EA/L`, `m = ρAL`
- Global dynamics: `M x¨ + C ẋ + K x = F(t)`
- FRF: `H(ω) = (K – ω²M + jωC)⁻¹`
- Eigenvalue problem: `|K – ω²M| = 0`

## SVIME Innovation

1. **Coupling discovery** – scans FRFs across DOF pairs to locate dominant vibration exchange.
2. **Optimization loop** – differential evolution updates stiffness/damping scalars for adjustable isolation mounts.
3. **Peak minimization** – objective minimizes the maximum FRF magnitude for the strongest coupling pair.
4. **Auto-updated matrices** – returns optimized matrices and detailed improvement metrics for reuse.

## Limitations

- Linear models only; nonlinear mounts not represented.
- Lumped damping assignment to line-body elements is not captured.
- SVIME adjusts only diagonal terms; full matrix tuning would require extra constraints.
- Random excitation uses white noise approximation without spectral shaping.

## Usage

## Project Structure

```
project-vib-sim/
├── models/              # Component models (MSD, line-body)
├── assembly/            # System assembly (global matrices)
├── analysis/            # Eigen, FRF, time-domain analysis
├── optimization/        # SVIME optimizer
├── app/                 # Streamlit UI
├── utils/               # Plotting and I/O utilities
├── notebooks/           # Demo Jupyter notebook
├── data/                # Example parameters and outputs
├── tests/               # Unit tests
├── main.py              # CLI smoke test
└── requirements.txt     # Python dependencies
```

## Dependencies

Python 3.10+ required. Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Required packages:
- `numpy>=1.24.0` - Numerical arrays
- `scipy>=1.10.0` - Scientific computing (eigen, ODE solvers, optimization)
- `matplotlib>=3.7.0` - Plotting
- `streamlit>=1.28.0` - Web UI
- `plotly>=5.17.0` - Interactive plots (optional)
- `pyyaml>=6.0` - YAML parameter files
- `networkx>=3.1` - Graph visualization (optional)
- `pytest>=7.4.0` - Testing framework

## Installation

### Option 1: Virtual Environment (Recommended)

```bash
# Windows PowerShell
cd project-vib-sim
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Linux/Mac
cd project-vib-sim
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option 2: Docker (Future)

```bash
# Dockerfile can be added for containerized deployment
```

## Usage

### 1. Run Smoke Test (CLI)

The smoke test builds a two-motor system, runs analysis, optimizes mounts, and saves plots:

```bash
python main.py --omega-max 500 --save-dir data/outputs
```

**Expected Output:**
```
SMOKE TEST: Two Isolated Vibrating Systems
Building two-motor system...
System assembled: 3 DOFs

[1/5] Running eigen analysis...
  Natural frequencies: 7.84 Hz, 1363.45 Hz, 2111.23 Hz

[2/5] Computing FRF (0-500 Hz)...
  Peak FRF magnitude Motor1->Motor2: 2.34e-05 at 7.84 Hz

[3/5] Running time-domain simulation...
  Simulation complete: 2000 time steps

[4/5] Running SVIME optimization...
  Optimized coupling pair: (0, 1)
  Peak reduction: 15.23%

[5/5] Generating plots...
✓ All outputs saved to: data/outputs
```

**Generated Files:**
- `system_schematic.png` - System layout diagram
- `mode_1.png`, `mode_2.png`, `mode_3.png` - Mode shapes
- `frf_motor1_to_motor2.png` - Frequency response
- `operational_response.png` - Time-domain response
- `smoke_test_results.npz` - Numerical data

### 2. Launch Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

The UI provides:
- **System Schematic** - Visual diagram of two isolated systems
- **Parameter Sliders** - Adjust mass, stiffness, damping for each component
- **Modal Analysis** - Natural frequencies and mode shapes
- **FRF Plots** - Frequency response functions with magnitude and phase
- **Time-Domain Simulation** - Operational response with force inputs
- **Animated System State** - Real-time visualization with time slider
- **SVIME Optimization** - One-click optimization to reduce coupling
- **Technical Diagram** - Displays diagram from `data/` or `assets/` folder

**Features:**
- Interactive parameter adjustment
- Real-time plot updates
- Before/after optimization comparison
- Download buttons for plots and optimized parameters

### 3. Run Jupyter Notebook

```bash
jupyter notebook notebooks/demo_two_motors.ipynb
```

The notebook demonstrates:
- System assembly
- Eigen analysis
- FRF computation
- Time-domain simulation
- SVIME optimization

### 4. Load Parameters from YAML

```python
from utils.io_utils import load_params_yaml
from models.component_builder import build_two_motor_system

params = load_params_yaml('data/example_params.yaml')
system = build_two_motor_system(
    params['components']['motor1'],
    params['components']['motor2'],
    params['components']['frame'],
    params['beam_elements']['connection1']
)
```

### 5. Run Tests

```bash
pytest tests/ -v
```

Tests cover:
- Mass-spring-damper models
- Line-body elements
- Eigen solver
- FRF computation

## Technical Details

### System Assembly

Components are assembled into global matrices:

```python
assembly = SystemAssembly(node_count=3)
assembly.add_component(MassSpringDamper(m, k, c), node_idx)
assembly.add_line_body(LineBody(area, length, E, rho), node_i, node_j)
M, C, K = assembly.finalize_system()
```

### Eigen Analysis

```python
nat_freqs, modes = solve_eigen(M, K)
# Returns: natural frequencies [rad/s], mode shapes [normalized]
```

### FRF Computation

```python
omega = np.linspace(0, 500*2*np.pi, 1000)  # Frequency grid [rad/s]
mag, phase = compute_frf(M, C, K, omega, dof_in=0, dof_out=1)
# Returns: magnitude and phase [deg]
```

### Time-Domain Simulation

```python
loads = {0: sine_force(amplitude=100.0, frequency=5.0)}
t_eval = np.linspace(0, 5, 2000)
sol = simulate_forced_response(M, C, K, (0, 5), loads, t_eval=t_eval)
# Returns: ODE solution with displacement history
```

### SVIME Optimization

```python
optimizer = SVIMEOptimizer(M, C, K, adjustable_dofs=[0, 1], omega_grid=omega)
result = optimizer.optimize()
# Returns: optimized parameters, improvement metrics, updated matrices
```

## File Formats

### YAML Parameter File

See `data/example_params.yaml` for structure:
- `components`: Motor and frame parameters (mass, stiffness, damping)
- `beam_elements`: Line-body properties (area, length, E, rho)
- `mounts`: Isolation mount parameters
- `analysis`: Frequency range and time settings
- `optimization`: SVIME configuration

### NPZ Data Files

Saved matrices and results:
- `M`, `C`, `K`: Global matrices
- `nat_freqs`, `modes`: Eigen results
- `frf_mag`, `frf_phase`, `frf_freq`: FRF data
- `time`, `response`: Time-domain results
- `svime_result`: Optimization results

## License

This project uses open-source libraries only. Code is provided for educational and research purposes.

## Contributing

1. Follow PEP 8 style guidelines
2. Add docstrings to all functions
3. Include unit tests for new features
4. Update README with new functionality

## Troubleshooting

**Import errors:**
- Ensure project root is in Python path
- Check all dependencies are installed: `pip list`

**Streamlit caching issues:**
- Clear cache: `streamlit cache clear`
- Restart Streamlit server

**Matrix assembly errors:**
- Verify node indices are within `node_count`
- Check component parameters are positive

**Optimization not converging:**
- Adjust bounds in SVIME optimizer
- Increase population size or generations
- Check system is not singular (det(K) ≠ 0)




