"""Streamlit UI for Model-Based Design of isolated vibrating systems."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path FIRST, before any other imports
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import standard libraries
import numpy as np
import streamlit as st

# Then import project modules
from analysis.eigen_analysis import solve_eigen
from analysis.frf_analysis import compute_frf
from analysis.time_analysis import simulate_forced_response, sine_force
from assembly.system_assembly import SystemAssembly
from models.mass_spring_damper import MassSpringDamper
from models.line_body import LineBody
from optimization.svime_optimizer import SVIMEOptimizer
from utils.plot_utils import plot_frf, plot_modes, plot_system_schematic, plot_operational_response, plot_animated_system_state
from utils.io_utils import load_params_yaml, get_diagram_path, load_diagram_image

st.set_page_config(page_title="SVIME - Vibration Mitigation", layout="wide")
st.title("Model-Based Design for Two Isolated Vibrating Systems")

# Try to load and display technical diagram
diagram_path = get_diagram_path()
if diagram_path.exists():
    try:
        diagram_bytes = load_diagram_image(diagram_path)
        if diagram_bytes:
            st.sidebar.image(diagram_bytes, caption="Technical Description", use_container_width=True)
    except Exception as e:
        st.sidebar.warning(f"Could not load diagram: {e}")
else:
    st.sidebar.info("Technical diagram not found. Place it in data/ or assets/ folder.")

with st.sidebar:
    st.header("Component Parameters")
    mass1 = st.number_input("Subsystem A mass [kg]", 10.0, 200.0, 50.0, 1.0)
    k1 = st.number_input("Subsystem A stiffness [N/m]", 1e4, 5e5, 2e5, 1e4)
    c1 = st.number_input("Subsystem A damping [Ns/m]", 10.0, 2e3, 400.0, 10.0)

    mass2 = st.number_input("Subsystem B mass [kg]", 10.0, 200.0, 40.0, 1.0)
    k2 = st.number_input("Subsystem B stiffness [N/m]", 1e4, 5e5, 1.5e5, 1e4)
    c2 = st.number_input("Subsystem B damping [Ns/m]", 10.0, 2e3, 350.0, 10.0)

    frame_mass = st.number_input("Frame mass [kg]", 20.0, 300.0, 80.0, 1.0)
    frame_k = st.number_input("Frame stiffness [N/m]", 1e4, 6e5, 2.5e5, 1e4)
    frame_c = st.number_input("Frame damping [Ns/m]", 10.0, 3e3, 600.0, 10.0)

    st.header("Line Body")
    area = st.number_input("Area [m²]", 0.001, 0.05, 0.01, 0.001, format="%.4f")
    length = st.number_input("Length [m]", 0.1, 2.0, 0.5, 0.1)
    E = st.number_input("Young's modulus [Pa]", 1e9, 3e11, 2e11, 1e9)
    rho = st.number_input("Density [kg/m³]", 500.0, 9000.0, 7800.0, 100.0)
    omega_max = st.slider("FRF ω max [rad/s]", 200.0, 2000.0, 800.0, 50.0)


@st.cache_data
def assemble(m1, k1, c1, m2, k2, c2, mf, kf, cf, area, length, E, rho):
    assembly = SystemAssembly(node_count=3)
    assembly.add_component(MassSpringDamper(m1, k1, c1, name="A"), 0)
    assembly.add_component(MassSpringDamper(m2, k2, c2, name="B"), 1)
    assembly.add_component(MassSpringDamper(mf, kf, cf, name="Frame"), 2)
    lb = LineBody(area, length, E, rho)
    assembly.add_line_body(lb, 0, 2)
    assembly.add_line_body(lb, 1, 2)
    return assembly.finalize_system()


@st.cache_data
def compute_eigen(_M, _K):
    return solve_eigen(_M, _K)


@st.cache_data
def compute_frf_data(_M, _C, _K, _omega, dof_in, dof_out):
    return compute_frf(_M, _C, _K, _omega, dof_in=dof_in, dof_out=dof_out)


# Note: compute_time_response removed - call simulate_forced_response directly
# to avoid Streamlit caching issues with function objects in loads_dict


M, C, K = assemble(mass1, k1, c1, mass2, k2, c2, frame_mass, frame_k, frame_c, area, length, E, rho)
omega = np.linspace(1.0, omega_max, 500)

# System Schematic
st.subheader("📐 System Schematic")
schematic_fig = plot_system_schematic(mass1, k1, c1, mass2, k2, c2, frame_mass, frame_k, frame_c, 
                                      "System A", "System B")
st.pyplot(schematic_fig)

st.divider()

nat_freqs, modes = compute_eigen(M, K)
st.subheader("Modal Analysis")
st.write("Natural frequencies [rad/s]:", np.round(nat_freqs, 2))
mode_idx = st.selectbox("Mode to visualize", range(len(nat_freqs)), format_func=lambda i: f"Mode {i+1}")
mode_fig = plot_modes(modes, mode_idx)
st.pyplot(mode_fig)

mag, phase = compute_frf_data(M, C, K, omega, 0, 1)
frf_fig = plot_frf(omega, mag, phase, "Motor → Pump FRF")
st.pyplot(frf_fig)

st.subheader("⏱️ Operational Response - Time Domain")
st.write("**Two Isolated Systems Under Operational Loads**")

# Force inputs
col1, col2 = st.columns(2)
with col1:
    force1_amp = st.number_input("Force on System A [N]", 0.0, 500.0, 100.0, 10.0)
    force1_freq = st.number_input("Force A frequency [Hz]", 0.1, 20.0, 5.0, 0.5)
with col2:
    force2_amp = st.number_input("Force on System B [N]", 0.0, 500.0, 0.0, 10.0)
    force2_freq = st.number_input("Force B frequency [Hz]", 0.1, 20.0, 3.0, 0.5)

sim_time = st.slider("Simulation time [s]", 0.5, 10.0, 3.0, 0.5)
t_eval = np.linspace(0, sim_time, int(sim_time * 500))

loads = {}
if force1_amp > 0:
    loads[0] = sine_force(force1_amp, force1_freq)
if force2_amp > 0:
    loads[1] = sine_force(force2_amp, force2_freq)

if loads:
    # Call directly to avoid caching issues with function objects
    sol = simulate_forced_response(M, C, K, (0, sim_time), loads, t_eval=t_eval)
    
    # Operational response plot
    op_fig = plot_operational_response(t_eval, sol.y[0, :], sol.y[1, :], "System A", "System B")
    st.pyplot(op_fig)
    
    # Animated state visualization
    st.subheader("🎬 Animated System State")
    time_slider = st.slider("Time [s]", 0.0, float(sim_time), 0.0, 0.01, key="time_slider")
    frame_response = sol.y[2, :] if sol.y.shape[0] > 2 else None
    frame_fig = plot_animated_system_state(t_eval, sol.y[0, :], sol.y[1, :], 
                                          time_slider, mass1, mass2, frame_mass,
                                          response_frame=frame_response)
    st.pyplot(frame_fig)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("System A Max Displacement", f"{np.max(np.abs(sol.y[0, :]))*1e6:.2f} μm")
    with col2:
        st.metric("System B Max Displacement", f"{np.max(np.abs(sol.y[1, :]))*1e6:.2f} μm")
    with col3:
        coupling_ratio = np.max(np.abs(sol.y[1, :])) / (np.max(np.abs(sol.y[0, :])) + 1e-10)
        st.metric("Coupling Ratio (B/A)", f"{coupling_ratio:.3f}")
else:
    st.info("⚠️ Please set at least one force amplitude > 0 to see operational response.")

st.subheader("Smart Vibration Interaction Mitigation Engine (SVIME)")
if st.button("Run SVIME optimization"):
    optimizer = SVIMEOptimizer(M, C, K, adjustable_dofs=[0, 1], omega_grid=omega)
    result = optimizer.optimize()
    st.success(
        f"Coupling pair {result['pair']} improved by {result['improvement_pct']:.2f}% "
        f"(peak {result['baseline_peak']:.2f} → {result['optimized_peak']:.2f})"
    )
    st.json({"best_params": result["best_params"].tolist()})
else:
    st.info("Press the button to minimize coupling via SVIME.")



