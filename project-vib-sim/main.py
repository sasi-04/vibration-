"""Entry point for model-based design demo with SVIME.
Smoke test: constructs two motors in beam enclosure, runs analysis, optimizes mounts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.eigen_analysis import solve_eigen
from analysis.frf_analysis import compute_frf
from analysis.time_analysis import random_force, simulate_forced_response, sine_force
from assembly.system_assembly import SystemAssembly
from models.mass_spring_damper import MassSpringDamper
from models.line_body import LineBody
from optimization.svime_optimizer import SVIMEOptimizer
from utils.plot_utils import plot_frf, plot_modes, plot_system_schematic, plot_operational_response


def build_two_motor_system() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a system with two motors inside a rectangular beam enclosure.
    
    Returns:
        Tuple of (M, C, K) global matrices
    """
    print("Building two-motor system...")
    assembly = SystemAssembly(node_count=3)
    
    # Motor 1 (left)
    motor1 = MassSpringDamper(50.0, 2e5, 500.0, name="Motor1")
    assembly.add_component(motor1, 0)
    
    # Motor 2 (right)
    motor2 = MassSpringDamper(40.0, 1.5e5, 400.0, name="Motor2")
    assembly.add_component(motor2, 1)
    
    # Enclosure frame
    frame = MassSpringDamper(80.0, 2.5e5, 650.0, name="Frame")
    assembly.add_component(frame, 2)
    
    # Beam elements connecting motors to frame (simulating rectangular enclosure)
    beam1 = LineBody(area=0.01, length=0.5, youngs_modulus=2e11, density=7800)
    beam2 = LineBody(area=0.01, length=0.5, youngs_modulus=2.1e11, density=7800)
    
    assembly.add_line_body(beam1, 0, 2)  # Motor1 -> Frame
    assembly.add_line_body(beam2, 1, 2)  # Motor2 -> Frame
    
    M, C, K = assembly.finalize_system()
    print(f"System assembled: {M.shape[0]} DOFs")
    return M, C, K


def run_smoke_test(save_dir: Path, omega_max: float = 500.0) -> None:
    """Run complete smoke test: build system, analyze, optimize, save plots.
    
    Args:
        save_dir: Directory to save outputs
        omega_max: Maximum frequency for FRF analysis (Hz, converted to rad/s)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SMOKE TEST: Two Isolated Vibrating Systems")
    print("=" * 60)
    
    # 1. Build system
    M, C, K = build_two_motor_system()
    
    # 2. Eigen analysis
    print("\n[1/5] Running eigen analysis...")
    nat_freqs, modes = solve_eigen(M, K)
    nat_freqs_hz = nat_freqs / (2 * np.pi)
    print(f"  Natural frequencies: {nat_freqs_hz[0]:.2f} Hz, {nat_freqs_hz[1]:.2f} Hz, {nat_freqs_hz[2]:.2f} Hz")
    print(f"  First mode shape: {modes[:, 0]}")
    
    # 3. FRF analysis (0-500 Hz)
    print("\n[2/5] Computing FRF (0-500 Hz)...")
    omega_max_rad = omega_max * 2 * np.pi
    omega = np.linspace(1.0, omega_max_rad, 1000)
    mag, phase = compute_frf(M, C, K, omega, dof_in=0, dof_out=1)
    peak_idx = np.argmax(mag)
    peak_freq_hz = omega[peak_idx] / (2 * np.pi)
    print(f"  Peak FRF magnitude Motor1->Motor2: {np.max(mag):.2e} at {peak_freq_hz:.2f} Hz")
    
    # 4. Time-domain simulation
    print("\n[3/5] Running time-domain simulation...")
    loads = {
        0: sine_force(amplitude=100.0, frequency=5.0),  # Motor1 excitation
        1: random_force(seed=42, level=20.0)  # Motor2 random excitation
    }
    t_eval = np.linspace(0, 5, 2000)
    sol = simulate_forced_response(M, C, K, (0, 5), loads, t_eval=t_eval)
    print(f"  Simulation complete: {len(sol.t)} time steps")
    print(f"  Final displacement Motor1: {sol.y[0, -1]*1e6:.2f} μm")
    print(f"  Final displacement Motor2: {sol.y[1, -1]*1e6:.2f} μm")
    
    # 5. SVIME optimization
    print("\n[4/5] Running SVIME optimization for mount stiffness...")
    optimizer = SVIMEOptimizer(M, C, K, adjustable_dofs=[0, 1], omega_grid=omega)
    opt_result = optimizer.optimize()
    print(f"  Optimized coupling pair: {opt_result['pair']}")
    print(f"  Peak reduction: {opt_result['improvement_pct']:.2f}%")
    print(f"  Baseline peak: {opt_result['baseline_peak']:.2e}")
    print(f"  Optimized peak: {opt_result['optimized_peak']:.2e}")
    
    # 6. Generate and save plots
    print("\n[5/5] Generating plots...")
    
    # System schematic
    fig_schematic = plot_system_schematic(50.0, 2e5, 500.0, 40.0, 1.5e5, 400.0,
                                         80.0, 2.5e5, 650.0, "Motor1", "Motor2")
    fig_schematic.savefig(save_dir / "system_schematic.png", dpi=150, bbox_inches='tight')
    plt.close(fig_schematic)
    
    # Mode shapes
    for i in range(len(nat_freqs)):
        fig_mode = plot_modes(modes, i)
        fig_mode.savefig(save_dir / f"mode_{i+1}.png", dpi=150, bbox_inches='tight')
        plt.close(fig_mode)
    
    # FRF
    omega_hz = omega / (2 * np.pi)
    fig_frf = plot_frf(omega_hz, mag, phase, "FRF: Motor1 → Motor2")
    fig_frf.savefig(save_dir / "frf_motor1_to_motor2.png", dpi=150, bbox_inches='tight')
    plt.close(fig_frf)
    
    # Operational response
    fig_op = plot_operational_response(sol.t, sol.y[0, :], sol.y[1, :], "Motor1", "Motor2")
    fig_op.savefig(save_dir / "operational_response.png", dpi=150, bbox_inches='tight')
    plt.close(fig_op)
    
    # Save data
    np.savez(
        save_dir / "smoke_test_results.npz",
        nat_freqs=nat_freqs,
        modes=modes,
        frf_mag=mag,
        frf_phase=phase,
        frf_freq=omega,
        time=sol.t,
        response=sol.y,
        svime_result=opt_result,
    )
    
    print(f"\n✓ All outputs saved to: {save_dir}")
    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  - System DOFs: {M.shape[0]}")
    print(f"  - Natural frequencies: {nat_freqs_hz[0]:.1f}, {nat_freqs_hz[1]:.1f}, {nat_freqs_hz[2]:.1f} Hz")
    print(f"  - Peak FRF: {np.max(mag):.2e} at {peak_freq_hz:.1f} Hz")
    print(f"  - SVIME improvement: {opt_result['improvement_pct']:.2f}%")
    print(f"  - Plots saved: 5 files")
    print(f"  - Data saved: smoke_test_results.npz")


def run_demo(save_dir: Path, omega_max: float) -> None:
    """Legacy demo function - redirects to smoke test."""
    # omega_max is in rad/s, convert to Hz for smoke test
    omega_max_hz = omega_max / (2 * np.pi)
    run_smoke_test(save_dir, omega_max_hz)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model-Based Design for isolated systems.")
    parser.add_argument("--omega-max", type=float, default=800.0, help="Maximum ω for FRF grid.")
    parser.add_argument(
        "--save-dir", type=Path, default=Path("artifacts"), help="Directory for results."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    run_demo(args.save_dir, args.omega_max)


if __name__ == "__main__":
    main()


