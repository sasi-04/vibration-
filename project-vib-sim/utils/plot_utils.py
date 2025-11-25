"""Plot utilities shared across CLI and Streamlit app."""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np


def plot_frf(omega: np.ndarray, magnitude: np.ndarray, phase: np.ndarray, title: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    axes[0].plot(omega, magnitude)
    axes[0].set_ylabel("|H(ω)|")
    axes[0].grid(True)
    axes[1].plot(omega, phase)
    axes[1].set_xlabel("ω [rad/s]")
    axes[1].set_ylabel("∠H(ω) [deg]")
    axes[1].grid(True)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_modes(modes: np.ndarray, mode_index: int) -> plt.Figure:
    shape = modes[:, mode_index]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(shape, marker="o")
    ax.set_title(f"Mode {mode_index + 1}")
    ax.set_xlabel("DOF")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_system_schematic(m1: float, k1: float, c1: float, m2: float, k2: float, c2: float,
                         mf: float, kf: float, cf: float, name1: str = "System A", 
                         name2: str = "System B") -> plt.Figure:
    """Create a schematic diagram of two isolated vibrating systems."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 9)
    ax.axis('off')
    
    # Frame/base (bottom)
    frame_y = 1.0
    frame_width = 10.0
    frame_height = 0.8
    frame = Rectangle((0, frame_y), frame_width, frame_height, 
                     facecolor='gray', edgecolor='black', linewidth=2, alpha=0.6)
    ax.add_patch(frame)
    ax.text(5, frame_y + frame_height/2, f'Frame (m={mf:.1f} kg)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # System A (left)
    sys1_x = 2.0
    sys1_y = frame_y + frame_height + 0.3
    
    # Mass 1
    mass1_size = 0.6 + 0.3 * (m1 / 200.0)  # Scale by mass
    mass1 = Circle((sys1_x, sys1_y + 2.5), mass1_size/2, 
                  facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(mass1)
    ax.text(sys1_x, sys1_y + 2.5, f'm₁\n{m1:.1f} kg', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Spring 1
    spring1_y1 = sys1_y + 1.0
    spring1_y2 = sys1_y + 2.5 - mass1_size/2
    ax.plot([sys1_x, sys1_x], [spring1_y1, spring1_y2], 'k-', linewidth=2)
    # Spring coils
    for i in range(5):
        y_coil = spring1_y1 + (spring1_y2 - spring1_y1) * (i / 4.0)
        ax.plot([sys1_x - 0.15, sys1_x + 0.15], [y_coil, y_coil], 'k-', linewidth=1.5)
    ax.text(sys1_x + 0.5, (spring1_y1 + spring1_y2)/2, f'k₁={k1/1e5:.1f}×10⁵', 
            fontsize=8, rotation=90, va='center')
    
    # Damper 1
    damper1_x = sys1_x + 0.4
    ax.plot([damper1_x, damper1_x], [spring1_y1, spring1_y2], 'r-', linewidth=2)
    ax.plot([damper1_x - 0.1, damper1_x + 0.1], [spring1_y1, spring1_y1], 'r-', linewidth=2)
    ax.plot([damper1_x - 0.1, damper1_x + 0.1], [spring1_y2, spring1_y2], 'r-', linewidth=2)
    ax.text(damper1_x + 0.3, (spring1_y1 + spring1_y2)/2, f'c₁={c1:.0f}', 
            fontsize=8, rotation=90, va='center', color='red')
    
    # Connection to frame
    ax.plot([sys1_x, sys1_x], [frame_y + frame_height, spring1_y1], 'k--', linewidth=1, alpha=0.5)
    
    # System label
    ax.text(sys1_x, sys1_y + 3.8, name1, ha='center', fontsize=11, weight='bold', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # System B (right)
    sys2_x = 8.0
    sys2_y = frame_y + frame_height + 0.3
    
    # Mass 2
    mass2_size = 0.6 + 0.3 * (m2 / 200.0)
    mass2 = Circle((sys2_x, sys2_y + 2.5), mass2_size/2, 
                  facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(mass2)
    ax.text(sys2_x, sys2_y + 2.5, f'm₂\n{m2:.1f} kg', 
            ha='center', va='center', fontsize=9, weight='bold')
    
    # Spring 2
    spring2_y1 = sys2_y + 1.0
    spring2_y2 = sys2_y + 2.5 - mass2_size/2
    ax.plot([sys2_x, sys2_x], [spring2_y1, spring2_y2], 'k-', linewidth=2)
    # Spring coils
    for i in range(5):
        y_coil = spring2_y1 + (spring2_y2 - spring2_y1) * (i / 4.0)
        ax.plot([sys2_x - 0.15, sys2_x + 0.15], [y_coil, y_coil], 'k-', linewidth=1.5)
    ax.text(sys2_x + 0.5, (spring2_y1 + spring2_y2)/2, f'k₂={k2/1e5:.1f}×10⁵', 
            fontsize=8, rotation=90, va='center')
    
    # Damper 2
    damper2_x = sys2_x + 0.4
    ax.plot([damper2_x, damper2_x], [spring2_y1, spring2_y2], 'r-', linewidth=2)
    ax.plot([damper2_x - 0.1, damper2_x + 0.1], [spring2_y1, spring2_y1], 'r-', linewidth=2)
    ax.plot([damper2_x - 0.1, damper2_x + 0.1], [spring2_y2, spring2_y2], 'r-', linewidth=2)
    ax.text(damper2_x + 0.3, (spring2_y1 + spring2_y2)/2, f'c₂={c2:.0f}', 
            fontsize=8, rotation=90, va='center', color='red')
    
    # Connection to frame
    ax.plot([sys2_x, sys2_x], [frame_y + frame_height, spring2_y1], 'k--', linewidth=1, alpha=0.5)
    
    # System label
    ax.text(sys2_x, sys2_y + 3.8, name2, ha='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # Coupling path (through frame)
    coupling_y = frame_y + frame_height + 0.1
    ax.annotate('', xy=(sys1_x, coupling_y), xytext=(sys2_x, coupling_y),
               arrowprops=dict(arrowstyle='<->', color='green', lw=2, alpha=0.7))
    ax.text((sys1_x + sys2_x)/2, coupling_y + 0.2, 'Coupling via Frame', 
            ha='center', fontsize=9, color='green', weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax.set_title('Two Isolated Vibrating Systems - Schematic', fontsize=14, weight='bold', pad=20)
    fig.tight_layout()
    return fig


def plot_operational_response(time: np.ndarray, response1: np.ndarray, response2: np.ndarray,
                             name1: str = "System A", name2: str = "System B") -> plt.Figure:
    """Plot operational response of both systems side by side."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # System A response
    axes[0].plot(time, response1, 'b-', linewidth=2, label=name1)
    axes[0].set_ylabel(f'{name1} Displacement [m]', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    axes[0].set_title(f'{name1} Operational Response', fontsize=12, weight='bold')
    
    # System B response
    axes[1].plot(time, response2, 'r-', linewidth=2, label=name2)
    axes[1].set_ylabel(f'{name2} Displacement [m]', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    axes[1].set_title(f'{name2} Operational Response', fontsize=12, weight='bold')
    
    # Comparison overlay
    axes[2].plot(time, response1, 'b-', linewidth=2, label=name1, alpha=0.7)
    axes[2].plot(time, response2, 'r-', linewidth=2, label=name2, alpha=0.7)
    axes[2].set_ylabel('Displacement [m]', fontsize=11)
    axes[2].set_xlabel('Time [s]', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')
    axes[2].set_title('Comparison: Both Systems', fontsize=12, weight='bold')
    
    fig.suptitle('Operational Behavior of Isolated Systems', fontsize=14, weight='bold', y=0.995)
    fig.tight_layout()
    return fig


def plot_animated_system_state(time: np.ndarray, response1: np.ndarray, response2: np.ndarray,
                               t_current: float, m1: float, m2: float, mf: float,
                               response_frame=None) -> plt.Figure:
    """Plot current state of systems at a specific time (for animation frames)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-2, 6)
    ax.axis('off')
    
    # Find current index
    idx = np.argmin(np.abs(time - t_current))
    if idx >= len(response1):
        idx = len(response1) - 1
    
    # Scale displacements for visualization (amplify for visibility)
    scale = 2.0
    disp1 = response1[idx] * scale
    disp2 = response2[idx] * scale
    dispf = response_frame[idx] * scale if response_frame is not None and len(response_frame) > idx else 0
    
    frame_y = 1.0 + dispf
    frame_width = 10.0
    frame_height = 0.8
    
    # Frame
    frame = Rectangle((0, frame_y), frame_width, frame_height, 
                     facecolor='gray', edgecolor='black', linewidth=2, alpha=0.6)
    ax.add_patch(frame)
    
    # System A
    sys1_x = 2.0
    sys1_y = frame_y + frame_height + 0.3 + disp1
    
    mass1_size = 0.6 + 0.3 * (m1 / 200.0)
    mass1 = Circle((sys1_x, sys1_y + 2.5), mass1_size/2, 
                  facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(mass1)
    ax.text(sys1_x, sys1_y + 2.5, f'm₁', ha='center', va='center', fontsize=9, weight='bold')
    
    # Spring 1 (deformed)
    spring1_y1 = frame_y + frame_height
    spring1_y2 = sys1_y + 2.5 - mass1_size/2
    ax.plot([sys1_x, sys1_x], [spring1_y1, spring1_y2], 'k-', linewidth=2)
    for i in range(5):
        y_coil = spring1_y1 + (spring1_y2 - spring1_y1) * (i / 4.0)
        ax.plot([sys1_x - 0.15, sys1_x + 0.15], [y_coil, y_coil], 'k-', linewidth=1.5)
    
    # System B
    sys2_x = 8.0
    sys2_y = frame_y + frame_height + 0.3 + disp2
    
    mass2_size = 0.6 + 0.3 * (m2 / 200.0)
    mass2 = Circle((sys2_x, sys2_y + 2.5), mass2_size/2, 
                  facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(mass2)
    ax.text(sys2_x, sys2_y + 2.5, f'm₂', ha='center', va='center', fontsize=9, weight='bold')
    
    # Spring 2 (deformed)
    spring2_y1 = frame_y + frame_height
    spring2_y2 = sys2_y + 2.5 - mass2_size/2
    ax.plot([sys2_x, sys2_x], [spring2_y1, spring2_y2], 'k-', linewidth=2)
    for i in range(5):
        y_coil = spring2_y1 + (spring2_y2 - spring2_y1) * (i / 4.0)
        ax.plot([sys2_x - 0.15, sys2_x + 0.15], [y_coil, y_coil], 'k-', linewidth=1.5)
    
    # Displacement indicators
    ax.annotate(f'Δ₁={response1[idx]*1e6:.2f} μm', 
               xy=(sys1_x, sys1_y + 2.5), xytext=(sys1_x - 1.5, sys1_y + 3.5),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2),
               fontsize=10, weight='bold', color='blue',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.annotate(f'Δ₂={response2[idx]*1e6:.2f} μm', 
               xy=(sys2_x, sys2_y + 2.5), xytext=(sys2_x + 1.5, sys2_y + 3.5),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=10, weight='bold', color='red',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(f'System State at t = {t_current:.3f} s', fontsize=14, weight='bold', pad=20)
    fig.tight_layout()
    return fig




