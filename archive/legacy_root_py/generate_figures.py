#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Figures for Quantum GR Report
Creates all the required charts and diagrams
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_images_directory():
    """Create images directory"""
    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)
    return images_dir

def figure_2_curvature_distribution():
    """Figure 2: Curvature Distribution Map"""
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # Curvature field: R(x,y) = -2(a+b)/(1+ax²+by²)
    a, b = 0.8, 1.2
    R = -2 * (a + b) / (1 + a * X**2 + b * Y**2)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.contourf(X, Y, R, levels=20, cmap='viridis')
    ax.contour(X, Y, R, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('曲率分布圖 (Curvature Distribution)', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('曲率 R(x,y)', fontsize=12)
    
    # Add center value annotation
    ax.text(0, 0, 'R(0,0) = -4.0', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('images/figure_2_curvature.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figure_2_curvature.png")

def figure_3_energy_density():
    """Figure 3: Energy Density Distribution"""
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # Energy density: ρ(x,y) = |R(x,y)|/(8πG)
    G = 6.67e-11
    a, b = 0.8, 1.2
    R = -2 * (a + b) / (1 + a * X**2 + b * Y**2)
    rho = np.abs(R) / (8 * np.pi * G)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap with log scale
    im = ax.contourf(X, Y, np.log10(rho), levels=20, cmap='plasma')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('能量密度分布 (Energy Density Distribution)', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(ρ) [J/m³]', fontsize=12)
    
    # Add center value annotation
    center_rho = np.abs(-4.0) / (8 * np.pi * G)
    ax.text(0, 0, f'ρ(0,0) = {center_rho:.2e} J/m³', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('images/figure_3_energy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figure_3_energy.png")

def figure_4_convergence_curve():
    """Figure 4: Loss Function Convergence Curve"""
    epochs = np.arange(0, 121)
    
    # Simulated convergence curves
    L_cov = 0.1 * np.exp(-epochs/30) + 1e-5 + 0.01 * np.random.normal(0, 1, len(epochs)) * np.exp(-epochs/20)
    L_energy = 0.07 * np.exp(-epochs/40) + 8e-6 + 0.008 * np.random.normal(0, 1, len(epochs)) * np.exp(-epochs/25)
    L_total = L_cov + L_energy
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot curves
    ax.semilogy(epochs, L_cov, 'b-', linewidth=2, label='L_cov (協變性損失)')
    ax.semilogy(epochs, L_energy, 'orange', linewidth=2, label='L_energy (能量守恆損失)')
    ax.semilogy(epochs, L_total, 'k-', linewidth=2, label='L_total (總損失)')
    
    # Add convergence point
    ax.axvline(x=80, color='red', linestyle='--', alpha=0.7, label='收斂點 (80 epochs)')
    
    ax.set_xlabel('訓練週期 (Epoch)', fontsize=12)
    ax.set_ylabel('損失值 (Loss)', fontsize=12)
    ax.set_title('多目標損失函數收斂曲線 (Multi-objective Loss Function Convergence)', fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 120)
    
    plt.tight_layout()
    plt.savefig('images/figure_4_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figure_4_convergence.png")

def figure_5_architecture_mapping():
    """Figure 5: Model Layer and Theoretical Layer Mapping"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define positions and labels
    positions = {
        'Encoder': (1, 3),
        'Fusion': (2.5, 2.5),
        'Reasoner': (4, 1.5),
        'Feedback': (1, 1),
        'g_munu': (0.5, 3),
        'S_ent': (3, 2.5),
        'T_munu': (4.5, 1.5),
        'Omega': (0.5, 1)
    }
    
    # Colors
    model_color = 'lightblue'
    theory_color = 'lightgreen'
    
    # Draw nodes
    for name, (x, y) in positions.items():
        if name in ['Encoder', 'Fusion', 'Reasoner', 'Feedback']:
            color = model_color
            fontsize = 10
        else:
            color = theory_color
            fontsize = 9
        
        # Create rounded rectangle
        bbox = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6,
                             boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(bbox)
        
        # Add text
        ax.text(x, y, name, ha='center', va='center', fontsize=fontsize, fontweight='bold')
    
    # Draw connections
    connections = [
        ('g_munu', 'Encoder'),
        ('Encoder', 'Fusion'),
        ('S_ent', 'Fusion'),
        ('Fusion', 'Reasoner'),
        ('T_munu', 'Reasoner'),
        ('Feedback', 'Omega'),
        ('Reasoner', 'Feedback')
    ]
    
    for start, end in connections:
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.7)
    
    ax.set_xlim(0, 5)
    ax.set_ylim(0.5, 3.5)
    ax.set_title('模型層與理論層對映圖 (Model Layer and Theoretical Layer Mapping)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=model_color, label='模型層 (Model Layer)'),
        Patch(facecolor=theory_color, label='理論層 (Theoretical Layer)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('images/figure_5_mapping.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figure_5_mapping.png")

def figure_6_quantum_correction():
    """Figure 6: Quantum Correction Term Distribution"""
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # Quantum correction term: Q_xx = λ(∇_x S_ent)²
    lambda_val = 0.05
    a, b = 0.8, 1.2
    
    # Entropy field gradient
    grad_x = 2 * a * X / (1 + a * X**2 + b * Y**2)
    grad_y = 2 * b * Y / (1 + a * X**2 + b * Y**2)
    
    Q_xx = lambda_val * grad_x**2
    Q_yy = lambda_val * grad_y**2
    Tr_Q = Q_xx + Q_yy
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Q_xx distribution
    im1 = ax1.contourf(X, Y, Q_xx, levels=20, cmap='RdBu_r')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('量子修正項 Q_{xx}', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Q_{xx}', fontsize=10)
    
    # Tr(Q) distribution
    im2 = ax2.contourf(X, Y, Tr_Q, levels=20, cmap='viridis')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('量子修正跡 Tr(Q)', fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Tr(Q)', fontsize=10)
    
    plt.suptitle('量子修正項分布 (Quantum Correction Term Distribution)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/figure_6_quantum_correction.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figure_6_quantum_correction.png")

def figure_7_observer_field():
    """Figure 7: Observer Field Effect"""
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # Observer field: Ω(x,y) = sin(x)cos(y)
    Omega = np.sin(X) * np.cos(Y)
    
    # Observer field effect: □Ω = κ(∇S_ent)²
    kappa = 0.4
    a, b = 0.8, 1.2
    grad_S_ent_sq = (2*a*X/(1+a*X**2+b*Y**2))**2 + (2*b*Y/(1+a*X**2+b*Y**2))**2
    Box_Omega = kappa * grad_S_ent_sq
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Entropy field
    S_ent = np.log(1 + a*X**2 + b*Y**2)
    im1 = ax1.contourf(X, Y, S_ent, levels=20, cmap='viridis')
    ax1.set_xlabel('x', fontsize=10)
    ax1.set_ylabel('y', fontsize=10)
    ax1.set_title('熵場 S_ent', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('S_ent', fontsize=10)
    
    # Observer field
    im2 = ax2.contourf(X, Y, Omega, levels=20, cmap='RdBu')
    ax2.set_xlabel('x', fontsize=10)
    ax2.set_ylabel('y', fontsize=10)
    ax2.set_title('觀測者場 Ω', fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Ω', fontsize=10)
    
    # Observer field effect
    im3 = ax3.contourf(X, Y, Box_Omega, levels=20, cmap='magma')
    ax3.set_xlabel('x', fontsize=10)
    ax3.set_ylabel('y', fontsize=10)
    ax3.set_title('觀測者場效應 □Ω', fontsize=12, fontweight='bold')
    ax3.set_aspect('equal')
    
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('□Ω', fontsize=10)
    
    plt.suptitle('觀測者場效應 (Observer Field Effect)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/figure_7_observer_field.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figure_7_observer_field.png")

def figure_8_hierarchy():
    """Figure 8: Theoretical Hierarchical Structure"""
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Define hierarchy levels
    levels = [
        ('基礎物理', '量子力學 + 廣義相對論', 'red'),
        ('QIGG', '量子資訊幾何引力', 'lightblue'),
        ('QHUT', '量子全像統一理論', 'lightgreen'),
        ('HAT', '全像自生成理論', 'yellow'),
        ('CSC', '完備自洽宇宙理論', 'lightpink')
    ]
    
    y_positions = np.linspace(4, 0, len(levels))
    
    for i, (name, description, color) in enumerate(levels):
        y = y_positions[i]
        
        # Main node
        circle = plt.Circle((1, y), 0.4, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(1, y, name, ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Description box
        rect = FancyBboxPatch((1.8, y-0.2), 2.5, 0.4,
                             boxstyle="round,pad=0.05",
                             facecolor='white', edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(3.05, y, description, ha='center', va='center', fontsize=10)
        
        # Connection line (except for the last one)
        if i < len(levels) - 1:
            ax.plot([1, 1], [y-0.4, y_positions[i+1]+0.4], 'k-', linewidth=3)
    
    ax.set_xlim(0, 6)
    ax.set_ylim(-0.5, 4.5)
    ax.set_title('理論層次結構 (Theoretical Hierarchical Structure)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('images/figure_8_hierarchy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figure_8_hierarchy.png")

def main():
    """Generate all figures"""
    print("=" * 60)
    print("Quantum GR Report - Figure Generator")
    print("=" * 60)
    print()
    
    # Create images directory
    images_dir = create_images_directory()
    
    # Generate all figures
    print("Generating figures...")
    print()
    
    try:
        figure_2_curvature_distribution()
        figure_3_energy_density()
        figure_4_convergence_curve()
        figure_5_architecture_mapping()
        figure_6_quantum_correction()
        figure_7_observer_field()
        figure_8_hierarchy()
        
        print()
        print("All figures generated successfully!")
        print(f"Images saved in: {images_dir}")
        print()
        print("Now you can:")
        print("1. Run the Word generator to create the document")
        print("2. Open the Word document")
        print("3. The images will be automatically inserted")
        
    except Exception as e:
        print(f"Error generating figures: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")

if __name__ == '__main__':
    main()
