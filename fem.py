import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def read_nodes(filename):
    return np.loadtxt(filename)

def read_elements(filename):
    elements = np.loadtxt(filename, dtype=int, delimiter=',')
    return (elements - 1).astype(int)

def read_bcs(filename):
    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    node_ids = data[:, 0].astype(int) - 1
    values = data[:, 1].astype(float)
    return node_ids, values

def plot_mesh_nodes_bcs(nodes_file, elements_file, bc_file, out_file):
    nodes = read_nodes(nodes_file)
    elements = read_elements(elements_file)
    bc_nodes, bc_values = read_bcs(bc_file)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for elem in elements:
        triangle = nodes[elem]
        tri = patches.Polygon(triangle, fill=False, edgecolor='blue', linewidth=1.5)
        ax.add_patch(tri)
    
    ax.plot(nodes[:, 0], nodes[:, 1], 'ko', markersize=8, label='Nodes')
    
    for i, (x, y) in enumerate(nodes):
        ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=10, color='black')
    
    ax.plot(nodes[bc_nodes, 0], nodes[bc_nodes, 1], 'ro', markersize=12, 
           label='BC Nodes', markerfacecolor='none', markeredgewidth=2)
    
    for node_id, value in zip(bc_nodes, bc_values):
        x, y = nodes[node_id]
        ax.annotate(f'T={value}°C', (x, y), xytext=(10, -15), 
                   textcoords='offset points', fontsize=9, color='red',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('FEM Mesh with Boundary Conditions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"Mesh plot saved to {out_file}")
    plt.close()

def calculate_element_stiffness_matrix(element_nodes, k=1.0, t=1.0):
    xy = np.asarray(element_nodes, dtype=float)
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]

    A = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    
    B_grad = np.vstack((b, c))
    
    Ke = (k * t / (4 * A)) * (B_grad.T @ B_grad)
    return Ke

def calculate_global_stiffness_matrix(nodes_file, elements_file, k=1.0, t=1.0):
    nodes = read_nodes(nodes_file)
    elements = read_elements(elements_file)
    
    N = nodes.shape[0]
    K = np.zeros((N, N), dtype=float)

    for elem in elements:
        xy = nodes[elem, :]
        Ke = calculate_element_stiffness_matrix(xy, k=k, t=t)
        
        for a_loc, a_glob in enumerate(elem):
            for b_loc, b_glob in enumerate(elem):
                K[a_glob, b_glob] += Ke[a_loc, b_loc]

    return K

def apply_dirichlet_bcs(K, F, bc_nodes, bc_values):
    K_mod = K.copy()
    F_mod = F.copy()
    
    bc_nodes_set = set(bc_nodes)
    
    for node_id, value in zip(bc_nodes, bc_values):
        for i in range(len(F_mod)):
            if i not in bc_nodes_set:
                F_mod[i] -= K_mod[i, node_id] * value
        
        K_mod[node_id, :] = 0.0
        K_mod[:, node_id] = 0.0
        K_mod[node_id, node_id] = 1.0
        F_mod[node_id] = value
    
    return K_mod, F_mod

def solve_fem(nodes_file, elements_file, bc_file, k=1.0, t=1.0, q_v=0.0):
    nodes = read_nodes(nodes_file)
    elements = read_elements(elements_file)
    bc_nodes, bc_values = read_bcs(bc_file)
    
    N = nodes.shape[0]
    
    print("Calculating global stiffness matrix...")
    K = calculate_global_stiffness_matrix(nodes_file, elements_file, k=k, t=t)
    
    F = np.full(N, q_v, dtype=float)
    
    print("Applying boundary conditions...")
    K_mod, F_mod = apply_dirichlet_bcs(K, F, bc_nodes, bc_values)
    
    print("Solving system of equations...")
    T = np.linalg.solve(K_mod, F_mod)

    return T, nodes, elements

def calculate_heat_flux(nodes, elements, T, k=1.0):
    heat_fluxes = []
    element_centers = []
    
    for elem in elements:
        xy = nodes[elem]
        temps = T[elem]
        
        x1, y1 = xy[0]
        x2, y2 = xy[1]
        x3, y3 = xy[2]
        
        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])
        
        A = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        
        dT_dx = (1 / (2 * A)) * np.dot(b, temps)
        dT_dy = (1 / (2 * A)) * np.dot(c, temps)
        
        jx = -k * dT_dx
        jy = -k * dT_dy
        
        heat_fluxes.append([jx, jy])
        
        center_x = np.mean(xy[:, 0])
        center_y = np.mean(xy[:, 1])
        element_centers.append([center_x, center_y])
    
    return np.array(heat_fluxes), np.array(element_centers)

def plot_results(nodes, elements, T, heat_fluxes, element_centers, out_file):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for elem in elements:
        triangle = nodes[elem]
        tri = patches.Polygon(triangle, fill=False, edgecolor='black', linewidth=0.5)
        ax1.add_patch(tri)
    
    tri_plot = ax1.tricontourf(nodes[:, 0], nodes[:, 1], T, levels=15, cmap='YlOrRd')
    plt.colorbar(tri_plot, ax=ax1, label='Temperature (°C)')
    
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_title('Temperature Distribution', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)
    
    for elem in elements:
        triangle = nodes[elem]
        tri = patches.Polygon(triangle, fill=False, edgecolor='black', linewidth=0.5)
        ax2.add_patch(tri)
    
    ax2.quiver(element_centers[:, 0], element_centers[:, 1],
              heat_fluxes[:, 0], heat_fluxes[:, 1],
              scale=50, width=0.005, color='blue', alpha=0.7)
    
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title('Heat Flux Vectors (j = -k∇T)', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"Results plot saved to {out_file}")
    plt.close()

def main():
    base_dir = Path(__file__).resolve().parent
    
    nodes_file = base_dir / "v8_motor_nodes_2D.txt"
    elements_file = base_dir / "v8_motor_triangles_2D.txt"
    bc_file = base_dir / "v8_motor_g_bc_hot_2D.txt"
    
    plot_mesh_nodes_bcs(nodes_file, elements_file, bc_file, str(base_dir / "v8_mesh_plot.png"))
    
    T, nodes, elements = solve_fem(nodes_file, elements_file, bc_file, k=1.0)
    
    print("\nNodal Temperatures:")
    print("-" * 30)
    for i, temp in enumerate(T):
        print(f"Node {i+1}: {temp:.2f} °C")
    
    heat_fluxes, element_centers = calculate_heat_flux(nodes, elements, T, k=1.0)
    
    print("\nHeat Flux per Element:")
    print("-" * 30)
    for i, (flux, center) in enumerate(zip(heat_fluxes, element_centers)):
        magnitude = np.linalg.norm(flux)
        print(f"Element {i+1}: jx={flux[0]:.3f}, jy={flux[1]:.3f}, |j|={magnitude:.3f} W/m²")
    
    plot_results(nodes, elements, T, heat_fluxes, element_centers, str(base_dir / "v8_results_plot.png"))

if __name__ == "__main__":
    main()