#FEM VERSION 1: EXAMPLE 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def read_nodes(filename):
    """Read nodes file. Expect two columns: x, y."""
    nodes = np.loadtxt(filename)
    return nodes

def read_elements(filename):
    """Read triangle elements. Expect three 1-based integer columns."""
    elements = np.loadtxt(filename, dtype=int)
    elements = elements - 1
    return elements.astype(int)

def read_bcs(filename):
    """Read Dirichlet BC file. Expect two columns: node_id, value."""
    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    node_ids = data[:, 0].astype(int) - 1
    values = data[:, 1].astype(float)
    return node_ids, values

def plot_mesh_nodes_bcs(nodes_file, elements_file, bc_file, out_file):
    """Plot nodes and elements with Dirichlet BCs."""
    nodes = read_nodes(nodes_file)
    elements = read_elements(elements_file)
    bc_nodes, bc_values = read_bcs(bc_file)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot elements
    for elem in elements:
        triangle = nodes[elem]
        tri = patches.Polygon(triangle, fill=False, edgecolor='blue', linewidth=1.5)
        ax.add_patch(tri)
    
    # Plot all nodes
    ax.plot(nodes[:, 0], nodes[:, 1], 'ko', markersize=8, label='Nodes')
    
    # Annotate node numbers
    for i, (x, y) in enumerate(nodes):
        ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=10, color='black')
    
    # Highlight BC nodes
    ax.plot(nodes[bc_nodes, 0], nodes[bc_nodes, 1], 'ro', markersize=12, 
           label='BC Nodes', markerfacecolor='none', markeredgewidth=2)
    
    # Annotate BC values
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
    """Calculate the 3x3 element stiffness matrix for a linear 3-node triangular element."""
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
    """
    Apply Dirichlet boundary conditions using the modification method.
    Steps from your notes:
    A. Zero entire row
    B. Zero the column (but move contributions to F first!)
    C. Put 1 on diagonal
    D. Set F[i] = known_temp
    """
    K_mod = K.copy()
    F_mod = F.copy()
    
    # Convert bc_nodes to a set for faster lookup
    bc_nodes_set = set(bc_nodes)
    
    for node_id, value in zip(bc_nodes, bc_values):
        # IMPORTANT: Before zeroing the column, move its contribution to F
        # For all unknown nodes i: F[i] -= K[i, node_id] * value
        for i in range(len(F_mod)):
            if i not in bc_nodes_set:  # Only modify unknown nodes
                F_mod[i] -= K_mod[i, node_id] * value
        
        # Step A: Zero entire row
        K_mod[node_id, :] = 0.0
        
        # Step B: Zero the column
        K_mod[:, node_id] = 0.0
        
        # Step C: Put 1 on diagonal
        K_mod[node_id, node_id] = 1.0
        
        # Step D: Set F[i] = known_temp
        F_mod[node_id] = value
    
    return K_mod, F_mod

def solve_fem(nodes_file, elements_file, bc_file, k=1.0, t=1.0, q_v=0.0):
    """
    Solve the FEM heat transfer problem.
    Returns the temperature at all nodes.
    """
    # Step 1: Load data
    nodes = read_nodes(nodes_file)
    elements = read_elements(elements_file)
    bc_nodes, bc_values = read_bcs(bc_file)
    
    N = nodes.shape[0]
    
    # Step 2 & 3: Calculate global stiffness matrix
    print("Calculating global stiffness matrix...")
    K = calculate_global_stiffness_matrix(nodes_file, elements_file, k=k, t=t)
    
    # Initialize force vector (heat sources)
    F = np.full(N, q_v, dtype=float)
    
    # Step 4: Apply Dirichlet boundary conditions
    print("Applying boundary conditions...")
    K_mod, F_mod = apply_dirichlet_bcs(K, F, bc_nodes, bc_values)
    
    # Step 5: Solve the system K*T = F
    print("Solving system of equations...")
    T = np.linalg.solve(K_mod, F_mod)

    return T, nodes, elements

def calculate_heat_flux(nodes, elements, T, k=1.0):
    """
    Calculate heat flux for each element using Fourier's Law: j = -k * ∇T
    """
    heat_fluxes = []
    element_centers = []
    
    for elem in elements:
        xy = nodes[elem]
        temps = T[elem]
        
        x1, y1 = xy[0]
        x2, y2 = xy[1]
        x3, y3 = xy[2]
        
        # Calculate b and c coefficients
        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])
        
        A = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        
        # Temperature gradient: ∇T = (1/2A) * [b; c] * T
        dT_dx = (1 / (2 * A)) * np.dot(b, temps)
        dT_dy = (1 / (2 * A)) * np.dot(c, temps)
        
        # Heat flux: j = -k * ∇T
        jx = -k * dT_dx
        jy = -k * dT_dy
        
        heat_fluxes.append([jx, jy])
        
        # Element center
        center_x = np.mean(xy[:, 0])
        center_y = np.mean(xy[:, 1])
        element_centers.append([center_x, center_y])
    
    return np.array(heat_fluxes), np.array(element_centers)

def plot_results(nodes, elements, T, heat_fluxes, element_centers, out_file):
    """Plot temperature distribution and heat flux vectors."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Temperature contours
    for elem in elements:
        triangle = nodes[elem]
        tri = patches.Polygon(triangle, fill=False, edgecolor='black', linewidth=0.5)
        ax1.add_patch(tri)
    
    # Create temperature contour
    tri_plot = ax1.tricontourf(nodes[:, 0], nodes[:, 1], T, levels=15, cmap='YlOrRd')
    plt.colorbar(tri_plot, ax=ax1, label='Temperature (°C)')
    
    # Plot nodes with temperature values
    scatter = ax1.scatter(nodes[:, 0], nodes[:, 1], c=T, s=100, 
                         edgecolors='black', cmap='YlOrRd', zorder=5)
    
    for i, (x, y) in enumerate(nodes):
        ax1.annotate(f'{i+1}\n{T[i]:.1f}°C', (x, y), 
                    fontsize=8, ha='center', va='bottom')
    
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_title('Temperature Distribution', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Heat flux vectors
    for elem in elements:
        triangle = nodes[elem]
        tri = patches.Polygon(triangle, fill=False, edgecolor='black', linewidth=0.5)
        ax2.add_patch(tri)
    
    # Plot heat flux vectors
    ax2.quiver(element_centers[:, 0], element_centers[:, 1],
              heat_fluxes[:, 0], heat_fluxes[:, 1],
              scale=50, width=0.005, color='blue', alpha=0.7)
    
    ax2.scatter(nodes[:, 0], nodes[:, 1], c='red', s=50, zorder=5)
    
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
    nodes_file = "figure1_example_nodes_2D.txt"
    elements_file = "figure1_example_triangles_2D.txt"
    bc_file = "figure1_example_g_bc_2D.txt"

    # Plot mesh and boundary conditions
    plot_mesh_nodes_bcs(nodes_file, elements_file, bc_file, "mesh_plot.png")
    
    # Solve FEM problem
    T, nodes, elements = solve_fem(nodes_file, elements_file, bc_file, k=1.0)
    
    print("\nNodal Temperatures:")
    print("-" * 30)
    for i, temp in enumerate(T):
        print(f"Node {i+1}: {temp:.2f} °C")
    
    # Calculate heat flux
    heat_fluxes, element_centers = calculate_heat_flux(nodes, elements, T, k=1.0)
    
    print("\nHeat Flux per Element:")
    print("-" * 30)
    for i, (flux, center) in enumerate(zip(heat_fluxes, element_centers)):
        magnitude = np.linalg.norm(flux)
        print(f"Element {i+1}: jx={flux[0]:.3f}, jy={flux[1]:.3f}, |j|={magnitude:.3f} W/m²")
    
    # Plot results
    plot_results(nodes, elements, T, heat_fluxes, element_centers, "results_plot.png")
    

if __name__ == "__main__":
    main()