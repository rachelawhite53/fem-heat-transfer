#FEM VERSION 2: V8 ENGINE EXAMPLE

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def read_nodes(filename):
    """Read nodes file. Expect two columns: x, y (comma or space separated)."""
    # Try to detect delimiter
    with open(filename, 'r') as f:
        first_line = f.readline().strip()
        if ',' in first_line:
            delimiter = ','
        else:
            delimiter = None  
    
    nodes = np.loadtxt(filename, delimiter=delimiter)
    return nodes

def read_elements(filename):
    """Read triangle elements. Expect three 1-based integer columns (comma or space separated)."""
   
    with open(filename, 'r') as f:
        first_line = f.readline().strip()
        if ',' in first_line:
            delimiter = ','
        else:
            delimiter = None
    
    elements = np.loadtxt(filename, dtype=int, delimiter=delimiter)
    elements = elements - 1
    return elements.astype(int)

def read_bcs(filename):
    """Read Dirichlet BC file. Expect two columns: node_id, value (comma or space separated)."""

    with open(filename, 'r') as f:
        first_line = f.readline().strip()
        if ',' in first_line:
            delimiter = ','
        else:
            delimiter = None  
    
    data = np.loadtxt(filename, delimiter=delimiter)
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
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot elements
    print(f"Plotting {len(elements)} elements...")
    for elem in elements:
        triangle = nodes[elem]
        tri = patches.Polygon(triangle, fill=False, edgecolor='lightblue', 
                             linewidth=0.3, alpha=0.5)
        ax.add_patch(tri)
    
    # Plot all nodes
    ax.plot(nodes[:, 0], nodes[:, 1], 'k.', markersize=2, alpha=0.3, label='Nodes')
    
    # Highlight BC nodes with color based on temperature
    scatter = ax.scatter(nodes[bc_nodes, 0], nodes[bc_nodes, 1], 
                        c=bc_values, s=50, cmap='hot', 
                        edgecolors='black', linewidth=0.5,
                        label='BC Nodes', zorder=5)
    
    plt.colorbar(scatter, ax=ax, label='Temperature (°C)')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'V8 Motor FEM Mesh ({len(nodes)} nodes, {len(elements)} elements)', 
                fontsize=14, fontweight='bold')
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
    
    if A < 1e-12:
        raise ValueError("Element has zero or near-zero area")

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

    print(f"Assembling global stiffness matrix for {N} nodes...")
    for i, elem in enumerate(elements):
        if (i + 1) % 100 == 0:
            print(f"  Processing element {i+1}/{len(elements)}...")
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
    
    print(f"Applying {len(bc_nodes)} boundary conditions...")
    for idx, (node_id, value) in enumerate(zip(bc_nodes, bc_values)):
        if (idx + 1) % 10 == 0:
            print(f"  Processing BC {idx+1}/{len(bc_nodes)}...")
            
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
    
    print(f"Problem size: {N} nodes, {len(elements)} elements, {len(bc_nodes)} BCs")
    
    # Step 2 & 3: Calculate global stiffness matrix
    print("\nCalculating global stiffness matrix...")
    K = calculate_global_stiffness_matrix(nodes_file, elements_file, k=k, t=t)
    
    # Initialize force vector (heat sources)
    F = np.full(N, q_v, dtype=float)
    
    # Step 4: Apply Dirichlet boundary conditions
    print("\nApplying boundary conditions...")
    K_mod, F_mod = apply_dirichlet_bcs(K, F, bc_nodes, bc_values)
    
    # Step 5: Solve the system K*T = F
    print("\nSolving system of equations...")
    T = np.linalg.solve(K_mod, F_mod)
    
    print("Solution complete!")
    return T, nodes, elements

def calculate_heat_flux(nodes, elements, T, k=1.0):
    """
    Calculate heat flux for each element using Fourier's Law: j = -k * ∇T
    """
    heat_fluxes = []
    element_centers = []
    
    print(f"Calculating heat flux for {len(elements)} elements...")
    for i, elem in enumerate(elements):
        if (i + 1) % 200 == 0:
            print(f"  Processing element {i+1}/{len(elements)}...")
            
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Temperature contours
    print("Creating temperature contour plot...")
    
    # Create temperature contour with more levels for smooth gradients
    tri_plot = ax1.tricontourf(nodes[:, 0], nodes[:, 1], T, levels=30, cmap='YlOrRd')
    
    # Plot element outlines ON TOP of the contours
    for elem in elements:
        triangle = nodes[elem]
        tri = patches.Polygon(triangle, fill=False, edgecolor='black', 
                             linewidth=0.3, alpha=0.8)
        ax1.add_patch(tri)
    
    cbar1 = plt.colorbar(tri_plot, ax=ax1, label='Temperature (°C)')
    
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_title('V8 Motor Temperature Distribution', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Heat flux vectors (subsample for large meshes)
    print("Creating heat flux vector plot...")
    
    for elem in elements:
        triangle = nodes[elem]
        tri = patches.Polygon(triangle, fill=False, edgecolor='black', 
                             linewidth=0.2, alpha=0.3)
        ax2.add_patch(tri)
    
    # Subsample heat flux vectors for visibility (every Nth element)
    subsample = max(1, len(element_centers) // 200)  # Show ~200 vectors max
    centers_sub = element_centers[::subsample]
    fluxes_sub = heat_fluxes[::subsample]
    
    # Calculate appropriate scale for arrows
    flux_magnitudes = np.linalg.norm(heat_fluxes, axis=1)
    max_flux = np.max(flux_magnitudes)
    scale = max_flux * 10 # Adjust as needed
    
    ax2.quiver(centers_sub[:, 0], centers_sub[:, 1],
              fluxes_sub[:, 0], fluxes_sub[:, 1],
              scale=scale, width=0.003, color='blue', alpha=0.6)
    
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title(f'Heat Flux Vectors (j = -k∇T)\nMax |j| = {max_flux:.2f} W/m²', 
                 fontsize=14, fontweight='bold')
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"Results plot saved to {out_file}")
    plt.close()
    
def main():
    # V8 Motor configuration
    nodes_file = "v8_motor_nodes_2D.txt"
    elements_file = "v8_motor_triangles_2D.txt"
    bc_file = "v8_motor_g_bc_hot_2D.txt"
    
    # Thermal conductivity for typical engine materials (W/m·K)
    # Steel: ~50, Aluminum: ~200, Cast Iron: ~50
    k_thermal = 50.0  # W/m·K
   
    # Plot mesh and boundary conditions
    plot_mesh_nodes_bcs(nodes_file, elements_file, bc_file, "v8_mesh_plot.png")
    
    # Solve FEM problem
    T, nodes, elements = solve_fem(nodes_file, elements_file, bc_file, k=k_thermal)
    
    # Print temperature statistics
    print("\n" + "="*60)
    print("TEMPERATURE STATISTICS")
    print("="*60)
    print(f"Minimum temperature: {np.min(T):.2f} °C")
    print(f"Maximum temperature: {np.max(T):.2f} °C")
    print(f"Mean temperature: {np.mean(T):.2f} °C")
    print(f"Std deviation: {np.std(T):.2f} °C")
    
    # Calculate heat flux
    print("\n" + "="*60)
    print("CALCULATING HEAT FLUX")
    print("="*60 + "\n")
    heat_fluxes, element_centers = calculate_heat_flux(nodes, elements, T, k=k_thermal)
    
    # Heat flux statistics
    flux_magnitudes = np.linalg.norm(heat_fluxes, axis=1)
    print(f"\nHeat Flux Statistics:")
    print(f"  Minimum |j|: {np.min(flux_magnitudes):.2f} W/m²")
    print(f"  Maximum |j|: {np.max(flux_magnitudes):.2f} W/m²")
    print(f"  Mean |j|: {np.mean(flux_magnitudes):.2f} W/m²")
    
    # Plot results
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60 + "\n")
    plot_results(nodes, elements, T, heat_fluxes, element_centers, "v8_results_plot.png")
    

if __name__ == "__main__":
    main()