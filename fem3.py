import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys

# =============================================================================
# FILE LOADING LOGIC (From fem2.py)
# =============================================================================

def read_nodes(filename):
    """Read nodes file. Expect two columns: x, y (comma or space separated)."""
    try:
        with open(filename, 'r') as f:
            first_line = f.readline().strip()
            if ',' in first_line:
                delimiter = ','
            else:
                delimiter = None  
        
        nodes = np.loadtxt(filename, delimiter=delimiter)
        return nodes
    except Exception as e:
        print(f"Error reading nodes file '{filename}': {e}")
        sys.exit(1)

def read_elements(filename):
    """Read triangle elements. Expect three 1-based integer columns."""
    try:
        with open(filename, 'r') as f:
            first_line = f.readline().strip()
            if ',' in first_line:
                delimiter = ','
            else:
                delimiter = None
        
        elements = np.loadtxt(filename, dtype=int, delimiter=delimiter)
        elements = elements - 1
        return elements.astype(int)
    except Exception as e:
        print(f"Error reading elements file '{filename}': {e}")
        sys.exit(1)

def read_bcs(filename):
    """Read Dirichlet BC file. Expect two columns: node_id, value."""
    try:
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
    except Exception as e:
        print(f"Error reading BC file '{filename}': {e}")
        sys.exit(1)

# =============================================================================
# MESH GENERATION LOGIC
# =============================================================================

def generate_rect_mesh(width, height, nx, ny, t_base, t_tip):
    """Generates nodes, elements, and BCs for a rectangular heat sink."""
    print(f"\nGenerating mesh: {width}x{height}m, Grid: {nx}x{ny}...")
    
    # 1. Nodes
    x = np.linspace(0, width, nx)
    y = np.linspace(0, height, ny)
    xv, yv = np.meshgrid(x, y)
    nodes = np.column_stack((xv.flatten(), yv.flatten()))
    
    # 2. Elements
    elements = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            n_bl = i * nx + j
            n_br = i * nx + (j + 1)
            n_tl = (i + 1) * nx + j
            n_tr = (i + 1) * nx + (j + 1)
            
            # Triangle 1
            elements.append([n_bl, n_br, n_tr])
            # Triangle 2
            elements.append([n_bl, n_tr, n_tl])
            
    elements = np.array(elements)
    
    # 3. Boundary Conditions
    bc_nodes_list = []
    bc_values_list = []
    
    # Bottom Edge (Fixed Temp)
    for k in range(nx):
        node_idx = k 
        bc_nodes_list.append(node_idx)
        bc_values_list.append(t_base)
        
    # Top Edge (Fixed Temp if not Insulated)
    if t_tip is not None:
        start_idx = (ny - 1) * nx
        for k in range(nx):
            node_idx = start_idx + k
            bc_nodes_list.append(node_idx)
            bc_values_list.append(t_tip)
            
    return nodes, elements, np.array(bc_nodes_list), np.array(bc_values_list)

# =============================================================================
# FEM SOLVER LOGIC (From fem2.py)
# =============================================================================

def calculate_element_stiffness_matrix(element_nodes, k=1.0, t=1.0):
    xy = np.asarray(element_nodes, dtype=float)
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]

    A = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    
    if A < 1e-12: return np.zeros((3,3))

    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    
    B_grad = np.vstack((b, c))
    
    Ke = (k * t / (4 * A)) * (B_grad.T @ B_grad)
    return Ke

def solve_fem(nodes, elements, bc_nodes, bc_values, k_cond=200.0, t=0.01):
    N = nodes.shape[0]
    K = np.zeros((N, N), dtype=float)

    print(f"Assembling stiffness matrix for {N} nodes...")
    for elem in elements:
        xy = nodes[elem, :]
        Ke = calculate_element_stiffness_matrix(xy, k=k_cond, t=t)
        
        for a_loc, a_glob in enumerate(elem):
            for b_loc, b_glob in enumerate(elem):
                K[a_glob, b_glob] += Ke[a_loc, b_loc]

    # Apply BCs
    F = np.zeros(N)
    K_mod = K.copy()
    F_mod = F.copy()
    
    bc_set = set(bc_nodes)
    
    print("Applying boundary conditions...")
    for node_id, value in zip(bc_nodes, bc_values):
        for i in range(N):
            if i not in bc_set:
                F_mod[i] -= K_mod[i, node_id] * value
        
        K_mod[node_id, :] = 0.0
        K_mod[:, node_id] = 0.0
        K_mod[node_id, node_id] = 1.0
        F_mod[node_id] = value
        
    print("Solving linear system...")
    T = np.linalg.solve(K_mod, F_mod)
    return T

def calculate_heat_flux(nodes, elements, T, k=200.0):
    heat_fluxes = []
    centers = []
    
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
        
        heat_fluxes.append([-k * dT_dx, -k * dT_dy])
        centers.append([np.mean(xy[:,0]), np.mean(xy[:,1])])
        
    return np.array(heat_fluxes), np.array(centers)

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_interactive_results(nodes, elements, T, fluxes, centers, base_name):
    print("Generating 2D visualizations...")
    
    # 1. Thermal Gradients (Gouraud)
    fig, ax = plt.subplots(figsize=(10, 8))
    trip = ax.tripcolor(nodes[:,0], nodes[:,1], elements, T, cmap='turbo', shading='gouraud')
    ax.triplot(nodes[:,0], nodes[:,1], elements, color='k', lw=0.1, alpha=0.3)
    plt.colorbar(trip, ax=ax, label='Temperature (°C)')
    ax.set_title('Thermal Distribution (Linear Gradient)')
    ax.set_aspect('equal')
    ax.grid(False)
    plt.savefig(f"{base_name}_2d.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 3D Surface (Static)
    print("Generating 3D surface plot...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(nodes[:,0], nodes[:,1], T, cmap='turbo', linewidth=0.1, antialiased=True)
    ax.view_init(elev=30, azim=45)
    fig.colorbar(surf, ax=ax, shrink=0.5, label='Temperature (°C)')
    ax.set_title('3D Thermal Profile')
    ax.grid(False)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    plt.savefig(f"{base_name}_3d.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Done! Saved to {base_name}_2d.png and {base_name}_3d.png")

# =============================================================================
# MAIN INTERACTIVE LOOP
# =============================================================================

def get_user_input():
    print("\n" + "="*50)
    print(" INTERACTIVE FEM SOLVER ")
    print("="*50)
    
    mode = input("Select Mode:\n  [G] Generate Mesh (Heat Sink)\n  [L] Load Files (CSV/TXT)\n> ").strip().upper()
    
    if mode == 'L':
        print("\n--- Load Files ---")
        print("Note: Supports comma or space separated values.")
        
        # PROMPT 1: Nodes
        f_nodes = input("Path to Nodes File [e.g. v8_nodes.txt]: ").strip()
        while not Path(f_nodes).exists():
            print(f"Error: File '{f_nodes}' not found.")
            f_nodes = input("Path to Nodes File: ").strip()
            
        # PROMPT 2: Elements 
        f_elems = input("Path to Elements File [e.g. v8_triangles.txt]: ").strip()
        while not Path(f_elems).exists():
            print(f"Error: File '{f_elems}' not found.")
            f_elems = input("Path to Elements File: ").strip()
            
        # PROMPT 3: BCs
        f_bcs = input("Path to BC File [e.g. v8_bcs.txt]: ").strip()
        while not Path(f_bcs).exists():
            print(f"Error: File '{f_bcs}' not found.")
            f_bcs = input("Path to BC File: ").strip()
            
        return 'LOAD', (f_nodes, f_elems, f_bcs)
        
    else:
        # Defaults to Generate
        print("\n--- Heat Sink Generator ---")
        try:
            w_in = input("Enter Width (m) [0.05]: ").strip()
            width = float(w_in) if w_in else 0.05
            
            h_in = input("Enter Height (m) [0.10]: ").strip()
            height = float(h_in) if h_in else 0.10
            
            nx_in = input("Nodes along Width (low=10, med=20, high=50) [20]: ").strip()
            nx = int(nx_in) if nx_in else 20
            
            # Scale ny roughly by aspect ratio to keep triangles nice
            aspect = height / width
            ny = int(nx * aspect)
            print(f"-> Calculated Nodes along Height: {ny}")
            
            tb_in = input("Base (Bottom) Temperature (°C) [100]: ").strip()
            t_base = float(tb_in) if tb_in else 100.0
            
            tt_in = input("Tip (Top) Temperature (°C) (or 'ins' for insulated) [25]: ").strip()
            if tt_in.lower() == 'ins':
                t_tip = None
                print("-> Top edge set to INSULATED (Adiabatic)")
            else:
                t_tip = float(tt_in) if tt_in else 25.0
                
            return 'GEN', (width, height, nx, ny, t_base, t_tip)
            
        except ValueError:
            print("Invalid input! using defaults.")
            return 'GEN', (0.05, 0.10, 20, 40, 100.0, 25.0)

def main():
    base_dir = Path(__file__).resolve().parent
    
    # 1. Get Inputs
    mode, data = get_user_input()
    
    if mode == 'GEN':
        width, height, nx, ny, t_base, t_tip = data
        nodes, elements, bc_nodes, bc_values = generate_rect_mesh(width, height, nx, ny, t_base, t_tip)
        out_prefix = "user_heatsink"
    else:
        f_nodes, f_elems, f_bcs = data
        print(f"\nLoading mesh data...")
        nodes = read_nodes(f_nodes)
        elements = read_elements(f_elems)
        bc_nodes, bc_values = read_bcs(f_bcs)
        out_prefix = "user_loaded"
        print(f"Loaded: {len(nodes)} nodes, {len(elements)} elements, {len(bc_nodes)} BCs")

    # 2. Solve
    k_alu = 200.0 # Standard Aluminum or similar
    T = solve_fem(nodes, elements, bc_nodes, bc_values, k_cond=k_alu)
    
    # 3. Post-Process
    fluxes, centers = calculate_heat_flux(nodes, elements, T, k=k_alu)
    
    # 4. Visualize
    out_name = str(base_dir / out_prefix)
    plot_interactive_results(nodes, elements, T, fluxes, centers, out_name)
    
    print("\nSimulation Complete!")

if __name__ == "__main__":
    main()

"""
ALGORITHM OVERVIEW:

1.  User Interaction: 
    - Mode G (Generate): Collect geometry params and build mesh in-memory.
    - Mode L (Load): Parse external CSV/TXT files for Nodes, Elements, and BCs.
2.  Mesh Processing: Verify node coordinates and element connectivity.
3.  FEM Solution: Assemble Stiffness Matrix (K) and Load Vector (F), then solve K * T = F.
4.  Visualization: Generate static 2D linear gradient plots and 3D surface plots.
"""
