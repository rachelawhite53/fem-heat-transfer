# 2D Finite Element Method (FEM) Heat Transfer Solver

This repository contains a Python implementation of a 2D Finite Element Method (FEM) solver for steady-state heat conduction problems. It uses linear triangular elements to calculate temperature distributions and heat flux vectors across complex geometries.

The project includes two distinct versions: a basic solver for educational verification (`fem.py`) and a more robust solver applied to a V8 engine block cross-section (`fem2.py`).

## Project Structure

### Python Scripts
* **`fem.py`**: A simplified FEM solver designed for small, manual meshes (Example 1).
* **`fem2.py`**: An advanced version of the solver optimized for larger datasets, specifically configured for the V8 engine mesh (Example 2).

### Input Data Files
The solver requires three text files defining the geometry and physics:
1. **Nodes**: XY coordinates of mesh vertices.
2. **Elements**: Connectivity list defining the triangular elements (node indices).
3. **Boundary Conditions (BC)**: List of nodes with fixed temperatures (Dirichlet conditions).

**Example 1: Simple Mesh** (Used by `fem.py`)
* `figure1_example_nodes_2D.txt`
* `figure1_example_triangles_2D.txt`
* `figure1_example_g_bc_2D.txt`

**Example 2: V8 Motor Block** (Used by `fem2.py`)
* `v8_motor_nodes_2D.txt`
* `v8_motor_triangles_2D.txt`
* `v8_motor_g_bc_hot_2D.txt`

## Features

* **Element Type**: Linear 3-node triangular elements  
* **Physics**: Steady-state 2D heat conduction  
* **Boundary Conditions**: Dirichlet (fixed temperature)  
* **Post-Processing**:
  - Computes nodal temperature fields  
  - Computes heat flux vectors using Fourier’s Law  
* **Visualization**: Mesh plots, temperature contours, and heat flux vector fields via `matplotlib`

## Dependencies

* Python 3.x  
* `numpy`  
* `matplotlib`

## Usage

### Running the Simple Example
```bash
python fem.py
```

**Output:**
* Console printout of nodal temperatures  
* `mesh_plot.png`  
* `results_plot.png`

### Running the V8 Engine Example
```bash
python fem2.py
```

**Configuration:**
* Thermal Conductivity (k): **50.0 W/m·K**

**Output:**
* Min/max temperatures and heat flux statistics  
* `v8_mesh_plot.png`  
* `v8_results_plot.png`

---

## Finite Element Algorithm

### 1. Pre-Processing
The script loads:
- Node coordinates  
- Element connectivity  
- Dirichlet boundary conditions  

---

### 2. Element Stiffness Matrix

For each 3-node triangular element, a local stiffness matrix $$K^{(e)}$$ is computed.

#### A. Area Calculation
The triangle area A is:

$$
A = \frac{1}{2}\big|(x_2 - x_1)(y_3 - y_1) - (x_3 - x_1)(y_2 - y_1)\big|
$$

#### B. Gradient Coefficients  
Coefficients for the shape function gradients (using standard linear triangular shape functions):

$$
\begin{align*}
b_1 &= y_2 - y_3 \\
b_2 &= y_3 - y_1 \\
b_3 &= y_1 - y_2 \\
c_1 &= x_3 - x_2 \\
c_2 &= x_1 - x_3 \\
c_3 &= x_2 - x_1
\end{align*}
$$

#### C. Local Matrix Assembly
The element stiffness matrix entries are:

$$
K_{ij}^{(e)} = \frac{k}{4A}(b_i b_j + c_i c_j)
$$

---

### 3. Global Assembly
Local matrices $$K^{(e)}$$ are added to the global stiffness matrix \(K\) using global node indices.

---

### 4. Application of Dirichlet Boundary Conditions

Dirichlet conditions are applied by modifying the system:

1. Move known-temperature contributions to the force vector \(F\).  
2. Zero out the row and column of each constrained node.  
3. Set the diagonal value to **1**.  
4. Set the force vector component to the known temperature.  

---

### 5. Solution
Solve for nodal temperatures:

$$
T = K^{-1}F
$$

---

### 6. Post-Processing (Heat Flux)

Element heat flux:

$$
\vec{j} = -k \nabla T
$$

Flux flows from hot regions toward cold regions.
