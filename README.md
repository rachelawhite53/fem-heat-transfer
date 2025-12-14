# Finite Element Method (FEM) Mathematical Formulation

This document details the mathematical logic behind the 2D steady-state heat conduction solvers implemented in `fem.py`, `fem2.py`, and `fem3.py`. The solver uses linear triangular elements to approximate the solution to the governing differential equation.

---

## 1. Governing Equation

The problem is governed by the steady-state heat diffusion equation. For a 2D domain $\Omega$ with thermal conductivity $k$:

$$
\nabla \cdot (k \nabla T) + Q = 0 \quad \text{in } \Omega
$$

Where:
*   $T(x,y)$ is the temperature field.
*   $Q$ is the volumetric heat generation (assumed zero in these examples).

Subject to Boundary Conditions (BCs):
*   **Dirichlet (Essential)**: $T = T_{fixed}$ on boundary $\Gamma_D$.
*   **Neumann (Natural)**: $-k \frac{\partial T}{\partial n} = q_{flux}$ on boundary $\Gamma_N$ (assumed adiabatic/zero-flux where not specified).

---

## 2. Weak Formulation

The Finite Element Method relies on the weak (integral) form of the equation. We multiply by an arbitrary test function $w$ (which vanishes on $\Gamma_D$) and integrate over the domain:

$$
\int_\Omega w [\nabla \cdot (k \nabla T)] \, d\Omega = 0
$$

Using integration by parts (Green's First Identity):

$$
-\int_\Omega k (\nabla w \cdot \nabla T) \, d\Omega + \oint_\Gamma w (k \nabla T \cdot \mathbf{n}) \, d\Gamma = 0
$$

Assuming adiabatic natural boundaries ($q_{flux}=0$) and noting $w=0$ on $\Gamma_D$, the boundary integral vanishes. The weak form becomes:

Find $T$ such that for all test functions $w$:
$$
\int_\Omega k (\nabla w \cdot \nabla T) \, d\Omega = 0
$$

---

## 3. Discretization (Linear Triangular Elements)

The domain $\Omega$ is discretized into small triangular elements $\Omega_e$. Within each element, the temperature $T$ is approximated by a linear combination of shape functions $N_i$:

$$
T(x, y) \approx \sum_{j=1}^{3} N_j(x, y) T_j
$$

Where $T_j$ are the nodal temperatures. For a linear triangle, the shape functions $N_i$ are:

$$
N_i = \frac{1}{2A} (a_i + b_i x + c_i y)
$$

Where $A$ is the area of the element.

The coefficients are derived from the node coordinates $(x_1, y_1), (x_2, y_2), (x_3, y_3)$ by cycling indices ($1 \to 2 \to 3 \to 1$):
*   $a_1 = x_2 y_3 - x_3 y_2$
*   $b_1 = y_2 - y_3$
*   $c_1 = x_3 - x_2$

And so on for indices 2 and 3.

---

## 4. Element Stiffness Matrix

Substituting the approximation $T = \mathbf{N}\mathbf{T}^e$ and $w = \mathbf{N}$ into the weak form integral for a single element:

$$
\mathbf{K}^e \mathbf{T}^e = 0
$$

The element stiffness matrix entries $K_{ij}^e$ are:

$$
K_{ij}^e = \int_{\Omega_e} k (\nabla N_i \cdot \nabla N_j) \, d\Omega
$$

Since we use linear elements, the gradients $\nabla N_i$ are constant over the element:

$$
\nabla N_i = \begin{bmatrix} \frac{\partial N_i}{\partial x} \\ \frac{\partial N_i}{\partial y} \end{bmatrix} = \frac{1}{2A} \begin{bmatrix} b_i \\ c_i \end{bmatrix}
$$

The integral becomes a simple multiplication by the element area $A$:

$$
K_{ij}^e = \int_{\Omega_e} k \left[ \left(\frac{b_i}{2A}\right)\left(\frac{b_j}{2A}\right) + \left(\frac{c_i}{2A}\right)\left(\frac{c_j}{2A}\right) \right] d\Omega
$$

Simplifying:

$$
K_{ij}^e = \frac{k}{4A} (b_i b_j + c_i c_j)
$$

This $3 \times 3$ matrix represents the thermal conductance of a single triangle.

---

## 5. Global Assembly

The total system behavior is the sum of all elements. We assemble the global stiffness matrix $\mathbf{K}$ (generic $N \times N$ size) by mapping local node indices $(1, 2, 3)$ to global node indices $(I, J, K)$:

$$
\mathbf{K}_{global} = \sum_{e} \mathbf{K}^e
$$

This results in a linear system:

$$
\mathbf{K}_{global} \cdot \mathbf{T}_{global} = \mathbf{F}_{global}
$$

Initially, $\mathbf{F}_{global} = 0$ (assuming no internal heat generation).

---

## 6. Boundary Conditions (Dirichlet)

We must enforce known temperatures ($T_k = T_{known}$) at specific nodes. The solvers use the **Matrix Modification Method**:

For a node $k$ fixed at value $V$:
1.  **Modify Load Vector $\mathbf{F}$**: Subtract the contribution of the known node from all other equations to maintain consistency.
    $$ F_i \leftarrow F_i - K_{ik} \cdot V \quad (\text{for all } i \neq k) $$
2.  **Isolate the Equation**: Zero out the $k$-th row and $k$-th column of $\mathbf{K}$.
    $$ K_{ik} = 0, \quad K_{ki} = 0 $$
3.  **Set Diagonal**: Set the diagonal entry to 1.
    $$ K_{kk} = 1 $$
4.  **Set Forcing**: Set the specific RHS entry to the known value.
    $$ F_k = V $$

This forces the solution $1 \cdot T_k = V$ while preserving the symmetry and solvability of the system.

---

## 7. Solution and Post-Processing

The system is now solved using a linear algebra solver (e.g., NumPy's `linalg.solve`):

$$
\mathbf{T} = \mathbf{K}^{-1} \mathbf{F}
$$

### Heat Flux Calculation
Once nodal temperatures $T$ are known, the heat flux vector $\mathbf{j}$ for each element is calculated using Fourier's Law:

$$
\mathbf{j} = -k \nabla T
$$

In discretized form for an element:

$$
\mathbf{j} = -k \sum_{i=1}^3 T_i \nabla N_i = -\frac{k}{2A} \sum_{i=1}^3 T_i \begin{bmatrix} b_i \\ c_i \end{bmatrix}
$$

This produces a constant flux vector for each triangular element.
