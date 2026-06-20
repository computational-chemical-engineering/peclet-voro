import os

markdown_content = """# Near-Incompressible Power-Cell Solver Specification

This document provides a comprehensive methodology for a thermodynamically consistent fluid solver using **Power Cells** (Weighted Voronoi Diagrams). The approach is derived from a variational principle, treating both material positions and cell volumes as dynamic degrees of freedom.

---

## 1. The Lagrangian Formulation

The system state is defined by $q = [x, w]^T$, where $x$ are the seed positions and $w$ are the power weights.

### 1.1 Kinetic Energy (Consistent Mass Matrix)
Instead of a lumped mass approach, we use a piecewise linear reconstruction of the velocity field $u(y)$ within the dual Delaunay tetrahedra. The kinetic energy is:

$$T(q, \dot{q}) = \\frac{1}{2} \\dot{q}^T M(q) \\dot{q}$$

The **Consistent Mass Matrix** $M(q)$ is non-diagonal and couples the linear motion of the seeds $\dot{x}$ to the expansion/contraction of the cells $\dot{w}$:

$$M(q) = \\begin{bmatrix} M_{xx} & M_{xw} \\\\ M_{wx} & M_{ww} \\end{bmatrix}$$

This coupling ensures that high-frequency mesh oscillations (checkerboarding) carry an energy cost, effectively regularizing the discretization.

### 1.2 Potential Energy and Lloyd Regularization
The total potential is $\Pi(q) = \Pi_{vol}(q) + E_{Lloyd}(q)$.

The **Lloyd Energy** acts as the "geometric spring":
$$E_{Lloyd} = \\gamma \\sum_i \\int_{V_i(q)} (\|y - x_i\|^2 - w_i) dy$$

Taking the gradient with respect to $x_i$ yields the restoring force:
$$F_{Lloyd, i} = -2 \\gamma V_i (x_i - C_i)$$
where $C_i$ is the center of mass of the Power Cell. This keeps the seeds centered, preventing facet-rotation instabilities.

---

## 2. Hamiltonian Dynamics

To resolve the motion, we define the conjugate momenta $p = M(q) \dot{q}$. The Hamiltonian $H$ (total energy) is:

$$H(q, p) = \\frac{1}{2} p^T M(q)^{-1} p + \\Pi_{vol}(q) + E_{Lloyd}(q)$$

### 2.1 Canonical Equations
The equations of motion are:
1.  **Kinematics:** $\dot{q} = M(q)^{-1} p = v$
2.  **Dynamics:** $\dot{p} = -\\nabla_q \\Pi(q) + \\frac{1}{2} v^T \\nabla_q M(q) v$

The second term in $\dot{p}$ is the **Metric Force**, which accounts for the energy exchange required to reshape the "fluid-filled" Voronoi cells.

---

## 3. Numerical Integration: Implicit Midpoint Rule

Since $M(q)$ depends on $q$, the Hamiltonian is non-separable. We use the **Implicit Midpoint Rule**, which is second-order and symplectic.

Given $(q_n, p_n)$, find $(q_{n+1}, p_{n+1})$ such that:
$$q^* = \\frac{q_n + q_{n+1}}{2}, \quad p^* = \\frac{p_n + p_{n+1}}{2}$$
$$q_{n+1} = q_n + \\Delta t [ M(q^*)^{-1} p^* ]$$
$$p_{n+1} = p_n - \\Delta t \\left[ \\nabla_q \\Pi(q^*) - \\frac{1}{2} (v^*)^T \\nabla_q M(q^*) v^* \\right]$$

---

## 4. Near-Incompressible Projection Method

To solve the stiff limit efficiently and suppress acoustic waves, we use a projection step based on the continuity equation.

### 4.1 The Continuity Residual
At the end of a predictor step, the volumes $V_i(x_{n+1}, w_{n+1})$ will deviate from the target volumes $V_i^*$. The residual is:
$$R_i = V_i(q^*) - V_i^*$$

### 4.2 Weight (Pressure) Correction
We solve for a correction $\\delta w$ that satisfies the volume constraint implicitly:
$$\\sum_{j \\in nb(i)} \\frac{A_{ij}}{2d_{ij}} (\\delta w_j - \\delta w_i) = \\frac{R_i}{\\Delta t}$$
where $A_{ij}$ is the facet area and $d_{ij}$ is the seed distance. This is the **Pressure Poisson Equation** in Power Diagram space.

### 4.3 Sound Wave Suppression
Treating $\\delta w$ implicitly filters out the acoustic modes, allowing the time step $\\Delta t$ to be determined by the fluid velocity (CFL condition) rather than the speed of sound.

---

## 5. Methodology Summary

1.  **Predictor:** Perform a Hamiltonian update for $p$ and $x$ using explicit forces and metric terms.
2.  **Consistent Solve:** Invert $M(q)$ to obtain generalized velocities $v$.
3.  **Projection:** Solve the elliptic weight system to enforce $V_i \\approx V_i^*$.
4.  **Corrector:** Update the weights and momenta to final values.

This framework preserves **Material Identity** by linking the topology of the Voronoi cells directly to the thermodynamic Action of the fluid.
"""

file_path = 'power_cell_solver_spec.md'
with open(file_path, 'w') as f:
    f.write(markdown_content)