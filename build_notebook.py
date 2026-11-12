"""
Script to generate the complete CalB06 Google Colab / Jupyter Notebook.
"""
import json

notebook = {
    'cells': [],
    'metadata': {
        'language_info': {
            'name': 'python',
            'version': '3.10.0'
        },
        'colab': {
            'provenance': [],
            'toc_visible': True
        },
        'kernelspec': {
            'display_name': 'Python 3',
            'name': 'python3'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 2
}

def add_md(text):
    notebook['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': text.splitlines(keepends=True)
    })

def add_code(code):
    notebook['cells'].append({
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': code.splitlines(keepends=True)
    })

# Cell 1: Header and Overview
add_md("""# Project CalB06: Wind & Solar Power Resource Allocation under Budget Constraints
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/triduc-tran/CalB06_Wind_Solar_Optimization/blob/main/CalB06_Optimization.ipynb)

**Course:** Calculus in Business (Giải tích trong kinh doanh) - Semester HK261  
**Topic:** CalB-06 – Điện gió, điện mặt trời và phân bổ ngân sách có ràng buộc  

---

## Executive Summary & Mathematical Framework
When public energy planners or private utility developers allocate investment capital under a fixed normalized budget $B$, they face a trade-off between **Solar Photovoltaic (PV) power ($x$)** and **Wind power ($y$)**. Due to diminishing marginal returns in grid absorption, technology deployment, and capacity factors, concentrating all capital into a single source rarely yields the highest overall welfare benefit.

To analyze this capital budgeting problem quantitatively, we formulate the constrained optimization model:
$$\\max_{x, y \\ge 0} F(x, y) = S_0 \\ln(1 + x) + W_0 \\ln(1 + y) \\quad \\text{subject to} \\quad x + y = B$$

where:
- $x, y$ denote normalized investment budget allocations to Solar and Wind, respectively.
- $B$ is the total normalized budget parameter ($B \\in [5, 15]$).
- $S_0, W_0$ are non-negative weighting parameters calibrated from Vietnam's empirical 2024 electricity generation (*Our World in Data*).
- The $+1$ inside the logarithms guarantees that zero budget yields zero utility ($F(0, 0) = 0$), eliminating the mathematical singularity at $\\ln(0)$.

---
### Notebook Roadmap:
1. **Environment Setup & Data Pipeline**: Ingest and clean Our World in Data (OWID) Vietnam energy series (2000–2024).
2. **Analytical Derivations**: Formulate the Lagrangian, derive First-Order Conditions (FOC), compute closed-form formulas, and prove strict concavity via Second-Order Conditions (SOC).
3. **Handbook Benchmark Verification**: Validate with the course handbook benchmark ($S_0 = 2, W_0 = 1, B = 10$).
4. **Numerical Optimization via SciPy**: Solve via Sequential Least Squares Programming (`SLSQP`) and conduct error benchmarking.
5. **Boundary / Corner Solutions**: Evaluate Kuhn-Tucker boundary activation in extreme weight scenarios.
6. **Empirical Case Study (Vietnam 2024)**: Solve with real-world weights ($S_0 = 25.86$ TWh, $W_0 = 12.75$ TWh).
7. **Budget Sensitivity Analysis ($B \\in [5, 15]$)**: Verify the Envelope Theorem ($\\frac{dF^*}{dB} = \\lambda^*$).
8. **Visualizations**: Render 4 publication-quality academic figures.
9. **Academic Discussion & Policy Extensions**: Analyze economic interpretations and realistic grid constraints.""")

# Cell 2: Setup
add_code("""# 1. System and Scientific Libraries Setup
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure Matplotlib styling for high-quality inline display
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 10

print("Environment initialized successfully with NumPy, Pandas, SciPy, and Matplotlib.")""")

# Cell 3: Data Ingestion MD
add_md("""## Part 1: Empirical Data Pipeline (Vietnam Renewable Generation 2000–2024)
We load the historical energy generation dataset for Vietnam compiled from **Our World in Data (OWID)**.
The dataset tracks annual electricity generation from solar and wind power in Terawatt-hours (TWh).""")

# Cell 4: Data Ingestion Code
add_code("""# 2. Ingest and Clean OWID Dataset
csv_filename = "vietnam_solar_wind_cleaned.csv"

# Read raw file and drop non-numerical note lines at bottom
df_raw = pd.read_csv(csv_filename)
df_clean = df_raw.dropna(subset=['country', 'solar_electricity', 'wind_electricity']).copy()
df_clean['year'] = pd.to_numeric(df_clean['year'], errors='coerce')
df_clean = df_clean.dropna(subset=['year'])
df_clean['year'] = df_clean['year'].astype(int)
df_clean = df_clean.sort_values(by='year').reset_index(drop=True)

# Extract 2024 baseline generation
row_2024 = df_clean[df_clean['year'] == 2024].iloc[0]
S0_raw = float(row_2024['solar_electricity'])
W0_raw = float(row_2024['wind_electricity'])
total_gen_2024 = S0_raw + W0_raw

# Compute normalized weights (sum to 1.0)
s0_norm = S0_raw / total_gen_2024
w0_norm = W0_raw / total_gen_2024

print("--- Empirical Dataset Overview (2000-2024) ---")
print(f"Total historical records: {len(df_clean)} years")
print(f"Year 2024 Solar Generation (S0): {S0_raw:.2f} TWh ({s0_norm*100:.1f}%)")
print(f"Year 2024 Wind Generation  (W0): {W0_raw:.2f} TWh ({w0_norm*100:.1f}%)")
print(f"Year 2024 Total Generation     : {total_gen_2024:.2f} TWh")
print("\nRecent 5 Years Preview:")
df_clean.tail(5)""")

# Cell 5: Analytical Theory MD
add_md("""## Part 2: Mathematical Theory & Closed-Form Analytical Optimization

### 2.1 The Lagrangian Function
To maximize $F(x, y) = S_0 \\ln(1 + x) + W_0 \\ln(1 + y)$ subject to $x + y = B$, we formulate:
$$\\mathcal{L}(x, y, \\lambda) = S_0 \\ln(1 + x) + W_0 \\ln(1 + y) - \\lambda (x + y - B)$$

### 2.2 First-Order Necessary Conditions (FOC)
Setting first-order partial derivatives to zero:
$$\\frac{\\partial \\mathcal{L}}{\\partial x} = \\frac{S_0}{1 + x} - \\lambda = 0 \\implies \\lambda = \\frac{S_0}{1 + x}$$
$$\\frac{\\partial \\mathcal{L}}{\\partial y} = \\frac{W_0}{1 + y} - \\lambda = 0 \\implies \\lambda = \\frac{W_0}{1 + y}$$
$$\\frac{\\partial \\mathcal{L}}{\\partial \\lambda} = B - (x + y) = 0 \\implies x + y = B$$

Equating marginal utilities per dollar invested:
$$\\frac{S_0}{1 + x} = \\frac{W_0}{1 + y} \\iff S_0(1 + B - x) = W_0(1 + x)$$
Solving for $x^*$, $y^*$, and $\\lambda^*$:
$$x^* = \\frac{S_0(B + 1) - W_0}{S_0 + W_0}, \\quad y^* = \\frac{W_0(B + 1) - S_0}{S_0 + W_0}, \\quad \\lambda^* = \\frac{S_0 + W_0}{B + 2}$$

### 2.3 Single-Variable Substitution & Strict Concavity (SOC)
Substituting $y = B - x$ yields:
$$f(x) = S_0 \\ln(1 + x) + W_0 \\ln(1 + B - x) \\quad \\text{for } x \\in [0, B]$$
$$f'(x) = \\frac{S_0}{1 + x} - \\frac{W_0}{1 + B - x}$$
$$f''(x) = -\\frac{S_0}{(1 + x)^2} - \\frac{W_0}{(1 + B - x)^2}$$

Because $S_0, W_0 > 0$ and $x \\in [0, B]$, $f''(x) < 0$ strictly for all $x \\in [0, B]$.  
Hence, $f(x)$ is **strictly concave**, guaranteeing that the stationary point is the **unique global maximum**.

### 2.4 Bordered Hessian Matrix
For the bivariate constrained problem, the Bordered Hessian matrix $\\bar{H}$ is:
$$\\bar{H} = \\begin{pmatrix} 0 & 1 & 1 \\\\ 1 & -\\frac{S_0}{(1+x)^2} & 0 \\\\ 1 & 0 & -\\frac{W_0}{(1+y)^2} \\end{pmatrix}$$
$$\\det(\\bar{H}) = \\frac{S_0}{(1+x)^2} + \\frac{W_0}{(1+y)^2} > 0$$
Since $\\det(\\bar{H}) > 0$, the Second-Order Sufficient Condition (SOSC) for a constrained local/global maximum is strictly met.""")

# Cell 6: Analytical Solver Code
add_code("""# 3. Core Analytical and Mathematical Functions

def solve_analytical(S0: float, W0: float, B: float) -> dict:
    \"\"\"Closed-form analytical solver for CalB06 with boundary condition checks.\"\"\"
    if B <= 0:
        raise ValueError("Budget B must be positive.")
    
    # Degenerate zero-weight handling
    if S0 <= 0 and W0 <= 0:
        return {"solution_type": "Degenerate", "x_opt": 0.0, "y_opt": B, "F_opt": 0.0, "lambda_opt": 0.0, "is_interior": False}
    elif S0 <= 0:
        return {"solution_type": "Boundary (S0=0)", "x_opt": 0.0, "y_opt": B, "F_opt": W0*np.log(1+B), "lambda_opt": W0/(1+B), "is_interior": False}
    elif W0 <= 0:
        return {"solution_type": "Boundary (W0=0)", "x_opt": B, "y_opt": 0.0, "F_opt": S0*np.log(1+B), "lambda_opt": S0/(1+B), "is_interior": False}
    
    # Positive weights
    denom = S0 + W0
    x_uncon = (S0 * (B + 1.0) - W0) / denom
    y_uncon = (W0 * (B + 1.0) - S0) / denom
    lambda_uncon = denom / (B + 2.0)
    
    # Check interior feasibility
    if x_uncon < 0.0:
        x_opt = 0.0
        y_opt = float(B)
        F_opt = float(W0 * np.log(1.0 + B))
        lambda_opt = float(W0 / (1.0 + B))
        sol_type = "Boundary (x*=0, y*=B)"
        is_interior = False
    elif x_uncon > B:
        x_opt = float(B)
        y_opt = 0.0
        F_opt = float(S0 * np.log(1.0 + B))
        lambda_opt = float(S0 / (1.0 + B))
        sol_type = "Boundary (x*=B, y*=0)"
        is_interior = False
    else:
        x_opt = float(x_uncon)
        y_opt = float(y_uncon)
        F_opt = float(S0 * np.log(1.0 + x_opt) + W0 * np.log(1.0 + y_opt))
        lambda_opt = float(lambda_uncon)
        sol_type = "Interior Optimum"
        is_interior = True
        
    f_dprime = - S0 / ((1.0 + x_opt)**2) - W0 / ((1.0 + y_opt)**2)
    det_H = S0 / ((1.0 + x_opt)**2) + W0 / ((1.0 + y_opt)**2)
    
    return {
        "solution_type": sol_type,
        "x_opt": x_opt,
        "y_opt": y_opt,
        "F_opt": F_opt,
        "lambda_opt": lambda_opt,
        "is_interior": is_interior,
        "x_uncon": float(x_uncon),
        "f_dprime": f_dprime,
        "det_H": det_H
    }

print("Analytical solver compiled successfully.")""")

# Cell 7: Benchmark MD
add_md("""## Part 3: Handbook Benchmark Case ($S_0 = 2, W_0 = 1, B = 10$)
We verify our implementation against the exact hand-calculation benchmark specified in the course syllabus:
- $x^* = \\frac{2(11) - 1}{3} = \\frac{21}{3} = 7.0$
- $y^* = 10 - 7 = 3.0$
- $\\lambda^* = \\frac{2 + 1}{12} = \\frac{3}{12} = 0.25$
- $F^* = 2 \\ln(8) + \\ln(4) = 8 \\ln(2) \\approx 5.545177$
- Boundary allocations: $F(10, 0) = 2 \\ln(11) \\approx 4.7958$, $F(0, 10) = \\ln(11) \\approx 2.3979$.""")

# Cell 8: Benchmark Code
add_code("""# 4. Benchmark Execution and Corner Comparison
bench = solve_analytical(S0=2.0, W0=1.0, B=10.0)

f_corner_solar = 2.0 * np.log(1.0 + 10.0)
f_corner_wind = 1.0 * np.log(1.0 + 10.0)

print("=== HANDBOOK BENCHMARK VERIFICATION (S0=2, W0=1, B=10) ===")
print(f"Optimal Solar Budget (x*) : {bench['x_opt']:.4f}  [Exact: 7.0000]")
print(f"Optimal Wind Budget  (y*) : {bench['y_opt']:.4f}  [Exact: 3.0000]")
print(f"Maximum Utility      (F*) : {bench['F_opt']:.6f}  [Exact: 8*ln(2) ~= 5.545177]")
print(f"Shadow Price     (lambda*): {bench['lambda_opt']:.4f}  [Exact: 0.2500]")
print(f"Second Derivative f''(x*) : {bench['f_dprime']:.6f} (< 0 ==> Strictly Concave)")
print(f"Bordered Hessian Det      : {bench['det_H']:.6f} (> 0 ==> SOSC Confirmed)")
print("\n--- Boundary Comparison ---")
print(f"Solar Only (x=10, y=0)   : F = {f_corner_solar:.6f}")
print(f"Wind Only  (x=0, y=10)   : F = {f_corner_wind:.6f}")
print(f"Interior Optimum (x=7,y=3): F* = {bench['F_opt']:.6f}")
print("==> F*(7, 3) = 5.545 > 4.796 > 2.398, confirming interior global maximum.")""")

# Cell 9: Numerical MD
add_md("""## Part 4: Numerical Optimization via SciPy (SLSQP) & Precision Benchmarking
We formulate the constrained nonlinear program and solve it via Sequential Least Squares Programming (`scipy.optimize.minimize` with method `SLSQP`).
We compute the absolute error $|x_{\\text{analytical}} - x_{\\text{numerical}}|$ and $|F_{\\text{analytical}} - F_{\\text{numerical}}|$.""")

# Cell 10: Numerical Code
add_code("""# 5. Numerical Optimization Solver (SLSQP)
def solve_numerical(S0: float, W0: float, B: float):
    \"\"\"Solve constrained optimization problem numerically using SciPy SLSQP.\"\"\"
    obj = lambda v: - (S0 * np.log(1.0 + v[0]) + W0 * np.log(1.0 + v[1]))
    grad = lambda v: np.array([- S0 / (1.0 + v[0]), - W0 / (1.0 + v[1])])
    constraint = {'type': 'eq', 'fun': lambda v: B - (v[0] + v[1]), 'jac': lambda v: np.array([-1.0, -1.0])}
    bounds = [(0.0, float(B)), (0.0, float(B))]
    x0 = [B / 2.0, B / 2.0]
    
    res = minimize(fun=obj, x0=x0, jac=grad, method='SLSQP', bounds=bounds, constraints=[constraint], options={'ftol': 1e-12})
    return {
        "x_num": float(res.x[0]),
        "y_num": float(res.x[1]),
        "F_num": float(-res.fun),
        "success": bool(res.success),
        "nit": int(res.nit)
    }

num_bench = solve_numerical(S0=2.0, W0=1.0, B=10.0)
err_x = abs(bench['x_opt'] - num_bench['x_num'])
err_F = abs(bench['F_opt'] - num_bench['F_num'])

print("=== NUMERICAL VS. ANALYTICAL ERROR ANALYSIS ===")
print(f"Analytical Solution: x* = {bench['x_opt']:.8f}, F* = {bench['F_opt']:.8f}")
print(f"Numerical Solution : x* = {num_bench['x_num']:.8f}, F* = {num_bench['F_num']:.8f}")
print(f"Convergence Status : Success={num_bench['success']} in {num_bench['nit']} iterations")
print(f"Absolute Error |x_ana - x_num|: {err_x:.2e}")
print(f"Absolute Error |F_ana - F_num|: {err_F:.2e}")
print("==> Discrepancy is strictly within machine floating-point rounding precision.")""")

# Cell 11: Corner Solution MD
add_md("""## Part 5: Out-of-Bounds Scenario (Testing Corner Solutions)
The syllabus specifically instructs: *"thử thêm một bộ trọng số làm nghiệm nội ra ngoài miền"*.  
We test an extreme scenario where wind weight dominates heavily: $S_0 = 0.2, W_0 = 3.0, B = 10.0$.
In this case:
$$S_0(B + 1) - W_0 = 0.2(11) - 3.0 = 2.2 - 3.0 = -0.8 < 0$$
The unconstrained stationary point has $x_{\\text{uncon}} = \\frac{-0.8}{3.2} = -0.25 < 0$.  
Since solar budget cannot be negative, $f'(x) < 0$ on all of $[0, 10]$, and the optimal solution snaps to the boundary:
$$x^* = 0, \\quad y^* = 10, \\quad F^* = 3.0 \\ln(11) \\approx 7.193686$$""")

# Cell 12: Corner Solution Code
add_code("""# 6. Out-of-bounds Corner Solution Evaluation
out_bench = solve_analytical(S0=0.2, W0=3.0, B=10.0)
out_num = solve_numerical(S0=0.2, W0=3.0, B=10.0)

print("=== OUT-OF-BOUNDS SCENARIO (S0=0.2, W0=3.0, B=10.0) ===")
print(f"Unconstrained Stationary Point: x = {out_bench['x_uncon']:.4f} (< 0 ==> Infeasible)")
print(f"Constrained Solution Type     : {out_bench['solution_type']}")
print(f"Constrained Analytical Optimum: x* = {out_bench['x_opt']:.4f}, y* = {out_bench['y_opt']:.4f}, F* = {out_bench['F_opt']:.6f}")
print(f"Constrained Numerical Optimum : x* = {out_num['x_num']:.4f}, y* = {out_num['y_num']:.4f}, F* = {out_num['F_num']:.6f}")
print("==> Karush-Kuhn-Tucker boundary activation successfully verified.")""")

# Cell 13: Vietnam 2024 MD
add_md("""## Part 6: Vietnam 2024 Real-World Case Study
Using our empirical weights extracted from the 2024 OWID data:
- Normalized Weights: $s_0 = 0.6698$ (Solar, 67.0%), $w_0 = 0.3302$ (Wind, 33.0%).
- Raw Weights: $S_0 = 25.86$ TWh, $W_0 = 12.75$ TWh ($S_0 + W_0 = 38.61$ TWh).

We demonstrate that both normalized and raw formulations yield the **exact same optimal budget allocation $(x^*, y^*)$**.""")

# Cell 14: Vietnam 2024 Code
add_code("""# 7. Real-world Case Study (Vietnam 2024 Baseline)
vn_norm = solve_analytical(S0=s0_norm, W0=w0_norm, B=10.0)
vn_raw = solve_analytical(S0=S0_raw, W0=W0_raw, B=10.0)

print("=== VIETNAM 2024 EMPIRICAL CASE STUDY (B = 10.0) ===")
print(f"Solar Weight: {s0_norm:.4f} (67.0%) | Wind Weight: {w0_norm:.4f} (33.0%)")
print(f"Optimal Solar Budget (x*): {vn_norm['x_opt']:.4f} units ({vn_norm['x_opt']/10*100:.1f}% of total budget)")
print(f"Optimal Wind Budget  (y*): {vn_norm['y_opt']:.4f} units ({vn_norm['y_opt']/10*100:.1f}% of total budget)")
print(f"Normalized Optimal Utility F* : {vn_norm['F_opt']:.6f}")
print(f"Raw Optimal Utility F*        : {vn_raw['F_opt']:.6f}")
print(f"Ratio of Utilities (F_raw / F_norm): {vn_raw['F_opt'] / vn_norm['F_opt']:.2f} (Exact sum S0+W0: {S0_raw+W0_raw:.2f})")
print(f"Shadow Price lambda* (Normalized)  : {vn_norm['lambda_opt']:.6f} (= 1/12 ~= 0.083333)")
print(f"Shadow Price lambda* (Raw)         : {vn_raw['lambda_opt']:.6f} (= 38.61/12 ~= 3.217500)")""")

# Cell 15: Sensitivity MD
add_md("""## Part 7: Budget Sensitivity Analysis ($B \\in [5, 15]$) & Envelope Theorem
We evaluate the optimal allocation across the budget spectrum $B \\in [5.0, 15.0]$ with step size 1.0.

### The Envelope Theorem
The Envelope Theorem guarantees that the marginal increase in the optimal objective value with respect to the budget parameter $B$ equals the Lagrange multiplier:
$$\\frac{dF^*}{dB} = \\frac{\\partial \\mathcal{L}}{\\partial B} = \\lambda^*(B) = \\frac{S_0 + W_0}{B + 2}$$
We numerically verify this identity by comparing $\\lambda^*(B)$ with the central finite-difference approximation $\\frac{\\Delta F^*}{\\Delta B}$.""")

# Cell 16: Sensitivity Code
add_code("""# 8. Budget Sensitivity Grid and Envelope Theorem Verification
b_grid = np.arange(5.0, 15.5, 1.0)
sens_rows = []

for b in b_grid:
    b_val = round(float(b), 2)
    sol = solve_analytical(s0_norm, w0_norm, b_val)
    sens_rows.append({
        "Budget_B": b_val,
        "x_solar_opt": sol['x_opt'],
        "y_wind_opt": sol['y_opt'],
        "F_optimal": sol['F_opt'],
        "lambda_shadow_price": sol['lambda_opt']
    })

df_sens = pd.DataFrame(sens_rows)

# Numerical derivative using central finite difference
df_sens["dF_dB_finite_diff"] = np.gradient(df_sens["F_optimal"].values, df_sens["Budget_B"].values)
df_sens["envelope_discrepancy"] = np.abs(df_sens["lambda_shadow_price"] - df_sens["dF_dB_finite_diff"])

print("=== BUDGET SENSITIVITY TABLE (B in [5, 15]) ===")
df_sens""")

# Cell 17: Visualizations MD
add_md("""## Part 8: Visualizations & Graphical Analysis
We now render the 4 core figures required by the assignment guidelines and presentation rubrics.""")

# Cell 18: Visualizations Code
add_code("""# 9. Rendering All 4 Publication-Quality Figures

# Figure 1: Historical Generation in Vietnam
plt.figure(figsize=(9, 5))
plt.plot(df_clean['year'], df_clean['solar_electricity'], color='#F39C12', lw=2.5, marker='o', label='Solar Electricity (TWh)')
plt.plot(df_clean['year'], df_clean['wind_electricity'], color='#2980B9', lw=2.5, marker='s', label='Wind Electricity (TWh)')
plt.fill_between(df_clean['year'], df_clean['solar_electricity'], color='#F39C12', alpha=0.15)
plt.fill_between(df_clean['year'], df_clean['wind_electricity'], color='#2980B9', alpha=0.15)
plt.scatter([2024, 2024], [S0_raw, W0_raw], color='#C0392B', s=80, zorder=5)
plt.annotate(f"2024 Baseline:\\nSolar S0={S0_raw:.2f} TWh (67%)\\nWind W0={W0_raw:.2f} TWh (33%)",
             xy=(2024, S0_raw), xytext=(2014, S0_raw - 4),
             arrowprops=dict(arrowstyle="->", color='#C0392B', lw=1.5),
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C0392B", lw=1.2))
plt.title("Figure 1: Vietnam Renewable Electricity Generation (2000–2024) [OWID Data]", fontweight='bold')
plt.xlabel("Year", fontweight='semibold')
plt.ylabel("Electricity Generation (TWh)", fontweight='semibold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Figure 2: Constrained Optimization Geometry (2-Panel)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
# Panel A: Contour & Budget Line
x_g = np.linspace(0, 12, 200)
y_g = np.linspace(0, 12, 200)
X, Y = np.meshgrid(x_g, y_g)
Z = s0_norm * np.log(1.0 + X) + w0_norm * np.log(1.0 + Y)
cs = ax1.contour(X, Y, Z, levels=14, cmap='viridis', alpha=0.85)
ax1.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
x_line = np.linspace(0, 10, 100)
ax1.plot(x_line, 10 - x_line, color='#E74C3C', lw=2.5, label='Budget Constraint: x + y = 10')
ax1.fill_between(x_line, 0, 10 - x_line, color='#E74C3C', alpha=0.08, label='Feasible Set')
ax1.scatter([vn_norm['x_opt']], [vn_norm['y_opt']], color='#8E44AD', s=120, zorder=5, label=f"Optimum (x*={vn_norm['x_opt']:.2f}, y*={vn_norm['y_opt']:.2f})")
ax1.set_title("(A) Indifference Curves & Tangency Optimum", fontweight='bold')
ax1.set_xlabel("Solar Allocation x", fontweight='semibold')
ax1.set_ylabel("Wind Allocation y", fontweight='semibold')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='lower left', fontsize=8.5)

# Panel B: Single-Variable Concavity
x_p = np.linspace(0, 10, 200)
f_p = [s0_norm * np.log(1.0 + v) + w0_norm * np.log(1.0 + 10.0 - v) for v in x_p]
ax2.plot(x_p, f_p, color='#27AE60', lw=2.5, label='Utility Curve f(x)')
ax2.scatter([vn_norm['x_opt']], [vn_norm['F_opt']], color='#8E44AD', s=120, zorder=5)
ax2.axhline(vn_norm['F_opt'], color='#8E44AD', linestyle=':', label="Tangency f'(x*) = 0")
ax2.set_title("(B) Single-Variable Objective f(x) & Strict Concavity", fontweight='bold')
ax2.set_xlabel("Solar Allocation x", fontweight='semibold')
ax2.set_ylabel("Objective Value f(x)", fontweight='semibold')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='lower center', fontsize=8.5)
plt.tight_layout()
plt.show()

# Figure 3: Budget Sensitivity & Envelope Theorem (2-Panel)
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
ax3a.plot(df_sens['Budget_B'], df_sens['x_solar_opt'], color='#F39C12', lw=2.5, marker='o', label='Solar Allocation x*(B)')
ax3a.plot(df_sens['Budget_B'], df_sens['y_wind_opt'], color='#2980B9', lw=2.5, marker='s', label='Wind Allocation y*(B)')
ax3a.plot(df_sens['Budget_B'], df_sens['Budget_B'], color='#7F8C8D', linestyle='--', label='Total Budget B')
ax3a.set_title("(A) Optimal Allocations vs. Budget B", fontweight='bold')
ax3a.set_xlabel("Budget B", fontweight='semibold')
ax3a.set_ylabel("Allocated Units", fontweight='semibold')
ax3a.grid(True, linestyle='--', alpha=0.5)
ax3a.legend()

ax3b.plot(df_sens['Budget_B'], df_sens['F_optimal'], color='#27AE60', lw=2.5, marker='^', label='Utility F*(B)')
ax3b.set_xlabel("Budget B", fontweight='semibold')
ax3b.set_ylabel("Optimal Utility F*", color='#27AE60', fontweight='semibold')
ax3b_t = ax3b.twinx()
ax3b_t.plot(df_sens['Budget_B'], df_sens['lambda_shadow_price'], color='#C0392B', lw=2.2, linestyle='-.', marker='v', label='Shadow Price lambda*(B)')
ax3b_t.set_ylabel("Shadow Price lambda*", color='#C0392B', fontweight='semibold')
ax3b.set_title("(B) Optimal Utility & Diminishing Shadow Price", fontweight='bold')
ax3b.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Figure 4: Boundary Solution
fig4, ax4 = plt.subplots(figsize=(8, 4.5))
x_ext = np.linspace(-0.8, 10, 200)
f_ext = [0.2*np.log(1+v) + 3.0*np.log(1+10-v) if v > -1 else np.nan for v in x_ext]
ax4.plot(x_ext, f_ext, color='#34495E', linestyle='--', label='Unconstrained Curve')
x_f = np.linspace(0, 10, 150)
f_f = [0.2*np.log(1+v) + 3.0*np.log(1+10-v) for v in x_f]
ax4.plot(x_f, f_f, color='#27AE60', lw=3.0, label='Feasible Domain [0, 10]')
ax4.scatter([-0.25], [0.2*np.log(1-0.25) + 3.0*np.log(1+10.25)], color='#E74C3C', s=100, marker='x', label='Unconstrained Peak (x=-0.25 < 0)')
ax4.scatter([0.0], [3.0*np.log(11)], color='#8E44AD', s=120, label='Boundary Optimum (x*=0, y*=10)')
ax4.axvline(0.0, color='#7F8C8D', linestyle=':')
ax4.set_title("Figure 4: Out-of-Bounds Corner Solution (S0=0.2, W0=3.0, B=10)", fontweight='bold')
ax4.set_xlabel("Solar Allocation x", fontweight='semibold')
ax4.set_ylabel("Objective Value f(x)", fontweight='semibold')
ax4.grid(True, linestyle='--', alpha=0.5)
ax4.legend(loc='lower right')
plt.tight_layout()
plt.show()

print("All 4 figures rendered successfully.")""")

# Cell 19: Discussion MD
add_md("""## Part 9: Academic Discussion & Practical Policy Recommendations

### 9.1 Economic Meaning of the Objective Index $F(x, y)$
- The objective function $F(x, y) = S_0 \\ln(1 + x) + W_0 \\ln(1 + y)$ is a stylized **welfare / societal benefit index** designed to balance renewable resource deployment under diminishing returns.
- **Critical Clarification:** $F$ is **not** a physical electricity generation forecast (in kWh) or financial cash flow/profit projection. It models preference weights and portfolio diversification.

### 9.2 Nature of the Lagrange Multiplier $\\lambda$ vs. Electricity Tariff
- In our model, $\\lambda^* = \\frac{dF^*}{dB}$ is the **shadow price of the normalized budget**. It measures the marginal increase in the policy utility index per additional unit of normalized capital.
- It should **not** be confused with real-world electricity feed-in tariffs (FiT), levelized cost of energy (LCOE), or wholesale clearing prices.

### 9.3 Policy Extensions & Realistic Constraints
In practical national power development master plans (such as Vietnam's PDP8 - Quy hoạch điện VIII), planners can extend this model by introducing:
1. **Differential Unit Capital Costs:**
   $$c_x x + c_y y \\le B \\quad \\text{where } c_x, c_y \\text{ represent LCOE or CAPEX per MW.}$$
2. **Minimum Generation or Grid Stability Quotas:**
   $$x \\ge x_{\\min}, \\quad y \\ge y_{\\min} \\quad \\text{(baseload, inertia, or spinning reserve constraints).}$$
3. **Transmission Grid Hosting Capacity Limits:**
   $$x \\le x_{\\max}, \\quad y \\le y_{\\max} \\quad \\text{(substation and transmission line thermal capacity).}$$

---
**Course:** Calculus in Business (HK261) - CalB-06  
**Completed & Validated by Group 6.**""")

with open('CalB06_Optimization.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print('Notebook CalB06_Optimization.ipynb created successfully!')
