"""
Script to build the streamlined CalB06 Google Colab / Jupyter Notebook.
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

# Cell 1: Header
add_md("""# CalB06: Wind & Solar Power Budget Allocation under Constraints
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/triduc-tran/CalB06_Wind_Solar_Optimization/blob/main/CalB06_Optimization.ipynb)

**Course:** Calculus in Business (HK261) - Project 6  
**Topic:** CalB-06 – Điện gió, điện mặt trời và phân bổ ngân sách có ràng buộc  

---

## 1. Problem Description & Mathematical Model
When investment budget is limited, planners must determine how to divide capital between solar power ($x$) and wind power ($y$). Because each additional unit of investment faces diminishing marginal returns, concentrating the entire budget into a single option does not necessarily maximize aggregate benefits.

We model this problem as:
$$\\max_{x, y \\ge 0} F(x, y) = S_0 \\ln(1 + x) + W_0 \\ln(1 + y) \\quad \\text{subject to} \\quad x + y = B$$

where:
- $x, y$: Normalized budget allocated to Solar and Wind ($x, y \\ge 0$).
- $B$: Total normalized budget ($B \\in [5, 15]$).
- $S_0, W_0$: Non-negative weights from Vietnam's empirical 2024 electricity generation (*Our World in Data*).
- $\\ln(1 + \\cdot)$: Ensures zero investment produces zero utility ($F(0, 0) = 0$).""")

# Cell 2: Imports
add_code("""# Setup standard libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

print("Environment initialized successfully.")""")

# Cell 3: Task 1 MD
add_md("""## Part 1 (Research Task 1): Data Ingestion & 2024 Baseline Weights
We read the cleaned historical dataset from Our World in Data (OWID) for Vietnam (2000–2024) and extract the solar ($S_0$) and wind ($W_0$) generation values in 2024.""")

# Cell 4: Task 1 Code
add_code("""# Load CSV data
csv_file = "vietnam_solar_wind_cleaned.csv"
df_raw = pd.read_csv(csv_file)

# Filter valid historical years
df = df_raw.dropna(subset=['country', 'solar_electricity', 'wind_electricity']).copy()
df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
df = df.sort_values(by='year').reset_index(drop=True)

# Extract 2024 empirical values (in TWh)
row_2024 = df[df['year'] == 2024].iloc[0]
S0_raw = float(row_2024['solar_electricity'])  # 25.86 TWh
W0_raw = float(row_2024['wind_electricity'])   # 12.75 TWh
total_2024 = S0_raw + W0_raw                    # 38.61 TWh

# Normalized weights (sum = 1.0)
s0_norm = S0_raw / total_2024  # approx 0.670
w0_norm = W0_raw / total_2024  # approx 0.330

print(f"Data period: {df['year'].min()} - {df['year'].max()} ({len(df)} years)")
print(f"2024 Solar Generation (S0): {S0_raw:.2f} TWh (Normalized weight s0 = {s0_norm:.4f})")
print(f"2024 Wind Generation  (W0): {W0_raw:.2f} TWh (Normalized weight w0 = {w0_norm:.4f})")
print(f"2024 Total Generation     : {total_2024:.2f} TWh")
df.tail(5)""")

# Cell 5: Task 2, 3, 4 MD
add_md("""## Part 2 (Research Tasks 2, 3, & 4): Lagrange Formulation & Analytical Solver

### First-Order Conditions (FOC):
Using the Lagrangian $\\mathcal{L}(x, y, \\lambda) = S_0 \\ln(1 + x) + W_0 \\ln(1 + y) - \\lambda(x + y - B)$:
$$\\frac{S_0}{1 + x} = \\lambda, \\quad \\frac{W_0}{1 + y} = \\lambda, \\quad x + y = B$$
Solving yields the closed-form stationary points:
$$x^* = \\frac{S_0(B + 1) - W_0}{S_0 + W_0}, \\quad y^* = \\frac{W_0(B + 1) - S_0}{S_0 + W_0}, \\quad \\lambda^* = \\frac{S_0 + W_0}{B + 2}$$

### Second-Order Conditions (SOC) - Proof of Strict Concavity:
Substituting $y = B - x$:
$$f(x) = S_0 \\ln(1 + x) + W_0 \\ln(1 + B - x)$$
$$f''(x) = -\\frac{S_0}{(1 + x)^2} - \\frac{W_0}{(1 + B - x)^2} < 0 \\quad \\forall x \\in [0, B]$$
Since $f''(x) < 0$ everywhere, $f(x)$ is **strictly concave**, proving that the stationary point is the **unique global maximum**.

### Boundary Solutions (Task 4):
- If $x^* < 0$, optimum snaps to the boundary: $x^* = 0, y^* = B$.
- If $x^* > B$, optimum snaps to the boundary: $x^* = B, y^* = 0$.""")

# Cell 6: Solver Code
add_code("""def solve_allocation(S0, W0, B):
    \"\"\"Analytical solver with boundary checking.\"\"\"
    denom = S0 + W0
    x_uncon = (S0 * (B + 1.0) - W0) / denom
    y_uncon = (W0 * (B + 1.0) - S0) / denom
    lambda_uncon = denom / (B + 2.0)
    
    # Boundary check (Task 4)
    if x_uncon < 0.0:
        x_opt = 0.0
        y_opt = float(B)
        lambda_opt = W0 / (1.0 + B)
        status = "Boundary (x*=0, y*=B)"
    elif x_uncon > B:
        x_opt = float(B)
        y_opt = 0.0
        lambda_opt = S0 / (1.0 + B)
        status = "Boundary (x*=B, y*=0)"
    else:
        x_opt = float(x_uncon)
        y_opt = float(y_uncon)
        lambda_opt = float(lambda_uncon)
        status = "Interior Optimum"

    F_opt = S0 * np.log(1.0 + x_opt) + W0 * np.log(1.0 + y_opt)
    f_dprime = - S0 / ((1.0 + x_opt)**2) - W0 / ((1.0 + y_opt)**2)

    return {
        'x': x_opt, 'y': y_opt, 'F': F_opt, 
        'lambda': lambda_opt, 'f_dprime': f_dprime, 
        'status': status, 'x_uncon': float(x_uncon)
    }

print("Solver function defined.")""")

# Cell 7: Benchmark MD & Code
add_md("""## Part 3 (Section 6): Handbook Benchmark Verification ($S_0 = 2, W_0 = 1, B = 10$)
We compare our model against the hand-calculation benchmark specified in the course guide:
- $x^* = \\frac{2(11) - 1}{3} = 7.0$
- $y^* = 10 - 7 = 3.0$
- $\\lambda^* = \\frac{2 + 1}{12} = 0.25$
- $F^* = 2\\ln(8) + \\ln(4) = 8\\ln(2) \\approx 5.545177$""")

add_code("""# Handbook Benchmark
bench = solve_allocation(S0=2.0, W0=1.0, B=10.0)

# Boundary values
F_corner_solar = 2.0 * np.log(1.0 + 10.0)
F_corner_wind  = 1.0 * np.log(1.0 + 10.0)

print(f"Status                  : {bench['status']}")
print(f"Optimal Solar Budget (x*): {bench['x']:.4f}  [Exact: 7.0]")
print(f"Optimal Wind Budget  (y*): {bench['y']:.4f}  [Exact: 3.0]")
print(f"Maximum Utility Value(F*): {bench['F']:.6f}  [Exact: 8*ln(2) ~= 5.545177]")
print(f"Shadow Price (lambda*)   : {bench['lambda']:.4f}  [Exact: 0.25]")
print(f"Second Derivative f''(x*): {bench['f_dprime']:.6f} (< 0 ==> Strictly Concave)")
print(f"\\nBoundary Comparison:")
print(f"  - Corner x=10, y=0 : F = {F_corner_solar:.6f}")
print(f"  - Corner x=0,  y=10: F = {F_corner_wind:.6f}")
print(f"  - Interior x=7, y=3: F* = {bench['F']:.6f}")
print("  ==> F*(7, 3) = 5.545 > 4.796 > 2.398 (Interior is verified superior!)")""")

# Cell 8: Out of bounds MD & Code
add_md("""## Part 4 (Section 6): Out-of-Bounds Test Case ($S_0 = 0.2, W_0 = 3.0, B = 10$)
The guide asks us to test a set of weights that pushes the unconstrained stationary point outside $[0, B]$:
$$x_{\\text{uncon}} = \\frac{0.2(11) - 3.0}{3.2} = -0.25 < 0$$
Since solar investment cannot be negative, the optimum snaps to the boundary: $x^* = 0, y^* = 10$.""")

add_code("""out_test = solve_allocation(S0=0.2, W0=3.0, B=10.0)
print(f"Unconstrained Stationary Point: x = {out_test['x_uncon']:.4f} (< 0, Infeasible)")
print(f"Activated Solution Type       : {out_test['status']}")
print(f"Constrained Optimum           : x* = {out_test['x']:.4f}, y* = {out_test['y']:.4f}")
print(f"Maximum Utility Value         : F* = {out_test['F']:.6f} (= 3*ln(11) ~= 7.193686)")""")

# Cell 9: Vietnam Case MD & Code
add_md("""## Part 5: Empirical Vietnam 2024 Case Study ($B = 10$)
Using the normalized weights ($s_0 = 0.670, w_0 = 0.330$):""")

add_code("""vn_norm = solve_allocation(S0=s0_norm, W0=w0_norm, B=10.0)
vn_raw  = solve_allocation(S0=S0_raw,  W0=W0_raw,  B=10.0)

print(f"Optimal Solar Allocation (x*): {vn_norm['x']:.4f} units ({vn_norm['x']/10*100:.1f}% of budget)")
print(f"Optimal Wind Allocation  (y*): {vn_norm['y']:.4f} units ({vn_norm['y']/10*100:.1f}% of budget)")
print(f"Maximum Utility Value    (F*): {vn_norm['F']:.6f}")
print(f"Shadow Price        (lambda*): {vn_norm['lambda']:.6f} (= 1/12 ~= 0.0833)")
print(f"\\nNote: Under raw weights, allocation is exactly the same: x* = {vn_raw['x']:.4f}, y* = {vn_raw['y']:.4f}")""")

# Cell 10: SciPy Optimization MD & Code
add_md("""## Part 6 (Section 6): Numerical Verification via SciPy (SLSQP)
We verify that the analytical formulas match numerical optimization using `scipy.optimize.minimize`.""")

add_code("""def solve_numerical(S0, W0, B):
    obj = lambda vars: - (S0 * np.log(1.0 + vars[0]) + W0 * np.log(1.0 + vars[1]))
    cons = {'type': 'eq', 'fun': lambda vars: B - (vars[0] + vars[1])}
    bounds = [(0.0, float(B)), (0.0, float(B))]
    res = minimize(obj, x0=[B/2.0, B/2.0], method='SLSQP', bounds=bounds, constraints=[cons])
    return res.x[0], res.x[1], -res.fun

x_num, y_num, F_num = solve_numerical(S0=2.0, W0=1.0, B=10.0)
print(f"Analytical Solution: x* = {bench['x']:.6f}, F* = {bench['F']:.6f}")
print(f"Numerical Solution : x* = {x_num:.6f}, F* = {F_num:.6f}")
print(f"Absolute Error |x_exact - x_num|: {abs(bench['x'] - x_num):.2e}")
print(f"Absolute Error |F_exact - F_num|: {abs(bench['F'] - F_num):.2e}")""")

# Cell 11: Sensitivity MD & Code
add_md("""## Part 7 (Research Task 5): Budget Sensitivity Analysis ($B \\in [5, 15]$)
We compute optimal allocations across budget levels from 5 to 15.""")

add_code("""b_list = list(range(5, 16))
results = []

for B in b_list:
    sol = solve_allocation(s0_norm, w0_norm, B)
    results.append({
        'Budget_B': B,
        'x_Solar': round(sol['x'], 4),
        'y_Wind': round(sol['y'], 4),
        'F_Optimal': round(sol['F'], 6),
        'Shadow_Price_lambda': round(sol['lambda'], 6)
    })

df_sens = pd.DataFrame(results)
display(df_sens)""")

# Cell 12: Visualizations MD & Code
add_md("""## Part 8: Visualizations (3 Core Figures)""")

add_code("""# Figure 1: Historical Generation in Vietnam
plt.figure(figsize=(9, 4.5))
plt.plot(df['year'], df['solar_electricity'], color='#F39C12', lw=2.5, marker='o', label='Solar Power (TWh)')
plt.plot(df['year'], df['wind_electricity'], color='#2980B9', lw=2.5, marker='s', label='Wind Power (TWh)')
plt.scatter([2024, 2024], [S0_raw, W0_raw], color='#C0392B', s=80, zorder=5)
plt.annotate(f"2024 Baseline:\\nSolar S0={S0_raw:.2f} TWh\\nWind W0={W0_raw:.2f} TWh",
             xy=(2024, S0_raw), xytext=(2014, S0_raw - 4),
             arrowprops=dict(arrowstyle="->", color='#C0392B', lw=1.5),
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C0392B"))
plt.title("Figure 1: Vietnam Solar and Wind Electricity Generation (2000–2024)", fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Electricity Generation (TWh)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Figure 2: Optimization Geometry (Contour + 1-Variable Function)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# (A) Contour
x_grid = np.linspace(0, 12, 150)
y_grid = np.linspace(0, 12, 150)
X, Y = np.meshgrid(x_grid, y_grid)
Z = s0_norm * np.log(1.0 + X) + w0_norm * np.log(1.0 + Y)
cs = ax1.contour(X, Y, Z, levels=12, cmap='viridis', alpha=0.8)
ax1.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
x_line = np.linspace(0, 10, 100)
ax1.plot(x_line, 10 - x_line, color='#E74C3C', lw=2.5, label='Budget: x + y = 10')
ax1.scatter([vn_norm['x']], [vn_norm['y']], color='#8E44AD', s=100, zorder=5, 
            label=f"Optimum (x*={vn_norm['x']:.2f}, y*={vn_norm['y']:.2f})")
ax1.set_title("(A) Indifference Curves & Budget Line", fontweight='bold')
ax1.set_xlabel("Solar Allocation x")
ax1.set_ylabel("Wind Allocation y")
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='lower left')

# (B) Single-variable objective curve
x_vals = np.linspace(0, 10, 200)
f_vals = [s0_norm * np.log(1.0 + v) + w0_norm * np.log(1.0 + 10.0 - v) for v in x_vals]
ax2.plot(x_vals, f_vals, color='#27AE60', lw=2.5, label='Utility Curve f(x)')
ax2.scatter([vn_norm['x']], [vn_norm['F']], color='#8E44AD', s=100, zorder=5)
ax2.axhline(vn_norm['F'], color='#8E44AD', linestyle=':', label="Horizontal Tangent (f'=0)")
ax2.set_title("(B) Single-Variable Objective f(x) (Strictly Concave)", fontweight='bold')
ax2.set_xlabel("Solar Allocation x")
ax2.set_ylabel("Total Utility f(x)")
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='lower center')
plt.tight_layout()
plt.show()

# Figure 3: Budget Sensitivity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# (A) Allocations
ax1.plot(df_sens['Budget_B'], df_sens['x_Solar'], color='#F39C12', lw=2.5, marker='o', label='Solar x*(B)')
ax1.plot(df_sens['Budget_B'], df_sens['y_Wind'], color='#2980B9', lw=2.5, marker='s', label='Wind y*(B)')
ax1.plot(df_sens['Budget_B'], df_sens['Budget_B'], color='#7F8C8D', linestyle='--', label='Total Budget B')
ax1.set_title("(A) Optimal Allocations vs. Budget B", fontweight='bold')
ax1.set_xlabel("Budget B")
ax1.set_ylabel("Allocated Units")
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# (B) Utility & Shadow Price
ax2.plot(df_sens['Budget_B'], df_sens['F_Optimal'], color='#27AE60', lw=2.5, marker='^', label='Utility F*(B)')
ax2.set_xlabel("Budget B")
ax2.set_ylabel("Optimal Utility F*", color='#27AE60')
ax2.tick_params(axis='y', labelcolor='#27AE60')
ax2_twin = ax2.twinx()
ax2_twin.plot(df_sens['Budget_B'], df_sens['Shadow_Price_lambda'], color='#C0392B', lw=2, linestyle='-.', marker='v', label='Shadow Price lambda*')
ax2_twin.set_ylabel("Shadow Price lambda*", color='#C0392B')
ax2_twin.tick_params(axis='y', labelcolor='#C0392B')
ax2.set_title("(B) Optimal Utility F* & Diminishing Shadow Price", fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()""")

with open('CalB06_Optimization.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook CalB06_Optimization.ipynb built successfully!")
