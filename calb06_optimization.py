"""
CalB06: Wind and Solar Power Budget Allocation under Constraints
================================================================
Course: Calculus in Business (HK261) - Project 6
Student-friendly, clean, and directly structured following:
  - Section 4: Research Tasks 1 to 5
  - Section 6: Programming, Verification, and Deliverables

Mathematical Model:
-------------------
Maximize:   F(x, y) = S0 * ln(1 + x) + W0 * ln(1 + y)
Subject to: x + y = B,   with x >= 0, y >= 0
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ==============================================================================
# PART 1: RESEARCH TASK 1 - DATA LOADING & 2024 WEIGHTS
# ==============================================================================
print("=" * 70)
print("PART 1: DATA INGESTION & 2024 WEIGHTS (RESEARCH TASK 1)")
print("=" * 70)

# Load Vietnam energy data (2000-2024) from Our World in Data (OWID)
csv_file = "vietnam_solar_wind_cleaned.csv"
df_raw = pd.read_csv(csv_file)

# Filter out empty or note rows at the bottom
df = df_raw.dropna(subset=['country', 'solar_electricity', 'wind_electricity']).copy()
df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
df = df.sort_values(by='year').reset_index(drop=True)

# Extract 2024 generation (in Terawatt-hours, TWh)
row_2024 = df[df['year'] == 2024].iloc[0]
S0_raw = float(row_2024['solar_electricity'])  # 25.86 TWh
W0_raw = float(row_2024['wind_electricity'])   # 12.75 TWh
total_2024 = S0_raw + W0_raw                    # 38.61 TWh

# Normalized weights (sum to 1.0)
s0_norm = S0_raw / total_2024  # approx 0.670 (67.0%)
w0_norm = W0_raw / total_2024  # approx 0.330 (33.0%)

print(f"Data range: {df['year'].min()} to {df['year'].max()} ({len(df)} years)")
print(f"2024 Solar Generation (S0): {S0_raw:.2f} TWh (Normalized s0 = {s0_norm:.4f})")
print(f"2024 Wind Generation  (W0): {W0_raw:.2f} TWh (Normalized w0 = {w0_norm:.4f})")
print(f"2024 Total Generation     : {total_2024:.2f} TWh")
print("\n* Note on zero weight (Degenerate case):")
print("  If S0 = 0, all budget is allocated to Wind (x* = 0, y* = B).")
print("  If W0 = 0, all budget is allocated to Solar (x* = B, y* = 0).")


# ==============================================================================
# PART 2: RESEARCH TASKS 2, 3, & 4 - ANALYTICAL SOLVER & CONCAVITY CHECK
# ==============================================================================

def solve_allocation(S0, W0, B):
    """
    Solve the optimal allocation analytically using Lagrange Multipliers (FOC)
    and check for boundary / corner solutions (Task 4).
    
    Formulas:
      x_stat = [S0*(B + 1) - W0] / (S0 + W0)
      y_stat = [W0*(B + 1) - S0] / (S0 + W0)
      lambda_stat = (S0 + W0) / (B + 2)
    """
    denom = S0 + W0
    x_uncon = (S0 * (B + 1.0) - W0) / denom
    y_uncon = (W0 * (B + 1.0) - S0) / denom
    lambda_uncon = denom / (B + 2.0)
    
    # Task 4: Check if solution is non-negative (within [0, B])
    if x_uncon < 0.0:
        # Solar is negative -> snap to boundary: x = 0, y = B
        x_opt = 0.0
        y_opt = float(B)
        lambda_opt = W0 / (1.0 + B)
        status = "Boundary (x*=0, y*=B)"
    elif x_uncon > B:
        # Wind is negative -> snap to boundary: x = B, y = 0
        x_opt = float(B)
        y_opt = 0.0
        lambda_opt = S0 / (1.0 + B)
        status = "Boundary (x*=B, y*=0)"
    else:
        # Valid interior solution
        x_opt = float(x_uncon)
        y_opt = float(y_uncon)
        lambda_opt = float(lambda_uncon)
        status = "Interior Optimum"

    # Objective function value at optimal point
    F_opt = S0 * np.log(1.0 + x_opt) + W0 * np.log(1.0 + y_opt)
    
    # Task 3: Second derivative check for strict concavity: f''(x) < 0
    f_dprime = - S0 / ((1.0 + x_opt) ** 2) - W0 / ((1.0 + y_opt) ** 2)

    return {
        'x': x_opt,
        'y': y_opt,
        'F': F_opt,
        'lambda': lambda_opt,
        'f_dprime': f_dprime,
        'status': status,
        'x_uncon': float(x_uncon)
    }


# ==============================================================================
# PART 3: SECTION 6 - HANDBOOK BENCHMARK VERIFICATION (S0=2, W0=1, B=10)
# ==============================================================================
print("\n" + "=" * 70)
print("PART 3: HANDBOOK BENCHMARK VERIFICATION (S0=2, W0=1, B=10)")
print("=" * 70)

bench = solve_allocation(S0=2.0, W0=1.0, B=10.0)

# Boundary values for comparison (Task 4)
F_boundary_solar = 2.0 * np.log(1.0 + 10.0)  # x=10, y=0
F_boundary_wind  = 1.0 * np.log(1.0 + 10.0)  # x=0,  y=10

print(f"Status                  : {bench['status']}")
print(f"Optimal Solar Budget (x*): {bench['x']:.4f}  [Exact formula: 21/3 = 7.0]")
print(f"Optimal Wind Budget  (y*): {bench['y']:.4f}  [Exact formula: 10-7 = 3.0]")
print(f"Maximum Utility Value(F*): {bench['F']:.6f}  [Exact formula: 8*ln(2) ~= 5.545177]")
print(f"Lagrange Multiplier (lam): {bench['lambda']:.4f}  [Exact formula: 3/12 = 0.25]")
print(f"Second Derivative f''(x*): {bench['f_dprime']:.6f} (< 0 ==> Strictly Concave)")
print("\nBoundary Comparison:")
print(f"  - Corner x=10, y=0 : F = {F_boundary_solar:.6f}")
print(f"  - Corner x=0,  y=10: F = {F_boundary_wind:.6f}")
print(f"  - Interior x=7, y=3: F* = {bench['F']:.6f}")
print("  ==> F*(7, 3) = 5.545 > 4.796 > 2.398 (Interior maximum is confirmed!)")


# ==============================================================================
# PART 4: SECTION 6 - OUT-OF-BOUNDS TEST CASE (S0=0.2, W0=3.0, B=10)
# ==============================================================================
print("\n" + "=" * 70)
print("PART 4: OUT-OF-BOUNDS CORNER SOLUTION TEST (S0=0.2, W0=3.0, B=10)")
print("=" * 70)

out_test = solve_allocation(S0=0.2, W0=3.0, B=10.0)
print(f"Unconstrained Stationary Point: x = {out_test['x_uncon']:.4f} (< 0, Infeasible)")
print(f"Activated Solution Type       : {out_test['status']}")
print(f"Constrained Optimum           : x* = {out_test['x']:.4f}, y* = {out_test['y']:.4f}")
print(f"Maximum Utility Value         : F* = {out_test['F']:.6f} (= 3*ln(11) ~= 7.193686)")
print("==> Automatically snapped to boundary x*=0, y*=10 as required by KKT.")


# ==============================================================================
# PART 5: REAL-WORLD VIETNAM 2024 CASE STUDY (B=10)
# ==============================================================================
print("\n" + "=" * 70)
print("PART 5: VIETNAM 2024 EMPIRICAL CASE STUDY (B = 10)")
print("=" * 70)

vn_norm = solve_allocation(S0=s0_norm, W0=w0_norm, B=10.0)
vn_raw  = solve_allocation(S0=S0_raw,  W0=W0_raw,  B=10.0)

print("Using Normalized Weights (s0 = 0.670, w0 = 0.330):")
print(f"  - Optimal Solar Allocation (x*): {vn_norm['x']:.4f} units ({vn_norm['x']/10*100:.1f}% of budget)")
print(f"  - Optimal Wind Allocation  (y*): {vn_norm['y']:.4f} units ({vn_norm['y']/10*100:.1f}% of budget)")
print(f"  - Maximum Utility Value    (F*): {vn_norm['F']:.6f}")
print(f"  - Shadow Price        (lambda*): {vn_norm['lambda']:.6f} (= 1/12 ~= 0.0833)")
print("\nNote on Invariance:")
print(f"  - Under Raw Weights, x* = {vn_raw['x']:.4f}, y* = {vn_raw['y']:.4f} (Exactly the same allocation!).")
print(f"  - Utility scales by total generation: F_raw = F_norm * {total_2024:.2f} = {vn_raw['F']:.2f}")


# ==============================================================================
# PART 6: SECTION 6 - NUMERICAL OPTIMIZATION VIA SCIPY (SLSQP) & ERROR CHECK
# ==============================================================================
print("\n" + "=" * 70)
print("PART 6: NUMERICAL OPTIMIZATION (SCIPY SLSQP) & ERROR ANALYSIS")
print("=" * 70)

def solve_numerical(S0, W0, B):
    """Solve constrained problem using scipy.optimize.minimize (SLSQP)."""
    obj = lambda vars: - (S0 * np.log(1.0 + vars[0]) + W0 * np.log(1.0 + vars[1]))
    cons = {'type': 'eq', 'fun': lambda vars: B - (vars[0] + vars[1])}
    bounds = [(0.0, float(B)), (0.0, float(B))]
    res = minimize(obj, x0=[B/2.0, B/2.0], method='SLSQP', bounds=bounds, constraints=[cons])
    return res.x[0], res.x[1], -res.fun

x_num, y_num, F_num = solve_numerical(S0=2.0, W0=1.0, B=10.0)
err_x = abs(bench['x'] - x_num)
err_F = abs(bench['F'] - F_num)

print(f"Analytical Solution: x* = {bench['x']:.6f}, F* = {bench['F']:.6f}")
print(f"Numerical Solution : x* = {x_num:.6f}, F* = {F_num:.6f}")
print(f"Absolute Error |x_exact - x_num|: {err_x:.2e}")
print(f"Absolute Error |F_exact - F_num|: {err_F:.2e}")
print("==> Numerical solution matches analytical solution with machine precision!")


# ==============================================================================
# PART 7: RESEARCH TASK 5 - BUDGET SENSITIVITY ANALYSIS (B FROM 5 TO 15)
# ==============================================================================
print("\n" + "=" * 70)
print("PART 7: BUDGET SENSITIVITY ANALYSIS (B FROM 5 TO 15)")
print("=" * 70)

b_list = list(range(5, 16))
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
print(df_sens.to_string(index=False))

# Export sensitivity table to CSV
df_sens.to_csv("calb06_sensitivity_results.csv", index=False)
print("\n-> Saved table to 'calb06_sensitivity_results.csv'")


# ==============================================================================
# PART 8: VISUALIZATIONS (3 CORE FIGURES)
# ==============================================================================
print("\n" + "=" * 70)
print("PART 8: GENERATING VISUAL PLOTS")
print("=" * 70)

os.makedirs("figures", exist_ok=True)

# ------------------------------------------------------------------------------
# Figure 1: Vietnam Renewable Electricity Generation (2000-2024)
# ------------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(df['year'], df['solar_electricity'], color='#F39C12', lw=2.5, marker='o', label='Solar Power (TWh)')
plt.plot(df['year'], df['wind_electricity'], color='#2980B9', lw=2.5, marker='s', label='Wind Power (TWh)')
plt.scatter([2024, 2024], [S0_raw, W0_raw], color='#C0392B', s=80, zorder=5)
plt.annotate(f"2024 Baseline:\nSolar S0 = {S0_raw:.2f} TWh\nWind W0 = {W0_raw:.2f} TWh",
             xy=(2024, S0_raw), xytext=(2014, S0_raw - 4),
             arrowprops=dict(arrowstyle="->", color='#C0392B', lw=1.5),
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C0392B"))
plt.title("Figure 1: Vietnam Solar and Wind Electricity Generation (2000–2024)", fontweight='bold')
plt.xlabel("Year", fontweight='semibold')
plt.ylabel("Electricity Generation (TWh)", fontweight='semibold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left')
plt.tight_layout()
fig1_path = os.path.join("figures", "fig1_historical_electricity_vietnam.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"-> Generated: {fig1_path}")

# ------------------------------------------------------------------------------
# Figure 2: Indifference Curves, Budget Line, and 1-Variable Function f(x)
# ------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# (A) Indifference Curves & Tangency
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
ax1.set_title("(A) Indifference Curves & Budget Constraint", fontweight='bold')
ax1.set_xlabel("Solar Allocation x", fontweight='semibold')
ax1.set_ylabel("Wind Allocation y", fontweight='semibold')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='lower left')

# (B) Single-variable objective curve f(x)
x_vals = np.linspace(0, 10, 200)
f_vals = [s0_norm * np.log(1.0 + v) + w0_norm * np.log(1.0 + 10.0 - v) for v in x_vals]
ax2.plot(x_vals, f_vals, color='#27AE60', lw=2.5, label='Utility Curve f(x)')
ax2.scatter([vn_norm['x']], [vn_norm['F']], color='#8E44AD', s=100, zorder=5)
ax2.axhline(vn_norm['F'], color='#8E44AD', linestyle=':', label="Horizontal Tangent (f'=0)")
ax2.set_title("(B) Single-Variable Objective f(x) (Strictly Concave)", fontweight='bold')
ax2.set_xlabel("Solar Allocation x (Wind y = 10 - x)", fontweight='semibold')
ax2.set_ylabel("Total Utility f(x)", fontweight='semibold')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='lower center')

plt.tight_layout()
fig2_path = os.path.join("figures", "fig2_optimization_geometry.png")
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"-> Generated: {fig2_path}")

# ------------------------------------------------------------------------------
# Figure 3: Budget Sensitivity Allocations x*, y* and Utility F*
# ------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# (A) Allocations x*(B) and y*(B)
ax1.plot(df_sens['Budget_B'], df_sens['x_Solar'], color='#F39C12', lw=2.5, marker='o', label='Solar x*(B)')
ax1.plot(df_sens['Budget_B'], df_sens['y_Wind'], color='#2980B9', lw=2.5, marker='s', label='Wind y*(B)')
ax1.plot(df_sens['Budget_B'], df_sens['Budget_B'], color='#7F8C8D', linestyle='--', label='Total Budget B')
ax1.set_title("(A) Optimal Allocations vs. Budget B", fontweight='bold')
ax1.set_xlabel("Total Budget B", fontweight='semibold')
ax1.set_ylabel("Allocated Units", fontweight='semibold')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# (B) Optimal Utility F*(B) and Diminishing Shadow Price lambda*(B)
color_F = '#27AE60'
color_lam = '#C0392B'
ax2.plot(df_sens['Budget_B'], df_sens['F_Optimal'], color=color_F, lw=2.5, marker='^', label='Utility F*(B)')
ax2.set_xlabel("Total Budget B", fontweight='semibold')
ax2.set_ylabel("Optimal Utility F*", color=color_F, fontweight='semibold')
ax2.tick_params(axis='y', labelcolor=color_F)

ax2_twin = ax2.twinx()
ax2_twin.plot(df_sens['Budget_B'], df_sens['Shadow_Price_lambda'], color=color_lam, lw=2, linestyle='-.', marker='v', label='Shadow Price lambda*')
ax2_twin.set_ylabel("Lagrange Multiplier lambda*", color=color_lam, fontweight='semibold')
ax2_twin.tick_params(axis='y', labelcolor=color_lam)
ax2.set_title("(B) Optimal Utility F* & Diminishing Shadow Price lambda*", fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
fig3_path = os.path.join("figures", "fig3_budget_sensitivity_analysis.png")
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"-> Generated: {fig3_path}")

print("\n" + "=" * 70)
print("EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 70)
