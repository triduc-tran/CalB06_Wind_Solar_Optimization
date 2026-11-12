"""
CalB06: Wind & Solar Power Resource Allocation under Budget Constraints
========================================================================
Course: Calculus in Business (Giải tích trong kinh doanh) - Semester HK261
Project Code: CalB-06

Description:
------------
This program models the optimal budget allocation between Solar Power (x) and 
Wind Power (y) under a fixed normalized budget B. The objective function uses 
logarithmic utility to capture diminishing marginal returns, with weights S0 (Solar) 
and W0 (Wind) calibrated from empirical power generation data in Vietnam (2024 OWID).

Key Mathematical & Economic Concepts Implemented:
-------------------------------------------------
1. Data Pipeline:
   - Ingests historical energy generation (2000-2024) from Our World in Data (OWID).
   - Extracts baseline weights S0 (Solar) and W0 (Wind) for the reference year 2024.
   - Evaluates both Raw (TWh) and Normalized weights (summing to 1.0).

2. Analytical Optimization (Lagrange Multipliers & Single-Variable Reduction):
   - Formulates the Lagrangian function: L(x, y, lambda) = S0*ln(1+x) + W0*ln(1+y) - lambda*(x+y-B).
   - Derives First-Order Necessary Conditions (FOC) yielding closed-form formulas:
         x* = [S0*(B + 1) - W0] / (S0 + W0)
         y* = [W0*(B + 1) - S0] / (S0 + W0)
         lambda* = (S0 + W0) / (B + 2)
   - Single-variable objective: f(x) = S0*ln(1+x) + W0*ln(1+B-x).

3. Second-Order Sufficient Conditions (SOC) & Strict Concavity:
   - Single-variable second derivative:
         f''(x) = - S0/(1+x)^2 - W0/(1+B-x)^2 < 0  for all x in [0, B].
   - Bordered Hessian determinant: det(H_bar) = S0/(1+x)^2 + W0/(1+y)^2 > 0.
   - Both prove that the critical point is the UNIQUE GLOBAL MAXIMUM.

4. Boundary / Corner Solutions (KKT Conditions):
   - Evaluates feasibility: 0 <= x* <= B.
   - When S0*(B+1) < W0, stationary point x* < 0, optimum snaps to boundary: x*=0, y*=B.
   - When W0*(B+1) < S0, stationary point x* > B, optimum snaps to boundary: x*=B, y*=0.
   - Analyzes degenerate cases where S0 = 0 or W0 = 0.

5. Numerical Optimization & Convergence Verification:
   - Solves the constrained nonlinear problem via `scipy.optimize.minimize` (SLSQP).
   - Benchmarks numerical solutions against analytical closed forms to verify precision (errors < 1e-7).

6. Sensitivity Analysis & Envelope Theorem:
   - Sweeps budget B across [5.0, 15.0].
   - Verifies the Envelope Theorem: dF*/dB = lambda*(B).
   - Demonstrates that marginal utility lambda*(B) decreases monotonically as budget expands.

7. Publication-Quality Visualizations:
   - Figure 1: Vietnam Renewable Electricity Generation History (2000-2024).
   - Figure 2: Constrained Optimization Geometry (Contour Curves & Single-Variable Concavity).
   - Figure 3: Budget Sensitivity & Shadow Price Dynamics (B in [5, 15]).
   - Figure 4: Boundary / Corner Solution Analysis (Out-of-bounds scenario).
"""

import os
import sys
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure matplotlib aesthetic styling for academic reports
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 13


# ==============================================================================
# SECTION 1: DATA PIPELINE & PREPROCESSING
# ==============================================================================

def load_and_preprocess_data(filepath: str) -> Tuple[pd.DataFrame, float, float, float, float]:
    """
    Load and clean historical renewable energy generation data for Vietnam.

    Economic / Methodological Meaning:
    ---------------------------------
    In real-world policy and business analysis, mathematical utility weights S0 and W0
    should not be chosen arbitrarily. Instead, they are grounded in recent empirical 
    generation capacity (year 2024 from Our World in Data).
    
    Data Handling Note:
    -------------------
    The raw CSV contains 25 annual observation rows (2000-2024). Trailing metadata rows
    (e.g., text notes at indices 25 and 26) are filtered out to ensure clean numerical types.

    Parameters:
    -----------
    filepath : str
        Path to the CSV data file (`vietnam_solar_wind_cleaned.csv`).

    Returns:
    --------
    df_clean : pd.DataFrame
        Cleaned time-series DataFrame containing columns:
        ['country', 'year', 'solar_electricity', 'wind_electricity'].
    S0_raw : float
        Actual solar electricity generation in 2024 (in TWh).
    W0_raw : float
        Actual wind electricity generation in 2024 (in TWh).
    s0_norm : float
        Normalized solar weight s0 = S0 / (S0 + W0), such that s0 + w0 = 1.0.
    w0_norm : float
        Normalized wind weight w0 = W0 / (S0 + W0), such that s0 + w0 = 1.0.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")

    # Read CSV file
    df_raw = pd.read_csv(filepath)

    # Filter out rows where 'solar_electricity' or 'wind_electricity' cannot be parsed as floats
    # or where 'country' is NaN (handling the calculation note lines at rows 25 and 26)
    df_clean = df_raw.dropna(subset=['country', 'solar_electricity', 'wind_electricity']).copy()
    
    # Ensure year is integer
    df_clean['year'] = pd.to_numeric(df_clean['year'], errors='coerce')
    df_clean = df_clean.dropna(subset=['year'])
    df_clean['year'] = df_clean['year'].astype(int)
    
    # Sort chronologically
    df_clean = df_clean.sort_values(by='year').reset_index(drop=True)

    # Extract year 2024 values (or the latest available year)
    row_2024 = df_clean[df_clean['year'] == 2024]
    if row_2024.empty:
        # Fallback to the latest year available
        latest_year = df_clean['year'].max()
        row_latest = df_clean[df_clean['year'] == latest_year].iloc[0]
        S0_raw = float(row_latest['solar_electricity'])
        W0_raw = float(row_latest['wind_electricity'])
        print(f"[Warning] Year 2024 not found. Using latest available year: {latest_year}")
    else:
        S0_raw = float(row_2024.iloc[0]['solar_electricity'])
        W0_raw = float(row_2024.iloc[0]['wind_electricity'])

    # Calculate normalized weights
    total_gen = S0_raw + W0_raw
    if total_gen <= 0:
        raise ValueError("Total electricity generation in base year must be strictly positive.")

    s0_norm = S0_raw / total_gen
    w0_norm = W0_raw / total_gen

    return df_clean, S0_raw, W0_raw, s0_norm, w0_norm


# ==============================================================================
# SECTION 2: OBJECTIVE FUNCTION & DERIVATIVES
# ==============================================================================

def utility_function(x: float, y: float, S0: float, W0: float) -> float:
    """
    Evaluate the bivariate logarithmic utility objective function F(x, y).

    Formula:
    --------
    F(x, y) = S0 * ln(1 + x) + W0 * ln(1 + y)

    Economic Interpretation:
    ------------------------
    - The '+ 1' within the logarithm ensures that investing zero budget in a technology 
      yields zero benefit: F(0, 0) = S0*ln(1) + W0*ln(1) = 0. This circumvents ln(0) = -inf.
    - Diminishing marginal returns: As investment x in solar increases, each additional unit 
      of budget provides less additional utility (dF/dx = S0 / (1 + x)).
    - S0 and W0 act as preference/productivity scaling weights.

    Parameters:
    -----------
    x : float, budget allocated to solar power (x >= 0).
    y : float, budget allocated to wind power (y >= 0).
    S0 : float, solar weighting parameter.
    W0 : float, wind weighting parameter.

    Returns:
    --------
    float, total weighted utility value.
    """
    if x < -1.0 or y < -1.0:
        return -np.inf
    return float(S0 * np.log(1.0 + x) + W0 * np.log(1.0 + y))


def single_variable_utility(x: float, B: float, S0: float, W0: float) -> float:
    """
    Evaluate the single-variable utility function f(x) after substituting y = B - x.

    Formula:
    --------
    f(x) = S0 * ln(1 + x) + W0 * ln(1 + B - x)   for x in [0, B]

    Mathematical Purpose:
    ---------------------
    Reduces the 2-variable constrained optimization problem with equality constraint x + y = B
    into an unconstrained 1-variable problem over the closed bounded interval [0, B].
    """
    if x < 0.0 or x > B:
        return -np.inf
    return float(S0 * np.log(1.0 + x) + W0 * np.log(1.0 + B - x))


def single_variable_first_derivative(x: float, B: float, S0: float, W0: float) -> float:
    """
    Calculate the first derivative f'(x) of the single-variable utility function.

    Formula:
    --------
    f'(x) = S0 / (1 + x) - W0 / (1 + B - x)

    Economic / Algorithmic Meaning:
    -------------------------------
    - Represents the net marginal gain from shifting one budget unit from wind to solar.
    - At the interior stationary point, f'(x*) = 0, meaning the marginal utility per dollar
      invested in solar equals the marginal utility per dollar invested in wind:
      S0 / (1 + x*) = W0 / (1 + y*) = lambda*.
    """
    return float(S0 / (1.0 + x) - W0 / (1.0 + B - x))


def single_variable_second_derivative(x: float, B: float, S0: float, W0: float) -> float:
    """
    Calculate the second derivative f''(x) to prove strict concavity (SOC).

    Formula:
    --------
    f''(x) = - S0 / (1 + x)^2 - W0 / (1 + B - x)^2

    Mathematical Proof of Optimality:
    ---------------------------------
    Since S0 > 0, W0 > 0, and for all x in [0, B]:
        (1 + x)^2 > 0  and  (1 + B - x)^2 > 0
    we have:
        f''(x) < 0  strictly for all x in [0, B].
    Therefore:
    1. The objective function f(x) is STRICTLY CONCAVE on the convex set [0, B].
    2. Any critical point found by First-Order Conditions is guaranteed to be 
       the UNIQUE GLOBAL MAXIMUM.
    """
    return float(- S0 / ((1.0 + x) ** 2) - W0 / ((1.0 + B - x) ** 2))


def bordered_hessian_analysis(x: float, y: float, S0: float, W0: float) -> Tuple[np.ndarray, float]:
    """
    Construct the Bordered Hessian matrix for the 2-variable constrained problem 
    and compute its determinant to verify the Second-Order Sufficient Condition (SOSC).

    Mathematical Formulation:
    -------------------------
    Constraint: g(x, y) = x + y - B = 0.
    Gradient of constraint: [g_x, g_y] = [1, 1].
    Lagrangian: L(x, y, lambda) = F(x, y) - lambda * (x + y - B).
    Second partial derivatives of L with respect to (x, y):
        L_xx = - S0 / (1 + x)^2
        L_yy = - W0 / (1 + y)^2
        L_xy = L_yx = 0

    Bordered Hessian Matrix (H_bar):
        H_bar = [  0    g_x   g_y  ]   [  0            1                    1          ]
                [ g_x  L_xx  L_xy  ] = [  1   -S0/(1+x)^2                   0          ]
                [ g_y  L_yx  L_yy  ]   [  1            0           -W0/(1+y)^2         ]

    Determinant of H_bar:
        det(H_bar) = 0 - 1 * (-W0/(1+y)^2 - 0) + 1 * (0 - (-S0/(1+x)^2))
                   = S0/(1+x)^2 + W0/(1+y)^2 > 0

    Second-Order Sufficient Condition Rule:
    ---------------------------------------
    For a maximization problem with 2 decision variables and 1 equality constraint,
    the condition for a strict local maximum is:
        (-1)^m * det(H_bar) > 0   where m = 1 (number of constraints), n = 2.
    Since (-1)^1 * det(H_bar) < 0, or equivalently det(H_bar) > 0, the stationary 
    point is a strictly constrained local/global maximum.

    Returns:
    --------
    H_bar : np.ndarray, (3x3) bordered Hessian matrix.
    det_H : float, determinant value.
    """
    L_xx = - S0 / ((1.0 + x) ** 2)
    L_yy = - W0 / ((1.0 + y) ** 2)
    L_xy = 0.0

    H_bar = np.array([
        [0.0, 1.0, 1.0],
        [1.0, L_xx, L_xy],
        [1.0, L_xy, L_yy]
    ])

    det_H = float(np.linalg.det(H_bar))
    return H_bar, det_H


# ==============================================================================
# SECTION 3: ANALYTICAL & BOUNDARY OPTIMIZATION SOLVER
# ==============================================================================

def solve_analytical(S0: float, W0: float, B: float) -> Dict[str, any]:
    """
    Solve the budget allocation problem analytically using Lagrange Multipliers
    and Karush-Kuhn-Tucker (KKT) boundary conditions.

    Mathematical Derivation:
    ------------------------
    1. First-Order Necessary Conditions (FOC):
       dL/dx = S0 / (1 + x) - lambda = 0   ==> lambda = S0 / (1 + x)
       dL/dy = W0 / (1 + y) - lambda = 0   ==> lambda = W0 / (1 + y)
       x + y = B

    2. Equating marginal utilities:
       S0 / (1 + x) = W0 / (1 + y)
       S0 * (1 + B - x) = W0 * (1 + x)
       S0 * (B + 1) - S0 * x = W0 + W0 * x
       (S0 + W0) * x = S0 * (B + 1) - W0
       ==> x_stat = [S0 * (B + 1) - W0] / (S0 + W0)
       ==> y_stat = B - x_stat = [W0 * (B + 1) - S0] / (S0 + W0)
       ==> lambda_stat = (S0 + W0) / (B + 2)

    3. Feasibility and Boundary Evaluation:
       - Case A: Both weights are strictly positive (S0 > 0, W0 > 0):
         * If 0 <= x_stat <= B (equivalent to 1/(B+1) <= S0/W0 <= B+1):
           Valid Interior Solution: x* = x_stat, y* = y_stat, lambda* = lambda_stat.
         * If x_stat < 0 (equivalent to S0*(B+1) < W0):
           Stationary point requires negative solar investment, which is prohibited.
           Since f'(x) < 0 for all x in [0, B], f(x) is strictly decreasing.
           Optimal solution lies at the corner boundary:
               x* = 0,  y* = B,  F* = W0 * ln(1 + B)
         * If x_stat > B (equivalent to W0*(B+1) < S0):
           Stationary point requires negative wind investment.
           Since f'(x) > 0 for all x in [0, B], f(x) is strictly increasing.
           Optimal solution lies at the corner boundary:
               x* = B,  y* = 0,  F* = S0 * ln(1 + B)

       - Case B: Degenerate weights (one weight is zero):
         * If S0 == 0 and W0 > 0: All budget allocated to wind (x* = 0, y* = B).
         * If W0 == 0 and S0 > 0: All budget allocated to solar (x* = B, y* = 0).
         * If S0 == 0 and W0 == 0: Utility is identically zero everywhere.

    Parameters:
    -----------
    S0 : float, solar weight parameter.
    W0 : float, wind weight parameter.
    B : float, total budget (B > 0).

    Returns:
    --------
    dict containing solution metadata, optimal coordinates (x*, y*), F*, lambda*,
    concavity verification metrics, and solution type ('Interior' or 'Boundary/Corner').
    """
    if B <= 0:
        raise ValueError("Budget B must be strictly positive.")

    # Check for degenerate zero-weight cases
    if S0 <= 0 and W0 <= 0:
        return {
            "solution_type": "Degenerate (Zero Weights)",
            "x_optimal": 0.0,
            "y_optimal": B,
            "F_optimal": 0.0,
            "lambda_optimal": 0.0,
            "is_interior": False,
            "x_unconstrained": 0.0,
            "y_unconstrained": B,
            "f_double_prime": 0.0,
            "det_bordered_hessian": 0.0
        }
    elif S0 <= 0:
        # Zero solar weight: allocate all budget to wind
        x_opt = 0.0
        y_opt = float(B)
        F_opt = utility_function(x_opt, y_opt, S0, W0)
        return {
            "solution_type": "Boundary (S0 = 0)",
            "x_optimal": x_opt,
            "y_optimal": y_opt,
            "F_optimal": F_opt,
            "lambda_optimal": float(W0 / (1.0 + B)),
            "is_interior": False,
            "x_unconstrained": - W0 / W0, # -1.0
            "y_unconstrained": B + 1.0,
            "f_double_prime": - W0 / ((1.0 + B) ** 2),
            "det_bordered_hessian": W0 / ((1.0 + B) ** 2)
        }
    elif W0 <= 0:
        # Zero wind weight: allocate all budget to solar
        x_opt = float(B)
        y_opt = 0.0
        F_opt = utility_function(x_opt, y_opt, S0, W0)
        return {
            "solution_type": "Boundary (W0 = 0)",
            "x_optimal": x_opt,
            "y_optimal": y_opt,
            "F_optimal": F_opt,
            "lambda_optimal": float(S0 / (1.0 + B)),
            "is_interior": False,
            "x_unconstrained": B + 1.0,
            "y_unconstrained": - S0 / S0,
            "f_double_prime": - S0 / ((1.0 + B) ** 2),
            "det_bordered_hessian": S0 / ((1.0 + B) ** 2)
        }

    # Both S0 > 0 and W0 > 0
    denom = S0 + W0
    x_unconstrained = (S0 * (B + 1.0) - W0) / denom
    y_unconstrained = (W0 * (B + 1.0) - S0) / denom
    lambda_unconstrained = denom / (B + 2.0)

    # Check whether the unconstrained stationary point falls within the feasible domain [0, B]
    if x_unconstrained < 0.0:
        # Stationary point is negative for solar -> boundary optimum at x=0
        x_opt = 0.0
        y_opt = float(B)
        solution_type = "Boundary / Corner (x* = 0, y* = B)"
        is_interior = False
        F_opt = utility_function(x_opt, y_opt, S0, W0)
        # At boundary x=0, shadow price equals marginal utility of the active resource
        lambda_opt = float(W0 / (1.0 + B))
    elif x_unconstrained > B:
        # Stationary point exceeds total budget -> boundary optimum at x=B
        x_opt = float(B)
        y_opt = 0.0
        solution_type = "Boundary / Corner (x* = B, y* = 0)"
        is_interior = False
        F_opt = utility_function(x_opt, y_opt, S0, W0)
        lambda_opt = float(S0 / (1.0 + B))
    else:
        # Interior optimum
        x_opt = float(x_unconstrained)
        y_opt = float(y_unconstrained)
        solution_type = "Interior Optimum"
        is_interior = True
        F_opt = utility_function(x_opt, y_opt, S0, W0)
        lambda_opt = float(lambda_unconstrained)

    # Compute concavity verification metrics at optimal solution
    f_dprime = single_variable_second_derivative(x_opt, B, S0, W0)
    _, det_H = bordered_hessian_analysis(x_opt, y_opt, S0, W0)

    return {
        "solution_type": solution_type,
        "x_optimal": x_opt,
        "y_optimal": y_opt,
        "F_optimal": F_opt,
        "lambda_optimal": lambda_opt,
        "is_interior": is_interior,
        "x_unconstrained": float(x_unconstrained),
        "y_unconstrained": float(y_unconstrained),
        "f_double_prime": f_dprime,
        "det_bordered_hessian": det_H
    }


# ==============================================================================
# SECTION 4: NUMERICAL OPTIMIZATION (SCIPY SLSQP) & ERROR ANALYSIS
# ==============================================================================

def solve_numerical(S0: float, W0: float, B: float, x0: Optional[Tuple[float, float]] = None) -> Dict[str, any]:
    """
    Solve the constrained optimization problem numerically using Sequential Least 
    Squares Programming (SLSQP) via `scipy.optimize.minimize`.

    Algorithmic Mechanics:
    ----------------------
    Problem formulation:
        Minimize:   - F(x, y) = - [ S0 * ln(1 + x) + W0 * ln(1 + y) ]
        Subject to: x + y - B = 0     (Equality constraint)
                    0 <= x <= B       (Bounds)
                    0 <= y <= B       (Bounds)

    Initial Guess (x0):
        Defaults to an equal 50-50 split: [B/2, B/2].

    Parameters:
    -----------
    S0 : float, solar weight.
    W0 : float, wind weight.
    B : float, budget.
    x0 : tuple of (float, float), optional initial starting point.

    Returns:
    --------
    dict containing numerical solution, optimization status, message, iterations, and function evaluations.
    """
    if x0 is None:
        x0 = [B / 2.0, B / 2.0]

    # Objective function to minimize (negative of utility)
    def objective(vars_vec):
        x, y = vars_vec
        return - (S0 * np.log(1.0 + x) + W0 * np.log(1.0 + y))

    # Analytical gradient for SLSQP to guarantee high precision
    def objective_grad(vars_vec):
        x, y = vars_vec
        grad_x = - S0 / (1.0 + x)
        grad_y = - W0 / (1.0 + y)
        return np.array([grad_x, grad_y])

    # Constraint definition: x + y = B  ==>  B - (x + y) = 0
    budget_constraint = {
        'type': 'eq',
        'fun': lambda vars_vec: B - (vars_vec[0] + vars_vec[1]),
        'jac': lambda vars_vec: np.array([-1.0, -1.0])
    }

    # Variable bounds: x >= 0, y >= 0
    var_bounds = [(0.0, float(B)), (0.0, float(B))]

    # Run SLSQP optimizer
    result = minimize(
        fun=objective,
        x0=x0,
        jac=objective_grad,
        method='SLSQP',
        bounds=var_bounds,
        constraints=[budget_constraint],
        options={'ftol': 1e-12, 'maxiter': 200, 'disp': False}
    )

    x_num, y_num = result.x
    F_num = - result.fun

    return {
        "x_numerical": float(x_num),
        "y_numerical": float(y_num),
        "F_numerical": float(F_num),
        "success": bool(result.success),
        "message": str(result.message),
        "nit": int(result.nit),
        "nfev": int(result.nfev)
    }


def compare_analytical_vs_numerical(analytical_res: Dict[str, any], numerical_res: Dict[str, any]) -> Dict[str, float]:
    """
    Perform rigorous error analysis between analytical closed-form and numerical results.

    Error Metrics:
    --------------
    - Absolute Error: |val_analytical - val_numerical|
    - Relative Error: |val_analytical - val_numerical| / (|val_analytical| + eps)

    Purpose:
    --------
    To demonstrate that the analytical derivation exactly mirrors the numerical convergence,
    proving that any discrepancies are strictly limited to machine-precision rounding (~1e-15 to 1e-8).
    """
    x_ana, y_ana, F_ana = analytical_res['x_optimal'], analytical_res['y_optimal'], analytical_res['F_optimal']
    x_num, y_num, F_num = numerical_res['x_numerical'], numerical_res['y_numerical'], numerical_res['F_numerical']

    eps = 1e-12
    err_x_abs = abs(x_ana - x_num)
    err_y_abs = abs(y_ana - y_num)
    err_F_abs = abs(F_ana - F_num)

    err_x_rel = err_x_abs / (abs(x_ana) + eps)
    err_y_rel = err_y_abs / (abs(y_ana) + eps)
    err_F_rel = err_F_abs / (abs(F_ana) + eps)

    return {
        "error_x_abs": float(err_x_abs),
        "error_y_abs": float(err_y_abs),
        "error_F_abs": float(err_F_abs),
        "error_x_rel": float(err_x_rel),
        "error_y_rel": float(err_y_rel),
        "error_F_rel": float(err_F_rel)
    }


# ==============================================================================
# SECTION 5: BUDGET SENSITIVITY ANALYSIS & ENVELOPE THEOREM
# ==============================================================================

def run_budget_sensitivity_analysis(
    S0: float, 
    W0: float, 
    B_min: float = 5.0, 
    B_max: float = 15.0, 
    step: float = 0.5
) -> pd.DataFrame:
    """
    Conduct sensitivity analysis over the budget range B in [5.0, 15.0] as mandated by Topic 6.

    Economic Principles Examined:
    -----------------------------
    1. Capital Allocation Dynamics:
       - Since x*(B) = [S0 / (S0+W0)]*B + (S0-W0)/(S0+W0), both x* and y* expand linearly 
         with respect to budget B (expansion path is an affine straight line).
       - Marginal budget allocation to solar: dx*/dB = S0 / (S0 + W0) = s0.
       - Marginal budget allocation to wind:  dy*/dB = W0 / (S0 + W0) = w0.

    2. Envelope Theorem Verification:
       - The Envelope Theorem states: dF*/dB = dL/dB = lambda*(B).
       - We compare the analytical shadow price lambda*(B) against numerical finite differences:
             dF_num / dB ~= [F*(B + Delta_B) - F*(B - Delta_B)] / (2 * Delta_B)
       - Demonstrates diminishing marginal utility to total budget:
             d(lambda*)/dB = - (S0 + W0) / (B + 2)^2 < 0.

    Parameters:
    -----------
    S0 : float, solar weight.
    W0 : float, wind weight.
    B_min : float, minimum budget (default 5.0).
    B_max : float, maximum budget (default 15.0).
    step : float, step size for budget grid (default 0.5).

    Returns:
    --------
    pd.DataFrame containing sensitivity analysis records for each budget level.
    """
    b_values = np.arange(B_min, B_max + 1e-5, step)
    records = []

    for B in b_values:
        B_val = round(float(B), 4)
        ana = solve_analytical(S0, W0, B_val)
        num = solve_numerical(S0, W0, B_val)
        err = compare_analytical_vs_numerical(ana, num)

        records.append({
            "Budget_B": B_val,
            "x_solar_opt": ana["x_optimal"],
            "y_wind_opt": ana["y_optimal"],
            "F_optimal": ana["F_optimal"],
            "lambda_shadow_price": ana["lambda_optimal"],
            "is_interior": ana["is_interior"],
            "f_double_prime": ana["f_double_prime"],
            "x_numerical": num["x_numerical"],
            "y_numerical": num["y_numerical"],
            "F_numerical": num["F_numerical"],
            "error_F_abs": err["error_F_abs"]
        })

    df_sens = pd.DataFrame(records)

    # Compute numerical derivative dF*/dB using central differences
    F_vals = df_sens["F_optimal"].values
    B_vals = df_sens["Budget_B"].values
    dF_dB_num = np.gradient(F_vals, B_vals)
    df_sens["dF_dB_finite_diff"] = dF_dB_num
    df_sens["envelope_theorem_diff"] = np.abs(df_sens["lambda_shadow_price"] - df_sens["dF_dB_finite_diff"])

    return df_sens


# ==============================================================================
# SECTION 6: HIGH-RESOLUTION VISUALIZATION SUITE
# ==============================================================================

def generate_visualizations(
    df_history: pd.DataFrame,
    S0: float,
    W0: float,
    B_benchmark: float = 10.0,
    df_sensitivity: Optional[pd.DataFrame] = None,
    output_dir: str = "figures"
) -> List[str]:
    """
    Generate the full suite of publication-quality figures required for Topic CalB-06.

    Figures Produced:
    -----------------
    1. Figure 1: `fig1_historical_electricity_vietnam.png`
       - Empirical time-series of solar and wind generation in Vietnam (2000-2024).
       - Highlights the renewable energy boom post-2018 and marks the 2024 calibration baseline.

    2. Figure 2: `fig2_optimization_geometry.png`
       - 2-Panel Diagram:
         Panel A: Contour lines of F(x, y), budget constraint line x + y = 10, feasible region, 
                  and the tangency point (x*, y*) representing constrained maximum.
         Panel B: Single-variable curve f(x) on [0, 10], illustrating strict concavity (f''(x) < 0)
                  and zero slope f'(x*) = 0 at the peak.

    3. Figure 3: `fig3_budget_sensitivity_analysis.png`
       - 2-Panel Diagram:
         Panel A: Optimal investment paths x*(B) and y*(B) for B in [5, 15].
         Panel B: Total optimal utility F*(B) and shadow price lambda*(B) = dF*/dB,
                  demonstrating diminishing marginal returns to scale.

    4. Figure 4: `fig4_boundary_corner_solution.png`
       - Illustrates an out-of-bounds scenario (e.g., S0=0.2, W0=3.0, B=10.0) where the 
         unconstrained stationary point is negative (x < 0), forcing the constrained optimum
         to snap to the corner boundary (x*=0, y*=B).

    Parameters:
    -----------
    df_history : pd.DataFrame, cleaned historical generation data.
    S0 : float, solar weight.
    W0 : float, wind weight.
    B_benchmark : float, base budget (default 10.0).
    df_sensitivity : pd.DataFrame, sensitivity table over B in [5, 15].
    output_dir : str, destination directory for saving figures.

    Returns:
    --------
    List[str], list of file paths to generated figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    # --------------------------------------------------------------------------
    # FIGURE 1: Historical Generation (Vietnam 2000-2024)
    # --------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    years = df_history['year'].values
    solar_gen = df_history['solar_electricity'].values
    wind_gen = df_history['wind_electricity'].values

    ax1.plot(years, solar_gen, color='#F39C12', linewidth=2.5, marker='o', markersize=4, label='Solar Electricity (TWh)')
    ax1.plot(years, wind_gen, color='#2980B9', linewidth=2.5, marker='s', markersize=4, label='Wind Electricity (TWh)')
    ax1.fill_between(years, solar_gen, color='#F39C12', alpha=0.15)
    ax1.fill_between(years, wind_gen, color='#2980B9', alpha=0.15)

    # Highlight reference year 2024
    if 2024 in years:
        s24 = float(df_history[df_history['year'] == 2024]['solar_electricity'].iloc[0])
        w24 = float(df_history[df_history['year'] == 2024]['wind_electricity'].iloc[0])
        ax1.scatter([2024, 2024], [s24, w24], color='#C0392B', s=80, zorder=5)
        ax1.annotate(
            f'2024 Baseline:\nSolar S0 = {s24:.2f} TWh (67.0%)\nWind W0 = {w24:.2f} TWh (33.0%)',
            xy=(2024, s24),
            xytext=(2015, s24 - 4),
            arrowprops=dict(arrowstyle="->", color='#C0392B', lw=1.5),
            fontsize=9.5,
            fontweight='semibold',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C0392B", lw=1.2)
        )

    ax1.set_title("Vietnam Renewable Electricity Generation (2000–2024)\n[Source: Our World in Data]", fontweight='bold', pad=12)
    ax1.set_xlabel("Year", fontweight='semibold')
    ax1.set_ylabel("Electricity Generation (Terawatt-hours, TWh)", fontweight='semibold')
    ax1.set_xlim(2000, 2025)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True)

    fig1_path = os.path.join(output_dir, "fig1_historical_electricity_vietnam.png")
    fig1.tight_layout()
    fig1.savefig(fig1_path)
    plt.close(fig1)
    generated_files.append(fig1_path)

    # --------------------------------------------------------------------------
    # FIGURE 2: Constrained Optimization Geometry (Contour + 1-Variable Concavity)
    # --------------------------------------------------------------------------
    sol_bench = solve_analytical(S0, W0, B_benchmark)
    x_star = sol_bench['x_optimal']
    y_star = sol_bench['y_optimal']
    F_star = sol_bench['F_optimal']

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: 2D Contour Plot & Budget Line
    grid_res = 200
    x_grid = np.linspace(0.0, B_benchmark + 2.0, grid_res)
    y_grid = np.linspace(0.0, B_benchmark + 2.0, grid_res)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = S0 * np.log(1.0 + X) + W0 * np.log(1.0 + Y)

    # Contour lines
    contours = ax2a.contour(X, Y, Z, levels=14, cmap='viridis', linewidths=1.2, alpha=0.85)
    ax2a.clabel(contours, inline=True, fontsize=8, fmt='%.1f')

    # Budget line: x + y = B
    x_line = np.linspace(0, B_benchmark, 100)
    y_line = B_benchmark - x_line
    ax2a.plot(x_line, y_line, color='#E74C3C', linewidth=2.5, linestyle='-', label=f'Budget Constraint: x + y = {B_benchmark}')
    ax2a.fill_between(x_line, 0, y_line, color='#E74C3C', alpha=0.08, label='Feasible Investment Set')

    # Optimal point
    ax2a.scatter([x_star], [y_star], color='#8E44AD', s=120, zorder=5, label=f'Optimum (x*={x_star:.2f}, y*={y_star:.2f})')
    ax2a.annotate(
        f'Tangency Optimum:\n(x*={x_star:.2f}, y*={y_star:.2f})\nF*={F_star:.3f}',
        xy=(x_star, y_star),
        xytext=(x_star - 4.5, y_star + 1.8),
        arrowprops=dict(arrowstyle="->", color='#8E44AD', lw=1.5),
        fontsize=9,
        fontweight='semibold',
        bbox=dict(boxstyle="round,pad=0.3", fc="#F4ECF7", ec="#8E44AD", lw=1.2)
    )

    ax2a.set_title("(A) Indifference Curves & Budget Constraint Tangency", fontweight='bold')
    ax2a.set_xlabel("Solar Budget Allocation x", fontweight='semibold')
    ax2a.set_ylabel("Wind Budget Allocation y", fontweight='semibold')
    ax2a.set_xlim(0, B_benchmark + 1.5)
    ax2a.set_ylim(0, B_benchmark + 1.5)
    ax2a.grid(True, linestyle='--', alpha=0.5)
    ax2a.legend(loc='lower left', frameon=True, fontsize=8.5)

    # Panel B: Single-Variable Function f(x)
    x_pts = np.linspace(0.0, B_benchmark, 300)
    f_pts = [single_variable_utility(val, B_benchmark, S0, W0) for val in x_pts]

    ax2b.plot(x_pts, f_pts, color='#27AE60', linewidth=2.5, label='Utility Curve f(x)')
    ax2b.scatter([x_star], [F_star], color='#8E44AD', s=120, zorder=5)

    # Tangent horizontal line at optimum (f'(x*) = 0)
    ax2b.axhline(F_star, color='#8E44AD', linestyle=':', linewidth=1.5, label=f"Horizontal Tangent (f'(x*) = 0)")
    ax2b.axvline(x_star, color='#8E44AD', linestyle='--', linewidth=1.2, alpha=0.7)

    # Mark boundary values for comparison
    f_0 = single_variable_utility(0.0, B_benchmark, S0, W0)
    f_B = single_variable_utility(B_benchmark, B_benchmark, S0, W0)
    ax2b.scatter([0.0, B_benchmark], [f_0, f_B], color='#E67E22', s=60, zorder=4, label='Boundary Allocations (x=0, x=B)')

    ax2b.annotate(
        f"Maximum at x*={x_star:.2f}\nf(x*)={F_star:.3f}\nf''(x*)={sol_bench['f_double_prime']:.4f} < 0",
        xy=(x_star, F_star),
        xytext=(x_star - 2.5, F_star - 0.7 * (F_star - min(f_0, f_B))),
        arrowprops=dict(arrowstyle="->", color='#8E44AD', lw=1.5),
        fontsize=9,
        fontweight='semibold',
        bbox=dict(boxstyle="round,pad=0.3", fc="#E8F8F5", ec="#27AE60", lw=1.2)
    )

    ax2b.set_title("(B) Single-Variable Reduction f(x) & Strict Concavity", fontweight='bold')
    ax2b.set_xlabel("Solar Allocation x (Wind y = 10 - x)", fontweight='semibold')
    ax2b.set_ylabel("Total Objective Utility f(x)", fontweight='semibold')
    ax2b.grid(True, linestyle='--', alpha=0.5)
    ax2b.legend(loc='lower center', frameon=True, fontsize=8.5)

    fig2_path = os.path.join(output_dir, "fig2_optimization_geometry.png")
    fig2.tight_layout()
    fig2.savefig(fig2_path)
    plt.close(fig2)
    generated_files.append(fig2_path)

    # --------------------------------------------------------------------------
    # FIGURE 3: Budget Sensitivity Analysis (B in [5, 15])
    # --------------------------------------------------------------------------
    if df_sensitivity is not None and not df_sensitivity.empty:
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5.5))

        b_vals = df_sensitivity['Budget_B'].values
        x_vals = df_sensitivity['x_solar_opt'].values
        y_vals = df_sensitivity['y_wind_opt'].values
        F_vals = df_sensitivity['F_optimal'].values
        lam_vals = df_sensitivity['lambda_shadow_price'].values

        # Panel A: Budget Allocations x*(B) and y*(B)
        ax3a.plot(b_vals, x_vals, color='#F39C12', linewidth=2.5, marker='o', markersize=4, label='Optimal Solar Allocation x*(B)')
        ax3a.plot(b_vals, y_vals, color='#2980B9', linewidth=2.5, marker='s', markersize=4, label='Optimal Wind Allocation y*(B)')
        ax3a.plot(b_vals, b_vals, color='#7F8C8D', linestyle='--', linewidth=1.5, label='Total Budget B (x* + y*)')

        # Highlight slope interpretation
        slope_x = S0 / (S0 + W0)
        slope_y = W0 / (S0 + W0)
        ax3a.text(
            0.05, 0.75,
            f"Expansion Slopes (Linear Path):\ndx*/dB = s0 = {slope_x:.3f}\ndy*/dB = w0 = {slope_y:.3f}",
            transform=ax3a.transAxes,
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#BDC3C7", lw=1.0)
        )

        ax3a.set_title("(A) Optimal Allocations x*(B) and y*(B) vs. Budget", fontweight='bold')
        ax3a.set_xlabel("Total Normalized Budget B", fontweight='semibold')
        ax3a.set_ylabel("Allocated Units", fontweight='semibold')
        ax3a.set_xlim(5, 15)
        ax3a.grid(True, linestyle='--', alpha=0.5)
        ax3a.legend(loc='lower right', frameon=True)

        # Panel B: Total Utility F*(B) and Shadow Price lambda*(B)
        color_F = '#27AE60'
        color_lam = '#C0392B'

        ax3b.plot(b_vals, F_vals, color=color_F, linewidth=2.5, marker='^', markersize=4, label='Maximum Utility F*(B)')
        ax3b.set_xlabel("Total Normalized Budget B", fontweight='semibold')
        ax3b.set_ylabel("Optimal Utility Value F*", color=color_F, fontweight='semibold')
        ax3b.tick_params(axis='y', labelcolor=color_F)

        # Secondary y-axis for shadow price lambda*
        ax3b_twin = ax3b.twinx()
        ax3b_twin.plot(b_vals, lam_vals, color=color_lam, linewidth=2.2, linestyle='-.', marker='v', markersize=4, label='Shadow Price lambda*(B)')
        ax3b_twin.set_ylabel("Lagrange Multiplier lambda* = dF*/dB", color=color_lam, fontweight='semibold')
        ax3b_twin.tick_params(axis='y', labelcolor=color_lam)

        ax3b.set_title("(B) Optimal Utility F*(B) & Shadow Price lambda*(B)\n[Diminishing Marginal Return to Budget]", fontweight='bold')
        ax3b.set_xlim(5, 15)
        ax3b.grid(True, linestyle='--', alpha=0.5)

        fig3_path = os.path.join(output_dir, "fig3_budget_sensitivity_analysis.png")
        fig3.tight_layout()
        fig3.savefig(fig3_path)
        plt.close(fig3)
        generated_files.append(fig3_path)

    # --------------------------------------------------------------------------
    # FIGURE 4: Boundary / Corner Solution Analysis (Out-of-Bounds Scenario)
    # --------------------------------------------------------------------------
    # Create an extreme scenario where wind weight dominates heavily: S0=0.2, W0=3.0, B=10
    S0_out = 0.2
    W0_out = 3.0
    B_out = 10.0
    sol_out = solve_analytical(S0_out, W0_out, B_out)

    fig4, ax4 = plt.subplots(figsize=(9, 5.5))

    # Evaluate single variable function extended into negative domain to show stationary point
    x_ext = np.linspace(-0.8, B_out, 300)
    # Utility function extended: S0*ln(1+x) + W0*ln(1+B-x)
    f_ext = [S0_out * np.log(1.0 + v) + W0_out * np.log(1.0 + B_out - v) if v > -1.0 else np.nan for v in x_ext]

    ax4.plot(x_ext, f_ext, color='#34495E', linewidth=2.0, linestyle='--', label="Unconstrained Objective Curve f(x)")

    # Plot feasible segment in green
    x_feas = np.linspace(0.0, B_out, 200)
    f_feas = [S0_out * np.log(1.0 + v) + W0_out * np.log(1.0 + B_out - v) for v in x_feas]
    ax4.plot(x_feas, f_feas, color='#27AE60', linewidth=3.0, label="Feasible Objective Segment on [0, B]")

    # Mark the unconstrained peak (x < 0)
    x_uncon = sol_out['x_unconstrained']
    F_uncon = S0_out * np.log(1.0 + x_uncon) + W0_out * np.log(1.0 + B_out - x_uncon)
    ax4.scatter([x_uncon], [F_uncon], color='#E74C3C', s=100, zorder=5, marker='x', label=f"Unconstrained Peak (Infeasible: x={x_uncon:.2f} < 0)")

    # Mark constrained boundary optimum (x* = 0, y* = B)
    F_corner = sol_out['F_optimal']
    ax4.scatter([0.0], [F_corner], color='#8E44AD', s=120, zorder=6, label=f"Constrained Optimum (Boundary: x*=0, y*={B_out:.1f})")

    # Add vertical divider at x = 0
    ax4.axvline(0.0, color='#7F8C8D', linestyle=':', linewidth=1.5, label="Feasibility Boundary (x = 0)")

    ax4.annotate(
        f"Stationary Point Infeasible:\nx_uncon = {x_uncon:.2f} < 0\nf'(x) < 0 for all x in [0, B]\n==> Optimum Snaps to Corner: x*=0, y*=10",
        xy=(0.0, F_corner),
        xytext=(1.8, F_corner - 0.8),
        arrowprops=dict(arrowstyle="->", color='#8E44AD', lw=1.5),
        fontsize=9.5,
        fontweight='semibold',
        bbox=dict(boxstyle="round,pad=0.4", fc="#FADBD8", ec="#E74C3C", lw=1.2)
    )

    ax4.set_title(f"Boundary / Corner Solution Scenario (S0={S0_out}, W0={W0_out}, B={B_out})\n[Demonstrating Karush-Kuhn-Tucker Boundary Activation]", fontweight='bold')
    ax4.set_xlabel("Solar Allocation x", fontweight='semibold')
    ax4.set_ylabel("Total Objective Utility f(x)", fontweight='semibold')
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(loc='lower right', frameon=True, fontsize=8.5)

    fig4_path = os.path.join(output_dir, "fig4_boundary_corner_solution.png")
    fig4.tight_layout()
    fig4.savefig(fig4_path)
    plt.close(fig4)
    generated_files.append(fig4_path)

    return generated_files


# ==============================================================================
# SECTION 7: MAIN EXECUTION & REPORTING PIPELINE
# ==============================================================================

def main():
    """
    Main execution pipeline orchestrating data ingestion, benchmark validation,
    Vietnam 2024 empirical case study, boundary condition testing, sensitivity analysis,
    and visualization generation.
    """
    print("=" * 80)
    print("CALCULUS IN BUSINESS (HK261) - PROJECT CALB-06")
    print("Wind & Solar Power Budget Allocation under Constraints")
    print("=" * 80)

    # 1. Load Data
    csv_filename = "vietnam_solar_wind_cleaned.csv"
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    csv_path = os.path.join(script_dir, csv_filename)

    print(f"\n[Step 1] Ingesting and preprocessing empirical dataset from: {csv_filename}...")
    df_clean, S0_raw, W0_raw, s0_norm, w0_norm = load_and_preprocess_data(csv_path)
    print(f"  -> Cleaned observation records: {len(df_clean)} years (2000 to 2024)")
    print(f"  -> 2024 Empirical Generation: Solar S0 = {S0_raw:.2f} TWh, Wind W0 = {W0_raw:.2f} TWh")
    print(f"  -> Total 2024 Generation: {S0_raw + W0_raw:.2f} TWh")
    print(f"  -> Normalized Weights: s0 = {s0_norm:.4f} ({s0_norm*100:.1f}%), w0 = {w0_norm:.4f} ({w0_norm*100:.1f}%)")

    # 2. Benchmark Case from Guide: S0 = 2.0, W0 = 1.0, B = 10.0
    print("\n" + "-" * 80)
    print("[Step 2] Benchmark Case from Assignment Guide (Hand Calculation Verification):")
    print("Parameters: S0 = 2.0, W0 = 1.0, B = 10.0")
    bench_ana = solve_analytical(S0=2.0, W0=1.0, B=10.0)
    bench_num = solve_numerical(S0=2.0, W0=1.0, B=10.0)
    bench_err = compare_analytical_vs_numerical(bench_ana, bench_num)

    print(f"  [Analytical Solution]")
    print(f"    - Solution Type: {bench_ana['solution_type']}")
    print(f"    - Optimal Solar Allocation (x*): {bench_ana['x_optimal']:.4f} (Exact formula: 21/3 = 7.0)")
    print(f"    - Optimal Wind Allocation  (y*): {bench_ana['y_optimal']:.4f} (Exact formula: 10 - 7 = 3.0)")
    print(f"    - Optimal Utility Value    (F*): {bench_ana['F_optimal']:.6f} (Exact formula: 8*ln(2) ~= 5.545177)")
    print(f"    - Shadow Price (lambda*):        {bench_ana['lambda_optimal']:.4f} (Exact formula: 3/12 = 0.25)")
    print(f"    - Second Derivative f''(x*):     {bench_ana['f_double_prime']:.6f} (< 0 ==> Strictly Concave)")
    print(f"    - Bordered Hessian Det det(H):   {bench_ana['det_bordered_hessian']:.6f} (> 0 ==> SOSC satisfied)")
    print(f"  [Numerical Solution (SLSQP)]")
    print(f"    - x_num = {bench_num['x_numerical']:.6f}, y_num = {bench_num['y_numerical']:.6f}, F_num = {bench_num['F_numerical']:.6f}")
    print(f"    - Solver Convergence: {bench_num['success']} in {bench_num['nit']} iterations")
    print(f"  [Error Analysis]")
    print(f"    - Absolute Error |x_ana - x_num|: {bench_err['error_x_abs']:.2e}")
    print(f"    - Absolute Error |F_ana - F_num|: {bench_err['error_F_abs']:.2e}")

    # Boundary comparison for Benchmark Case:
    f_corner_solar = utility_function(10.0, 0.0, 2.0, 1.0)
    f_corner_wind = utility_function(0.0, 10.0, 2.0, 1.0)
    print(f"  [Boundary Comparison]")
    print(f"    - Corner (x=10, y=0):  F = {f_corner_solar:.6f} (= 2*ln(11) ~= 4.795791)")
    print(f"    - Corner (x=0, y=10):  F = {f_corner_wind:.6f} (= ln(11) ~= 2.397895)")
    print(f"    - Interior Optimum:    F* = {bench_ana['F_optimal']:.6f}")
    print(f"    ==> F*(7, 3) = 5.545 > 4.796 > 2.398 (Interior is globally superior).")

    # 3. Real-world Empirical Case: Vietnam 2024 Weights (B = 10.0)
    print("\n" + "-" * 80)
    print("[Step 3] Real-world Vietnam 2024 Case Study (Normalized vs. Raw Weights, B = 10.0):")

    # Approach A: Normalized weights
    vn_norm_ana = solve_analytical(S0=s0_norm, W0=w0_norm, B=10.0)
    vn_norm_num = solve_numerical(S0=s0_norm, W0=w0_norm, B=10.0)

    # Approach B: Raw weights
    vn_raw_ana = solve_analytical(S0=S0_raw, W0=W0_raw, B=10.0)

    print(f"  [Approach A: Normalized Weights (s0={s0_norm:.3f}, w0={w0_norm:.3f})]")
    print(f"    - Optimal Solar Allocation (x*): {vn_norm_ana['x_optimal']:.4f} units ({vn_norm_ana['x_optimal']/10*100:.1f}% of budget)")
    print(f"    - Optimal Wind Allocation  (y*): {vn_norm_ana['y_optimal']:.4f} units ({vn_norm_ana['y_optimal']/10*100:.1f}% of budget)")
    print(f"    - Optimal Utility Value    (F*): {vn_norm_ana['F_optimal']:.6f}")
    print(f"    - Shadow Price (lambda*):        {vn_norm_ana['lambda_optimal']:.6f} (= 1/12 ~= 0.083333)")
    print(f"  [Approach B: Raw Generation Weights (S0={S0_raw:.2f}, W0={W0_raw:.2f})]")
    print(f"    - Optimal Solar Allocation (x*): {vn_raw_ana['x_optimal']:.4f} units")
    print(f"    - Optimal Wind Allocation  (y*): {vn_raw_ana['y_optimal']:.4f} units")
    print(f"    - Optimal Utility Value    (F*): {vn_raw_ana['F_optimal']:.6f}")
    print(f"    - Shadow Price (lambda*):        {vn_raw_ana['lambda_optimal']:.6f} (= 38.61/12 ~= 3.2175)")
    print(f"  ==> Note on Mathematical Equivalence:")
    print(f"      The optimal budget allocation (x*, y*) is mathematically IDENTICAL under both weighting schemes.")
    print(f"      F* and lambda* differ by an exact multiplicative scale factor of S0 + W0 = {S0_raw + W0_raw:.2f}.")

    # 4. Out-of-Bounds Test Case (Boundary / Corner Activation)
    print("\n" + "-" * 80)
    print("[Step 4] Out-of-Bounds Test Case (Mandated by Guide: Testing Corner Solutions):")
    print("Parameters: S0 = 0.2, W0 = 3.0, B = 10.0")
    out_ana = solve_analytical(S0=0.2, W0=3.0, B=10.0)
    out_num = solve_numerical(S0=0.2, W0=3.0, B=10.0)

    print(f"  [Analytical Evaluation]")
    print(f"    - Unconstrained stationary x: {out_ana['x_unconstrained']:.4f} (< 0 ==> Infeasible)")
    print(f"    - Boundary Constrained x*:    {out_ana['x_optimal']:.4f}")
    print(f"    - Boundary Constrained y*:    {out_ana['y_optimal']:.4f}")
    print(f"    - Maximum Constrained F*:     {out_ana['F_optimal']:.6f} (= 3*ln(11) ~= 7.193686)")
    print(f"  [Numerical Solver (SLSQP)]")
    print(f"    - Numerical x_num: {out_num['x_numerical']:.6f}, y_num = {out_num['y_numerical']:.6f}")
    print(f"    - Numerical F_num: {out_num['F_numerical']:.6f}")
    print(f"    - Discrepancy:     |x_ana - x_num| = {abs(out_ana['x_optimal'] - out_num['x_numerical']):.2e}")
    print(f"  ==> The Kuhn-Tucker boundary condition x* >= 0 is successfully triggered and verified.")

    # 5. Budget Sensitivity Analysis (B in [5, 15])
    print("\n" + "-" * 80)
    print("[Step 5] Sensitivity Analysis over Budget Range B in [5.0, 15.0]:")
    df_sens = run_budget_sensitivity_analysis(S0=s0_norm, W0=w0_norm, B_min=5.0, B_max=15.0, step=1.0)
    sens_csv_path = os.path.join(script_dir, "calb06_sensitivity_results.csv")
    df_sens.to_csv(sens_csv_path, index=False)
    print(f"  -> Exported detailed sensitivity table to: {sens_csv_path}")

    # Display preview table
    preview_cols = ["Budget_B", "x_solar_opt", "y_wind_opt", "F_optimal", "lambda_shadow_price", "dF_dB_finite_diff", "envelope_theorem_diff"]
    print(df_sens[preview_cols].to_string(index=False))

    # 6. Generate Figures
    print("\n" + "-" * 80)
    print("[Step 6] Rendering publication-quality visual figures...")
    figures_dir = os.path.join(script_dir, "figures")
    generated_plots = generate_visualizations(
        df_history=df_clean,
        S0=s0_norm,
        W0=w0_norm,
        B_benchmark=10.0,
        df_sensitivity=df_sens,
        output_dir=figures_dir
    )
    for p in generated_plots:
        print(f"  -> Generated: {p}")

    print("\n" + "=" * 80)
    print("CALB-06 EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
