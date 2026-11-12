# Project CalB06: Wind & Solar Power Resource Allocation under Budget Constraints

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/triduc-tran/CalB06_Wind_Solar_Optimization/blob/main/CalB06_Optimization.ipynb)

**Course:** Calculus in Business (*Giải tích trong kinh doanh*) — Semester HK261  
**Project Topic:** CalB-06 – *Điện gió, điện mặt trời và phân bổ ngân sách có ràng buộc*  
**Environment:** Python 3.10+ / Google Colab / Jupyter Notebook  

---

## 1. Project Overview & Problem Formulation

In energy infrastructure planning, capital must be divided across competing renewable technologies under a binding budget constraint $B$. Due to integration bottlenecks, transmission curtailment, and diminishing marginal generation returns, dedicating 100% of capital to a single technology rarely optimizes aggregate welfare.

This project investigates the constrained optimization problem:
$$\max_{x, y \ge 0} F(x, y) = S_0 \ln(1 + x) + W_0 \ln(1 + y) \quad \text{subject to} \quad x + y = B$$

where:
- $x \ge 0$: Normalized capital budget allocated to **Solar Photovoltaic (PV)**.
- $y \ge 0$: Normalized capital budget allocated to **Wind Power**.
- $B$: Total normalized budget parameter ($B \in [5, 15]$).
- $S_0, W_0$: Positive preference/productivity weights calibrated from Vietnam's empirical 2024 electricity generation (*Our World in Data*).
- $\ln(1 + \cdot)$: Logarithmic utility ensuring zero investment yields zero benefit ($F(0, 0) = 0$), avoiding logarithmic singularity at zero.

---

## 2. Repository Structure

```text
.
├── 0_ BTL Tiếng Việt.pdf             # Official assignment syllabus and project guidelines
├── 01_ CalB_Topic6.xlsx              # Group project management, timeline, and role assignment
├── 2. + 5._.docx                     # Initial theoretical draft
├── vietnam_solar_wind_cleaned.csv    # Historical OWID dataset for Vietnam (2000–2024)
├── calb06_optimization.py            # Main modular Python execution pipeline
├── CalB06_Optimization.ipynb         # Interactive Google Colab / Jupyter Notebook
├── build_notebook.py                 # Reproducible script to generate the notebook
├── calb06_sensitivity_results.csv    # Exported sensitivity analysis dataset (B in [5, 15])
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation and algorithm reference
└── figures/                          # 300-DPI publication-ready charts
    ├── fig1_historical_electricity_vietnam.png
    ├── fig2_optimization_geometry.png
    ├── fig3_budget_sensitivity_analysis.png
    └── fig4_boundary_corner_solution.png
```

---

## 3. Function & Algorithm Dictionary (Annotated)

Every function and algorithm in `calb06_optimization.py` is annotated below with its theoretical purpose, mathematical formulation, and practical usage:

| Function Name | Mathematical / Algorithmic Concept | Meaning & Practical Use |
| :--- | :--- | :--- |
| `load_and_preprocess_data(filepath)` | **Data Pipeline & Sanitization** | Ingests historical time-series (2000–2024), filters out non-numeric trailing metadata notes, extracts 2024 empirical generation ($S_0 = 25.86$ TWh, $W_0 = 12.75$ TWh), and computes normalized weights ($s_0 \approx 0.670, w_0 \approx 0.330$). |
| `utility_function(x, y, S0, W0)` | **Bivariate Objective Function**<br>$F(x,y) = S_0\ln(1+x) + W_0\ln(1+y)$ | Evaluates total policy utility. Guarantees diminishing marginal returns: $\partial F/\partial x = S_0/(1+x) > 0$, $\partial^2 F/\partial x^2 = -S_0/(1+x)^2 < 0$. |
| `single_variable_utility(x, B, S0, W0)` | **Single-Variable Reduction**<br>$f(x) = S_0\ln(1+x) + W_0\ln(1+B-x)$ | Substitutes $y = B - x$ ($x \in [0, B]$) to transform a 2-variable constrained problem into an unconstrained 1-variable problem over a compact domain. |
| `single_variable_first_derivative(...)` | **First-Order Derivative (FOC)**<br>$f'(x) = \frac{S_0}{1+x} - \frac{W_0}{1+B-x}$ | Measures net marginal utility gained by shifting 1 budget unit from wind to solar. At optimal $x^*$, $f'(x^*) = 0 \iff$ marginal utilities are equalized. |
| `single_variable_second_derivative(...)` | **Second-Order Concavity (SOC)**<br>$f''(x) = -\frac{S_0}{(1+x)^2} - \frac{W_0}{(1+B-x)^2}$ | Proves $f''(x) < 0$ strictly $\forall x \in [0, B]$. Guarantees that any stationary point is the **unique global maximum**. |
| `bordered_hessian_analysis(x, y, S0, W0)` | **Bordered Hessian Matrix ($\bar{H}$)**<br>$\det(\bar{H}) = \frac{S_0}{(1+x)^2} + \frac{W_0}{(1+y)^2} > 0$ | Evaluates the Second-Order Sufficient Condition (SOSC) in the full $\mathbb{R}^2$ constrained formulation. Positive determinant confirms a constrained maximum. |
| `solve_analytical(S0, W0, B)` | **Lagrange & KKT Closed-Form Solver**<br>$x^* = \frac{S_0(B+1) - W_0}{S_0 + W_0}$<br>$\lambda^* = \frac{S_0+W_0}{B+2}$ | Calculates closed-form optimal solutions. Tests interior feasibility ($0 \le x^* \le B$); if boundary conditions are violated, automatically activates Kuhn-Tucker corner solutions ($x^*=0$ or $x^*=B$). |
| `solve_numerical(S0, W0, B)` | **SLSQP Constrained Optimization**<br>$\min -F(x,y)$ s.t. $x+y=B, x,y \ge 0$ | Solves the non-linear program numerically via `scipy.optimize.minimize` using analytical gradients and equality constraints. |
| `compare_analytical_vs_numerical(...)` | **Precision & Convergence Verification** | Calculates absolute error $|x_{\text{ana}} - x_{\text{num}}|$ and $|F_{\text{ana}} - F_{\text{num}}|$ to verify that numerical solvers converge to the exact analytical global maximum. |
| `run_budget_sensitivity_analysis(...)` | **Sensitivity Grid & Envelope Theorem**<br>$\frac{dF^*}{dB} = \lambda^*(B) = \frac{S_0+W_0}{B+2}$ | Evaluates allocations across $B \in [5, 15]$. Verifies the Envelope Theorem by comparing analytical shadow price $\lambda^*$ against numerical finite difference $\frac{\Delta F^*}{\Delta B}$. |
| `generate_visualizations(...)` | **Academic Graphical Suite (300 DPI)** | Renders and saves 4 publication-quality figures for report integration. |

---

## 4. Key Mathematical Results & Verification

### 4.1 Benchmark Verification ($S_0 = 2, W_0 = 1, B = 10$)
From Section 6 of the project syllabus:
- **Optimal Solar Allocation ($x^*$):** $7.0000$ (Exact formula: $\frac{2(11) - 1}{3} = 7$)
- **Optimal Wind Allocation ($y^*$):** $3.0000$ (Exact formula: $10 - 7 = 3$)
- **Maximum Utility ($F^*$):** $5.545177$ (Exact formula: $8\ln(2)$)
- **Shadow Price ($\lambda^*$):** $0.2500$ (Exact formula: $\frac{3}{12} = 0.25$)
- **Strict Concavity:** $f''(x^*) = -0.093750 < 0$, $\det(\bar{H}) = 0.093750 > 0$.
- **Numerical SLSQP Discrepancy:** $|x_{\text{ana}} - x_{\text{num}}| = 2.39 \times 10^{-8}$ (Floating-point precision).
- **Corner Allocations Comparison:**
  - All Solar ($x=10, y=0$): $F = 2\ln(11) \approx 4.795791$
  - All Wind ($x=0, y=10$): $F = \ln(11) \approx 2.397895$
  - Interior Optimum ($x=7, y=3$): $F^* = 5.545177$  
  $\implies F^*(7, 3) > F(10, 0) > F(0, 10)$, confirming interior global superiority.

### 4.2 Out-of-Bounds Corner Solution Test ($S_0 = 0.2, W_0 = 3.0, B = 10$)
- Unconstrained stationary point: $x_{\text{uncon}} = \frac{0.2(11) - 3.0}{3.2} = -0.25 < 0$ (Infeasible).
- Because $f'(x) < 0$ for all $x \in [0, 10]$, utility is strictly decreasing on the feasible interval.
- **Kuhn-Tucker Boundary Solution:** $x^* = 0.0000$, $y^* = 10.0000$, $F^* = 3\ln(11) \approx 7.193686$.
- Numerical solver converges exactly to $(0.0000, 10.0000)$ with error $0.00$.

### 4.3 Real-World Vietnam 2024 Baseline ($B = 10$)
- **Empirical Generation (OWID):** Solar $= 25.86$ TWh ($67.0\%$), Wind $= 12.75$ TWh ($33.0\%$).
- **Normalized Optimal Allocation:** $x^* = 7.0373$ units ($70.4\%$), $y^* = 2.9627$ units ($29.6\%$).
- **Equivalence Note:** Whether using raw weights ($S_0=25.86, W_0=12.75$) or normalized weights ($s_0=0.670, w_0=0.330$), the optimal capital split $(x^*, y^*)$ is **mathematically identical**. Utility $F^*$ and shadow price $\lambda^*$ scale by exactly $S_0 + W_0 = 38.61$.

### 4.4 Budget Sensitivity & Envelope Theorem ($B \in [5, 15]$)
- Expansion paths are linear:
  $$\frac{dx^*}{dB} = \frac{S_0}{S_0 + W_0} = s_0 \approx 0.670, \quad \frac{dy^*}{dB} = \frac{W_0}{S_0 + W_0} = w_0 \approx 0.330$$
- Shadow price $\lambda^*(B) = \frac{dF^*}{dB} = \frac{1}{B + 2}$ declines from $0.1429$ ($B=5$) to $0.0588$ ($B=15$).
- Numerical finite difference $\frac{\Delta F^*}{\Delta B}$ matches analytical $\lambda^*(B)$ within truncation error ($< 0.001$).

---

## 5. How to Run the Project

### Option A: Running the Standalone Python Pipeline
1. Ensure Python 3.10+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python calb06_optimization.py
   ```
4. Output figures are automatically generated in `figures/`, and numerical sensitivity data is saved to `calb06_sensitivity_results.csv`.

### Option B: Running via Google Colab / Jupyter Notebook
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `CalB06_Optimization.ipynb` and `vietnam_solar_wind_cleaned.csv`.
3. Select **Runtime -> Run all**.
4. All derivations, tables, and charts will render interactively inline.

---

## 6. Academic Discussion & Policy Insights

1. **Nature of Utility Function $F(x, y)$:**
   $F$ represents a dimensionless policy index capturing aggregate social/grid welfare under diminishing marginal returns. It does **not** represent a direct engineering forecast of MWh generation or financial profit.
2. **Lagrange Multiplier $\lambda$ (Shadow Price):**
   $\lambda^*$ quantifies the marginal societal benefit per additional unit of normalized investment budget ($\partial F^*/\partial B$). It is distinct from wholesale market clearing prices or Feed-in Tariffs (FiT).
3. **Realistic Extensions for National Power Planning (PDP8):**
   - **Cost Differentials:** $c_x x + c_y y \le B$ where $c_x, c_y$ reflect Levelized Cost of Electricity (LCOE) or CAPEX per installed MW.
   - **Grid Stability Reserves:** $x \ge x_{\min}, y \ge y_{\min}$ to ensure minimum spinning inertia and reactive power control.
   - **Transmission Bottlenecks:** $x \le x_{\max}, y \le y_{\max}$ to account for regional substation hosting capacity limits.
