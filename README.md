# Project CalB06: Wind & Solar Power Resource Allocation under Constraints

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/triduc-tran/CalB06_Wind_Solar_Optimization/blob/main/CalB06_Optimization.ipynb)

**Course:** Calculus in Business (*Giải tích trong kinh doanh*) — Semester HK261  
**Project Topic:** CalB-06 – *Điện gió, điện mặt trời và phân bổ ngân sách có ràng buộc*  
**Environment:** Python 3.10+ / Google Colab  

---

## 1. Overview & Mathematical Model

When allocating an investment budget between **Solar power ($x$)** and **Wind power ($y$)** under a fixed total budget $B$, planners must account for diminishing marginal returns:
$$\max_{x, y \ge 0} F(x, y) = S_0 \ln(1 + x) + W_0 \ln(1 + y) \quad \text{subject to} \quad x + y = B$$

- $x, y \ge 0$: Normalized capital budget allocated to Solar and Wind.
- $B \in [5, 15]$: Total budget.
- $S_0, W_0$: Empirical weights calibrated from Vietnam's 2024 electricity generation (from *Our World in Data*):
  - **Solar ($S_0$):** $25.86$ TWh (Normalized $s_0 = 0.6698 \approx 0.670$)
  - **Wind ($W_0$):** $12.75$ TWh (Normalized $w_0 = 0.3302 \approx 0.330$)

---

## 2. Code Structure (Mapped directly to Course Guide)

The code in `calb06_optimization.py` and `CalB06_Optimization.ipynb` is structured linearly into parts that follow the assignment:

| Part in Code | Guide Section | Content & Functionality |
| :--- | :--- | :--- |
| **Part 1** | Section 4: Task 1 | Read CSV data, extract 2024 generation ($S_0, W_0$), compute normalized weights ($s_0, w_0$). Explains degenerate case when a weight is zero. |
| **Part 2** | Section 4: Tasks 2, 3, 4 | Formulate Lagrangian $\mathcal{L}$, compute stationary formulas ($x^*, y^*, \lambda^*$), and verify strict concavity $f''(x) < 0$. |
| **Part 3** | Section 6: Verification | Handbook benchmark ($S_0=2, W_0=1, B=10$) showing $x^*=7.0, y^*=3.0, F^*=5.545$ and comparing with boundary points ($x=0$ and $x=10$). |
| **Part 4** | Section 6: Out-of-bounds | Out-of-bounds test case ($S_0=0.2, W_0=3.0, B=10$) showing stationary point $x = -0.25 < 0$ snapping to boundary $x^*=0, y^*=10$. |
| **Part 5** | Section 4: Baseline | Real-world Vietnam 2024 case study ($B=10$) showing optimal split: Solar $7.037$ units ($70.4\%$), Wind $2.963$ units ($29.6\%$). |
| **Part 6** | Section 6: Numerical Check | Solve with `scipy.optimize.minimize` (SLSQP) and verify machine-precision match with analytical formulas. |
| **Part 7** | Section 4: Task 5 | Loop over budget $B$ from 5 to 15, save results to `calb06_sensitivity_results.csv`. |
| **Part 8** | Section 6: Deliverables | Generate 3 publication-ready figures in `figures/`. |

---

## 3. How to Run

### Option 1: Standalone Python Script
```bash
python calb06_optimization.py
```

### Option 2: Google Colab
Click the badge above or open `CalB06_Optimization.ipynb` directly in Google Colab or Jupyter.

---

## 4. Key Verification Results

- **Handbook Benchmark ($S_0=2, W_0=1, B=10$):**
  - Analytical: $x^* = 7.0000$, $y^* = 3.0000$, $F^* = 5.545177$, $\lambda^* = 0.2500$.
  - Numerical (SLSQP): $x^* = 7.0000$, $y^* = 3.0000$, error $< 10^{-4}$.
  - Boundary comparison: $F(7, 3) = 5.545 > F(10, 0) = 4.796 > F(0, 10) = 2.398$ (Interior maximum confirmed).
- **Out-of-Bounds Test ($S_0=0.2, W_0=3.0, B=10$):**
  - Unconstrained: $x = -0.25 < 0$.
  - Constrained optimum: $x^* = 0.0000, y^* = 10.0000, F^* = 3\ln(11) \approx 7.193686$.
- **Vietnam 2024 Baseline ($B=10$):**
  - Solar allocation $x^* = 7.0373$ ($70.4\%$).
  - Wind allocation $y^* = 2.9627$ ($29.6\%$).
