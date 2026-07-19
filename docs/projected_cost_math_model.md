# Holguin-Wang Projected Deprivation Cost — Mathematical Formulation

## 1. Source Functions

### Holguin-Veras Deprivation Cost Function (DCF)

Holguin-Veras et al. (2013) estimate the deprivation cost for water as an exponential function calibrated over a 120-hour disaster horizon:

$$g_{\text{water}}(t_h) = e^{1.5031 + 0.1172\,t_h} - e^{1.5031}$$

where $t_h \in [0, 120]$ is deprivation time in **hours**.

Expressed in **days** ($x = t_h / 24$):

$$g_{\text{water}}(x) = e^{1.5031 + 2.8128\,x} - e^{1.5031}$$

since $0.1172 \times 24 = 2.8128$.

### Wang Deprivation Level Functions (DLF)

Wang et al. (2017) estimate supply-specific deprivation level functions using numerical rating scale (NRS) data:

$$W_c(x) = \frac{K_c}{1 + A_c\,e^{-B_c\,x}}, \quad x \text{ in days}$$

| Class $c$ | $K_c$ | $A_c$ | $B_c$ |
|-----------|--------|--------|--------|
| Medicine  | 9.772697 | 3.9031 | 0.7919 |
| Food      | 9.745492 | 4.2280 | 0.7407 |
| Tent      | 9.752874 | 4.2047 | 0.7437 |

Key insight: Wang et al. state that DLFs and DCFs are **almost identical in the short-time segment**, differing only at large deprivation times where logistic functions saturate while exponentials continue to grow.

## 2. Projection Method

### Baseline Correction

Wang's logistic has a positive value at $x = 0$. For a delay penalty, only the **increment** caused by delay should be charged:

$$\Delta W_c(x) = W_c(x) - W_c(0)$$

### Exponential Projection Family

Project the front segment onto a Holguin-Veras-style exponential family:

$$H_c(x) = \alpha_c \left(e^{a + \beta_c\,x} - e^{a}\right)$$

where:

| Symbol | Meaning | Value |
|--------|---------|-------|
| $a$ | Holguin-Veras intercept | 1.5031 |
| $\beta_c$ | Class-specific exponential growth rate | (projected) |
| $\alpha_c$ | Class-specific scale | (projected) |
| $x = \rho\,\tau$ | Literature deprivation time in days | |

### Two-Point Matching

Match: (1) initial marginal growth at $x = 0$; (2) cumulative increment at $x_{\tau,\max}$.

$$m_c = W_c'(0) = \frac{K_c\,A_c\,B_c}{(1+A_c)^2}, \qquad y_c = \Delta W_c(x_{\tau,\max})$$

where $x_{\tau,\max} = \rho\,H_\tau$ and $H_\tau = 4.4947$ h is the maximum tardiness from benchmark instances.

Solve:

$$\frac{e^{\beta_c\,x_{\tau,\max}} - 1}{\beta_c} = \frac{y_c}{m_c}, \qquad \alpha_c = \frac{m_c}{e^a\,\beta_c}$$

### Projected Parameters ($\rho = 1/24$, i.e., no time compression)

| Class | $m_c$ | $y_c$ | $\beta_c$ | $\alpha_c$ |
|-------|-------:|-------:|----------:|----------:|
| Medicine | 1.2565 | 0.2456 | 0.4558 | 0.6132 |
| Food     | 1.1166 | 0.2181 | 0.4464 | 0.5564 |
| Tent     | 1.1258 | 0.2199 | 0.4469 | 0.5604 |

Water retains the original Holguin-Veras DCF with $\beta_{\text{water}} = 2.8128$ (= $0.1172 \times 24$).

## 3. Operational Cost Function

### Core Formula

$$\boxed{f_{c_i}(\tau_i) = \lambda\,\omega_{c_i}\;\frac{e^{\,a\,+\,\beta_{c_i}\,\rho\,\tau_i} - e^{\,a}}{e^{\,a\,+\,\beta_{c_i}\,x_{\tau,\max}} - e^{\,a}}}$$

where:

| Symbol | Meaning | Value/Range |
|--------|---------|-------------|
| $\tau_i$ | Tardiness beyond soft deadline $o_i$ (hours) | $0 \leq \tau_i \leq l_i - o_i$ |
| $\lambda$ | Global cost-strength parameter | 8–20 (calibrated) |
| $\omega_{c_i}$ | Supply-class importance weight | See below |
| $\beta_{c_i}$ | Projected exponential growth rate | See above |
| $\rho$ | Time-scale mapping (operational hours → literature days) | $\{1/24, 0.10, 0.25, 0.50, 0.75, 1.00\}$ |
| $a$ | Holguin-Veras intercept | 1.5031 |
| $x_{\tau,\max}$ | Maximum tardiness in literature days | $\rho \times 4.4947$ |

### Equivalent Non-Normalized Form

$$f_{c_i}(\tau_i) = \lambda\,\omega_{c_i}\;\left(e^{\,a\,+\,\beta_{c_i}\,\rho\,\tau_i} - e^{\,a}\right)$$

This is the direct form without normalization. **Not recommended for calibration** because Water's $\beta = 2.8128$ causes cost magnitudes at $\tau_{\max}$ to be 24× larger than Medicine's at $\rho = 0.2083$.

### Why Normalization Is Recommended

The normalization divisor equalizes all classes at $\tau = H_\tau$:

$$f_{c_i}(H_\tau) = \lambda\,\omega_{c_i}$$

This **separates curve shape from class importance**: $\beta_c$ controls the curvature, $\omega_c$ controls relative importance, and $\lambda$ controls the overall delay-cost share. Without normalization, raw DLF magnitudes and DCF magnitudes are not directly comparable (Wang measures NRS, Holguin-Veras measures willingness-to-pay).

## 4. Complete Parameter Table

| Class $c$ | $\beta_c$ | $\omega_c$ | $\Delta^o_c$ (h) | $\Delta^l_c$ (h) |
|-----------|----------:|----------:|-------------------:|-------------------:|
| Medicine (C1) | 0.4558 | 1.35 | $U(0.50, 1.00)$ | $U(0.80, 1.50)$ |
| Water (C2)    | 2.8128 | 1.35 | $U(1.00, 1.80)$ | $U(1.20, 2.20)$ |
| Food (C3)     | 0.4464 | 1.00 | $U(1.80, 3.00)$ | $U(1.80, 3.00)$ |
| Tent (C4)     | 0.4469 | 0.75 | $U(2.50, 4.00)$ | $U(2.50, 4.50)$ |

### Supply-Class Deadlines

Each node $i$ is assigned a class $c_i$ and generates:

- Soft deadline: $o_i = r_i + \Delta^o_{c_i}$ (optimal arrival time)
- Hard deadline: $l_i = o_i + \Delta^l_{c_i}$ (latest arrival time)

where $r_i$ is the earliest reachable time from the depot.

## 5. Optimization Objective

$$\min \;\; c^T\!\sum_{k \in \mathcal{K}^T}\!\sum_{(i,j) \in A}\!d_{ij}^T\,x_{ijk} \;+\; c^D\!\sum_{d \in \mathcal{K}^D}\!\sum_{(i,j) \in A}\!d_{ij}^D\,y_{ijd} \;+\; \sum_{i \in N} f_{c_i}(\tau_i)$$

subject to:

$$0 \leq a_{ik}^T \leq l_i, \quad 0 \leq a_{id}^D \leq l_i \qquad \forall\, i \in N,\; k \in \mathcal{K}^T,\; d \in \mathcal{K}^D$$

$$a_{ik}^T - o_i \leq \tau_{ik}^T, \qquad a_{id}^D - o_i \leq \tau_{id}^D$$

$$\tau_{ik}^T \geq 0, \qquad \tau_{id}^D \geq 0$$

where $a_{ik}^T$ and $a_{id}^D$ are arrival times at node $i$ by truck $k$ and drone $d$ respectively.

## 6. Cost Values at Key Operating Points

### $\lambda = 12$, $\rho = 0.2083$, normalized

| $\tau$ (h) | Medicine | Water | Food | Tent |
|------------|----------|-------|------|------|
| 0.0  | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.1  | 0.29 | 0.08 | 0.22 | 0.16 |
| 0.25 | 0.73 | 0.20 | 0.54 | 0.41 |
| 0.5  | 1.48 | 0.43 | 1.10 | 0.83 |
| 1.0  | 3.03 | 1.00 | 2.25 | 1.69 |
| 2.0  | 6.36 | 2.79 | 4.73 | 3.54 |
| 4.49 | 16.20 | 16.20 | 12.00 | 9.00 |

### $\lambda = 12$, $\rho = 1/24$, normalized (no compression)

| $\tau$ (h) | Medicine | Water | Food | Tent |
|------------|----------|-------|------|------|
| 0.0  | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.1  | 0.35 | 0.28 | 0.26 | 0.19 |
| 0.25 | 0.87 | 0.69 | 0.64 | 0.48 |
| 0.5  | 1.73 | 1.41 | 1.29 | 0.96 |
| 1.0  | 3.49 | 2.90 | 2.58 | 1.94 |
| 2.0  | 7.04 | 6.17 | 5.22 | 3.91 |
| 4.49 | 16.20 | 16.20 | 12.00 | 9.00 |

## 7. Calibration Targets

| Diagnostic | Target |
|------------|--------|
| Average delay cost / transportation cost | 5%–12% |
| 90th percentile delay cost / transportation cost | < 20%–25% |
| Feasibility rate | 100% |
| Medicine & Water delay < Food & Tent delay | ✓ (enforced by $\Delta^o_c$) |
| Route structure meaningfully different from no-delay-penalty | ✓ |
| Transportation detour not excessively increased | ✓ |

Recommended initial calibration ranges:

$$\rho \in \{1/24,\; 0.10,\; 0.25,\; 0.50,\; 0.75,\; 1.00\}, \qquad \lambda \in \{8,\; 12,\; 16,\; 20\}$$

$$\omega_{\text{medicine}} \in [1.2,\; 1.5], \quad \omega_{\text{water}} \in [1.2,\; 1.5], \quad \omega_{\text{food}} = 1.0, \quad \omega_{\text{tent}} \in [0.4,\; 0.8]$$