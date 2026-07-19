# Holguin-Wang Exponential Projection for Supply-Class Delay Costs

## 1. Purpose

This note documents a unified way to convert the short-time segment of Wang et al.'s supply-specific deprivation level functions into a Holguin-Veras-style exponential deprivation cost family. The goal is to keep the economic-cost interpretation of the Holguin-Veras function while using Wang et al. to justify heterogeneous supply classes and class-specific delay sensitivity.

The proposed modeling logic is:

1. Use Holguin-Veras et al. (2013) as the theoretical basis for deprivation cost and social cost.
2. Use Wang et al. (2017) as the empirical basis for heterogeneous relief-supply deprivation patterns.
3. Use Wang's statement that DLFs and Holguin-Veras DCFs are almost the same in the early response stage to project the Wang logistic front segment onto an exponential function family.
4. Use calibrated class parameters and deadlines to reflect medicine, water, food, and tent urgency.

## 2. Current Delay-Cost Formulation in the Paper

The paper currently defines the tardiness cost function as:

```tex
f(\tau) = e^{1.5031 + 7.032\tau} - e^{1.5031},
```

where `tau` is the tardiness beyond the soft deadline `o_i`.

The objective function is:

```tex
\min \; c^\mathrm{T}\sum_{k\in \mathcal{K}^\mathrm{T}}\sum_{(i,j)\in A}d_{ij}^\mathrm{T} x_{ijk}
+ c^\mathrm{D}\sum_{d\in \mathcal{K}^\mathrm{D}}\sum_{(i,j)\in A}d_{ij}^\mathrm{D} y_{ijd}
+ \sum_{i\in N}\left(\sum_{k\in \mathcal{K}^\mathrm{T}}f(\tau_{ik}^\mathrm{T})
+\sum_{d\in \mathcal{K}^\mathrm{D}}f(\tau_{id}^\mathrm{D})\right).
```

The time-window constraints enforce:

```tex
0 \leq a_{ik}^{\mathrm{T}} \leq l_i,
0 \leq a_{id}^{\mathrm{D}} \leq l_i,
a_{ik}^{\mathrm{T}} - o_i \leq \tau_{ik}^{\mathrm{T}},
a_{id}^{\mathrm{D}} - o_i \leq \tau_{id}^{\mathrm{D}}.
```

Thus, arrivals after the soft deadline incur a nonlinear delay cost, while arrivals after the hard deadline are infeasible.

The experimental section explains that this cost is a rescaled version of the Holguin-Veras water deprivation cost:

```tex
g(\tau) = e^{1.5031 + 0.1172\tau} - e^{1.5031},
```

where the original function is calibrated over a 120-hour deprivation horizon. The paper rescales the time axis to make the nonlinear cost visible over the shorter operational horizon of the routing instances.

For this project, the benchmark horizon should be derived from the generated instances rather than chosen arbitrarily. Scanning the 75 standard instances in `data/Instance10`, `data/Instance25`, `data/Instance50`, `data/Instance75`, and `data/Instance100` with the current class-based deadline generator gives:

| Horizon statistic | Value | Instance |
|---|---:|---|
| Maximum absolute hard deadline `max_i l_i` | 8.7187 h | `R_50_75_1` |
| Maximum soft deadline `max_i o_i` | 4.3608 h | `R_50_75_1` |
| Maximum allowable tardiness `max_i(l_i - o_i)` | 4.4947 h | `R_30_50_1` and other class-based runs |

Because the delay-cost function uses tardiness `tau = max(0, arrival - o_i)`, the relevant delay-time interval is:

```tex
\tau \in [0, H_\tau], \qquad H_\tau = \max_i(l_i-o_i) = 4.4947 \text{ h}.
```

The absolute hard-deadline horizon, `8.7187 h`, is useful for describing the full operational schedule horizon, but it should not be used as the upper bound of the delay-cost curve unless the curve is plotted against absolute arrival time instead of tardiness.

## 3. Source Functions

### 3.1 Holguin-Veras Deprivation Cost Function

Holguin-Veras et al. (2013) propose deprivation cost as part of the appropriate social-cost objective for post-disaster humanitarian logistics. Social cost combines logistics cost and deprivation cost, where deprivation cost monetizes human suffering caused by delayed access to critical relief goods.

For water, the commonly used original exponential form is:

```tex
g_{\mathrm{water}}(t_h)
= e^{1.5031 + 0.1172 t_h} - e^{1.5031},
```

where `t_h` is deprivation time in hours. If the independent variable is expressed in days, `x = t_h / 24`, the same function becomes:

```tex
g_{\mathrm{water}}(x)
= e^{1.5031 + 2.8128x} - e^{1.5031},
```

because `2.8128 = 0.1172 * 24`.

### 3.2 Wang Deprivation Level Functions

Wang et al. (2017) estimate deprivation level functions (DLFs) using a numerical rating scale (NRS). They explicitly distinguish DLFs from deprivation cost functions (DCFs): NRS data measure perceived suffering levels but do not contain economic willingness-to-pay information. Therefore, Wang's DLFs should not be directly interpreted as monetary deprivation costs.

Wang et al. estimate logistic DLFs for three relief-supply types:

```tex
W_c(x) = \frac{K_c}{1 + A_c e^{-B_c x}},
```

where `x` is deprivation time in days. Table 6 in Wang et al. reports:

| Class | Wang DLF |
|---|---|
| Tent | `9.752874 / (1 + 4.2047 e^{-0.7437x})` |
| Food | `9.745492 / (1 + 4.2280 e^{-0.7407x})` |
| Medicine | `9.772697 / (1 + 3.9031 e^{-0.7919x})` |

The important theoretical bridge is Wang et al.'s discussion that Holguin-Veras DCFs are monotonic, nonlinear, and convex in deprivation time, while Wang DLFs are weakly monotonic, nonlinear, convex at first, and concave later. Wang et al. state that the two function types are almost the same for short deprivation times and differ mainly when deprivation time becomes large. This supports using a Holguin-Veras-style exponential function as a short-time approximation of Wang's logistic DLF front segment.

## 4. Projection Method

### 4.1 Baseline Adjustment

Wang's logistic DLF has a positive value at `x = 0`. For a delay penalty, only the additional deprivation caused by delay should be charged. Therefore, define the baseline-adjusted Wang increment:

```tex
\Delta W_c(x) = W_c(x) - W_c(0).
```

### 4.2 Exponential Projection Family

Project the front segment of `Delta W_c(x)` onto a Holguin-Veras-style exponential family:

```tex
H_c(x) = \alpha_c \left(e^{a + \beta_c x} - e^a\right),
```

where:

| Symbol | Meaning |
|---|---|
| `a = 1.5031` | Holguin-Veras intercept |
| `beta_c` | class-specific exponential growth rate |
| `alpha_c` | class-specific scale matching Wang's DLF magnitude |
| `x` | literature deprivation time in days |

This keeps the Holguin-Veras exponential structure but lets each supply class have different parameters.

### 4.3 Two-Point Matching

A simple and transparent calibration is to match:

1. The initial marginal growth at `x = 0`.
2. The cumulative deprivation increment at the project-specific maximum tardiness horizon.

For Wang's logistic function:

```tex
W_c'(0) = \frac{K_c A_c B_c}{(1 + A_c)^2}.
```

Let:

```tex
m_c = W_c'(0),
y_c = \Delta W_c(x_{\tau,\max}).
```

Here:

```tex
x_{\tau,\max} = \rho H_\tau,
```

where `H_tau = 4.4947 h` is the maximum allowable tardiness in the generated benchmark instances and `rho` maps operational hours to literature deprivation days. If no additional time compression is applied, then:

```tex
\rho = \frac{1}{24}, \qquad x_{\tau,\max} = \frac{4.4947}{24} = 0.1873 \text{ days}.
```

The exponential projection has:

```tex
H_c'(0) = \alpha_c e^a \beta_c,
H_c(x_{\tau,\max}) = \alpha_c e^a (e^{\beta_c x_{\tau,\max}} - 1).
```

Matching the initial slope and the reference-point value gives:

```tex
\alpha_c e^a \beta_c = m_c,
```

```tex
\alpha_c e^a (e^{\beta_c x_{\tau,\max}} - 1) = y_c.
```

Eliminating `alpha_c`, solve:

```tex
\frac{e^{\beta_c x_{\tau,\max}} - 1}{\beta_c}
= \frac{y_c}{m_c}.
```

Then:

```tex
\alpha_c = \frac{m_c}{e^a \beta_c}.
```

This method is easy to explain in the paper because it preserves both the local slope and the short-time cumulative effect of Wang's DLF.

## 5. Projected Parameters

Using the project-specific no-extra-compression endpoint `x_{\tau,\max} = 4.4947/24 = 0.1873 days`, the projected exponential parameters are:

| Class | `m_c = W_c'(0)` | `Delta W_c(x_tau,max)` | `beta_c` | `alpha_c` |
|---|---:|---:|---:|---:|
| Medicine | 1.2565 | 0.2456 | 0.4558 | 0.6132 |
| Food | 1.1166 | 0.2181 | 0.4464 | 0.5564 |
| Tent | 1.1258 | 0.2199 | 0.4469 | 0.5604 |

These values correspond to `rho = 1/24`, i.e., one operational hour is treated as one actual hour in Wang's day-scale DLF. If `rho` is later selected as a stronger operational time-compression parameter, the endpoint should be recomputed as `x_{\tau,\max} = rho * 4.4947`, and the two-point projection parameters should be recomputed accordingly. In other words, the benchmark-derived operational delay interval remains fixed at `[0, 4.4947 h]`, while the literature-scale endpoint depends on the chosen time-scale mapping.

For comparison, using each logistic curve's inflection point as the front-segment endpoint gives:

| Class | Inflection point `ln(A_c)/B_c` | `beta_c` | `alpha_c` |
|---|---:|---:|---:|
| Medicine | 1.720 days | 0.3245 | 0.8613 |
| Food | 1.946 days | 0.3178 | 0.7816 |
| Tent | 1.931 days | 0.3181 | 0.7873 |

The fitted `beta_c` values are very close across medicine, food, and tent. This is consistent with Wang et al.'s statement that the short-time behavior of DLFs is close to the Holguin-Veras DCF shape. In practice, the class differences should therefore be represented more strongly through deadline generation and class weights than through sharply different exponential growth rates.

## 6. Recommended Operational Cost Function

For the routing model, the operational tardiness is measured in hours and is bounded by the generated hard deadlines:

```tex
0 \leq \tau_i \leq l_i-o_i \leq H_\tau = 4.4947 \text{ h}.
```

Define:

```tex
x_i = \rho \tau_i,
```

where:

| Symbol | Meaning |
|---|---|
| `tau_i` | operational tardiness beyond the soft deadline, in hours |
| `rho` | mapping from operational hours to literature deprivation days |
| `x_i` | projected deprivation time in literature days |

The class-specific delay cost can be written as:

```tex
f_{c_i}(\tau_i)
= \lambda \omega_{c_i}
\left(e^{1.5031 + \beta_{c_i}\rho\tau_i} - e^{1.5031}\right),
```

where:

| Symbol | Meaning |
|---|---|
| `lambda` | global cost-strength parameter controlling the delay-cost share in total objective value |
| `omega_c` | supply-class importance weight |
| `beta_c` | projected exponential growth rate |
| `rho` | time-scale mapping |

An alternative normalized version is:

```tex
f_{c_i}(\tau_i)
= \lambda \omega_{c_i}
\frac{e^{1.5031 + \beta_{c_i}\rho\tau_i} - e^{1.5031}}
{e^{1.5031 + \beta_{c_i}x_{\tau,\max}} - e^{1.5031}}.
```

The normalized version is preferable for calibration because it separates curve shape from class importance. Without normalization, raw water costs and Wang-derived DLF values are not directly comparable because they come from different measurement methods.

## 7. Treatment of Water

Wang et al. (2017) provide logistic DLF parameters for medicine, food, and tent, but **not for water**. The Holguin-Veras (2013) water DCF has slope 0.1172/hour, which converts to `beta_water = 2.8128/day` — **6x larger** than the other classes' projected betas (~0.45).

### 7.1 Problem with beta_water = 2.8128

Using beta_water = 2.8128 in the normalized cost function causes:

- **Normalization disaster**: The normalizing denominator `exp(1.5031 + 2.8128 * rho * H_tau) - exp(1.5031)` becomes astronomically large (e.g., ~1.88M at rho=1.0), squashing water delay costs near zero for `tau < H_tau`.
- **Non-normalized explosion**: Without normalization, water costs reach ~1.88M at `tau = H_tau` under rho=1.0, making water delay dominate all other costs.
- **Inconsistency**: Water's cost curve shape (steep exponential) is fundamentally different from the other three classes (near-identical logistic-projected exponentials), violating the model's assumption that all classes follow the same structural family.

### 7.2 Resolution: Projected beta_water = 0.4525

Since Wang et al. do not provide a logistic DLF for water, we adopt the **projected exponential growth rate** that is consistent with the other three classes. This is justified by:

1. **Wang et al.'s own observation** that DLFs are nearly identical in the short-time segment (0–5 hours operational), differing mainly in saturation levels and inflection points at much longer time scales.
2. **Physical reasoning**: In the 0–5 hour operational window, water and medicine deprivation have comparable urgency patterns — both are survival-critical supplies whose marginal cost starts low and grows exponentially.
3. **Numerical coherence**: Using `beta_water = 0.4525` (the average of medicine's 0.4558 and food's 0.4464, weighted toward medicine) keeps all four classes within the same near-linear regime for realistic `rho` values, ensuring that **class differentiation is fully captured by the class weights `omega_c` and the deadline windows `(delta_o, delta_l)`**.

The updated parameter table is:

| Class | `beta_c` | `omega_c` | Source |
|---|---:|---:|---|
| Medicine | 0.4558 | 1.35 | Wang projection |
| Water | 0.4525 | 1.35 | Consensus proxy (Med/Food average) |
| Food | 0.4464 | 1.00 | Wang projection |
| Tent | 0.4469 | 0.75 | Wang projection |

### 7.3 Verification

With `beta_water = 0.4525` and normalized form, all four supply classes produce cost curves of nearly identical **shape**, differing only in magnitude via `omega_c`:

```
tau(h)   Medicine    Water       Food         Tent
0.5      0.614       0.619       0.466        0.349
1.0      1.384       1.395       1.049        0.786
2.0      3.568       3.589       2.688        2.014
4.5      16.200      16.200      12.000       9.000
```

The class ordering (Medicine ≈ Water > Food > Tent) correctly reflects urgency, and the deadline windows ensure Medicine/Water have tighter deadlines (arriving sooner, incurring delay sooner).

## 8. Calibration Targets

The calibration should choose `lambda`, `rho`, and `omega_c` based on route behavior rather than only curve appearance. Recommended diagnostics:

| Diagnostic | Target |
|---|---|
| Average delay cost / transportation cost | 5%-12% |
| 90th percentile delay cost / transportation cost | below 20%-25% |
| Feasibility rate | 100% |
| Medicine and water delay | lower than food and tent delay |
| Route structure | meaningfully different from a no-delay-penalty baseline |
| Transportation detour | not excessively increased only to avoid tiny tardiness |

Initial operational candidates:

```text
rho in {0.2083, 1.0}   (rho=0.2083 = 5/24 maps 1h operation to 0.2083 days in Wang scale)
lambda in {20, 30, 50}
omega_medicine = 1.35
omega_water    = 1.35
omega_food     = 1.00
omega_tent     = 0.75
```

### 8.1 Test Results (R_30_25, 3 instances, seed=42, 2000 iters)

With `beta_water = 0.4525`, `normalized = True`:

| λ | ρ | R_30_25_1 | R_30_25_2 | R_30_25_3 | Avg Delay% |
|---|---|---|---|---|---|
| 20 | 0.2083 | 9.6% | 4.3% | 4.2% | **6.0%** |
| 20 | 1.0 | 2.3% | 1.4% | 2.0% | **1.9%** |
| 30 | 0.2083 | 8.6% | 5.7% | 4.7% | **6.3%** |
| 30 | 1.0 | 2.8% | 3.6% | 2.9% | **3.1%** |
| 50 | 0.2083 | 14.3% | 16.5% | 7.9% | **12.9%** |
| 50 | 1.0 | 3.5% | 8.5% | 5.4% | **5.8%** |
| 80 | 0.2083 | 30.5% | 2.7% | 13.6% | **15.6%** |
| 80 | 1.0 | 8.5% | 10.6% | 5.2% | **8.1%** |

**Recommended configuration**: `lambda = 30`, `rho = 0.2083` (targeting 5–8% delay/transport ratio).

When testing these `rho` values, the operational interval is still `[0, 4.4947 h]`. The implied Wang literature endpoints are:

| `rho` | `x_tau,max = rho * 4.4947` |
|---:|---:|
| 0.75 | 3.3710 days |
| 1.00 | 4.4947 days |
| 1.25 | 5.6184 days |
| 1.50 | 6.7420 days |

These settings intentionally compress the Wang literature time scale into the operational delay interval. If this is considered too aggressive, use a smaller range such as `rho in {1/24, 0.10, 0.25, 0.50}`.

All curve plots and PWL checks should use the benchmark-derived tardiness interval:

```text
tau_hours in [0, 4.4947]
```

If a plot is intended to show the full absolute schedule horizon rather than tardiness, use:

```text
arrival_time_hours in [0, 8.7187]
```

## 9. Suggested Paper Wording

The following wording can be adapted for the paper:

```text
Following Holguin-Veras et al. (2013), we model deprivation-related tardiness penalties using an exponential function. To account for heterogeneous relief supplies, we further use the deprivation level functions estimated by Wang et al. (2017). Wang et al. show that deprivation level functions estimated from numerical rating scale data are logistic in the long run, but their early-stage behavior is consistent with the monotonic, nonlinear, and convex deprivation cost functions proposed by Holguin-Veras et al. Motivated by this observation, we project the early segment of each Wang et al. supply-specific logistic function onto a Holguin-Veras-style exponential family. This yields a unified class-specific delay cost function with the same exponential structure but different calibrated parameters for medicine, water, food, and tents.
```

If using the normalized form:

```text
Because Wang et al.'s deprivation level functions are based on NRS data and do not directly contain economic information, we do not interpret their raw values as monetary costs. Instead, they are used to calibrate the relative shape and urgency of supply-specific delay penalties. A global scaling parameter is then introduced to align the magnitude of delay penalties with transportation costs in the routing objective.
```

## 10. References

Holguin-Veras, J., Perez, N., Jaller, M., Van Wassenhove, L. N., & Aros-Vera, F. (2013). On the appropriate objective function for post-disaster humanitarian logistics models. *Journal of Operations Management*, 31(5), 262-280.

Wang, X., Wang, X., Liang, L., Yue, X., & Van Wassenhove, L. (2017). Estimation of deprivation level functions using a numerical rating scale. *Production and Operations Management*. DOI: 10.1111/poms.12760.

## 11. Implementation Notes

The current Python implementation in `alns_vrpfd/deprivation.py` already centralizes delay-cost calculation. The next code revision should avoid hard-coding separate Wang logistic and Holguin water branches in the evaluator. Instead, it should define a single class-parameter table:

```text
class -> beta_c, omega_c, deadline deltas
```

and implement:

```text
delay_cost = lambda * omega_c * normalized_or_direct_exponential(beta_c, rho, tau_hours)
```

This keeps ALNS, MIP PWL approximation, plotting scripts, and experiments consistent with one theoretical cost function family.
