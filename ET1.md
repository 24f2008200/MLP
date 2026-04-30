## Alternative Time Series Decomposition Models

---

### 1. **Classical Additive Decomposition Model**
$$Y_t = T_t + S_t + E_t$$

- Seasonal effect is **constant** regardless of the trend level
- Best when fluctuations don't grow with the series
- **Use when:** Data has stable, uniform seasonal swings

---

### 2. **STL Decomposition**
*(Seasonal and Trend decomposition using Loess)*
$$Y_t = T_t + S_t + R_t$$

- Uses **locally weighted regression (Loess)** to estimate components
- Highly flexible — seasonal component can **change over time**
- Robust to **outliers**
- Only handles additive form (log-transform for multiplicative)
- **Use when:** Seasonality evolves over time, or data has outliers

---

### 3. **X-11 / X-12-ARIMA / X-13-ARIMA-SEATS**

- Developed by the **US Census Bureau**
- Iteratively applies **moving averages** to extract components
- X-13 adds **ARIMA pre-adjustment** for outliers, trading days, holidays
- Industry standard for **economic & government data**
- **Use when:** Official statistical reporting, economic indicators

---

### 4. **SEATS Decomposition**
*(Signal Extraction in ARIMA Time Series)*

- Based on fitting an **ARIMA model** to the series first
- Then extracts trend, seasonal, irregular components from the ARIMA structure
- Often combined with X-13 as **X-13-ARIMA-SEATS**
- **Use when:** You want a model-based statistical decomposition

---

### 5. **ETS Models**
*(Error, Trend, Seasonality)*

$$\text{e.g., } Y_t = (\ell_{t-1} + b_{t-1}) \cdot S_{t-m} \cdot \varepsilon_t$$

- A family of **exponential smoothing** state space models
- Components can each be **additive, multiplicative, or absent**
- 30 possible combinations (e.g., ETS(A,A,A), ETS(M,M,M))
- Parameters estimated by **maximum likelihood**
- **Use when:** Forecasting is the primary goal

| Code | Error | Trend | Season |
|------|-------|-------|--------|
| ETS(A,A,A) | Add | Add | Add |
| ETS(M,M,M) | Mult | Mult | Mult |
| ETS(M,Ad,M) | Mult | Damped | Mult |

---

### 6. **TBATS Model**
*(Trigonometric, Box-Cox, ARMA, Trend, Seasonal)*

$$Y_t^{(\omega)} = \ell_{t-1} + \phi b_{t-1} + \sum s_t^{(i)} + d_t$$

- Handles **multiple seasonalities** simultaneously (e.g., hourly data with daily + weekly + annual cycles)
- Applies **Box-Cox transformation** automatically
- Uses **Fourier terms** for seasonal representation
- **Use when:** Complex or multiple overlapping seasonal patterns (electricity demand, traffic)

---

### 7. **Prophet (by Meta/Facebook)**

$$y(t) = g(t) + s(t) + h(t) + \varepsilon_t$$

| Term | Meaning |
|------|---------|
| $g(t)$ | Trend (linear or logistic growth) |
| $s(t)$ | Seasonality (Fourier series) |
| $h(t)$ | Holiday effects |
| $\varepsilon_t$ | Error |

- Designed for **business time series** with strong seasonality
- Handles **missing data**, outliers, and **changepoints** automatically
- Very easy to use with minimal tuning
- **Use when:** Business forecasting with holidays and irregular events

---

### 8. **State Space Models / Structural Time Series**

$$\text{Observation: } Y_t = Z_t \alpha_t + \varepsilon_t$$
$$\text{State: } \alpha_{t+1} = T_t \alpha_t + \eta_t$$

- Components modeled as **unobserved latent states**
- Estimated using the **Kalman Filter**
- Very flexible — can model time-varying trends and seasonality
- **Use when:** You need dynamic, evolving components with uncertainty estimates

---

### 9. **Wavelet Decomposition**

- Decomposes series across **multiple time-frequency scales**
- Captures both **time and frequency** information simultaneously
- Useful for **non-stationary** signals
- **Use when:** Signal processing, financial data, irregular patterns at multiple scales

---

## Quick Comparison Summary

| Model | Handles Changing Seasonality | Multiple Seasonalities | Robust to Outliers | Best For |
|-------|:---:|:---:|:---:|---------|
| Classical Additive/Multiplicative | ❌ | ❌ | ❌ | Simple, textbook use |
| STL | ✅ | ❌ | ✅ | Evolving seasonality |
| X-13-ARIMA | ✅ | ❌ | ✅ | Economic/govt data |
| ETS | ✅ | ❌ | ❌ | General forecasting |
| TBATS | ✅ | ✅ | ❌ | Complex seasonality |
| Prophet | ✅ | ✅ | ✅ | Business forecasting |
| State Space | ✅ | ✅ | ✅ | Dynamic modelling |
| Wavelet | ✅ | ✅ | ❌ | Signal processing |
