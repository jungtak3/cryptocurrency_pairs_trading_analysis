# Cryptocurrency Pairs Trading Analysis: Methodology Comparison and Issues

## Research Paper Methodology vs Implementation Analysis

### Paper Overview (Saji T.G, 2021)
- **Title**: "Pairs trading in cryptocurrency market: A long-short story"
- **Data**: 4 cryptocurrencies (BTC, ETH, LTC, NEO) from Jan 1, 2018 to Dec 31, 2019
- **Methods**: Correlation analysis, distance approach, stochastic differential, cointegration analysis
- **Approach**: Out-of-sample formation with 6-month panels

## Major Methodology Discrepancies Identified

### ✅ CORRECTLY IMPLEMENTED

1. **Out-of-sample Formation Structure**
   - Paper: Form pairs in one period, trade in next 6-month period
   - Code: ✅ Correctly implements this with `formation_panel_map`

2. **Cointegration Analysis**
   - Paper: Uses Engle-Granger two-step procedure
   - Code: ✅ Implements ADF tests, cointegration regression, residual testing

3. **Hedge Ratio Calculation**
   - Paper: Uses cointegration coefficient as optimal hedge ratio
   - Code: ✅ Uses beta from cointegration regression

4. **Basic Statistical Tests**
   - Paper: ADF tests, correlation, distance measures
   - Code: ✅ All implemented correctly

### ❌ CRITICAL ISSUES IDENTIFIED

#### 1. **Missing Mean Reversion Trading Logic**
**Paper Says:**
> "Pairs trader will look for two assets with a high positive correlation, wait for a divergence in their prices, and then trade on the expectation that the assets will revert to their historic correlation."

**Code Reality:**
- ❌ No divergence detection mechanism
- ❌ No convergence trigger for position entry/exit
- ❌ Simply holds position for entire period regardless of price relationship

#### 2. **Oversimplified Trading Strategy**
**Paper Says:**
> "The trader will go long on the relatively underpriced asset and make a short on the relatively overpriced asset; a profit may be made by relaxing his position upon the convergence of the spread"

**Code Reality:**
- ❌ Always goes long dependent variable, short independent variable
- ❌ No assessment of which asset is over/underpriced
- ❌ No spread-based entry/exit signals

#### 3. **Incomplete Stochastic Differential Method**
**Paper Says:**
> Implements stochastic differential residuals approach using CAPM/APT models

**Code Reality:**
- ❌ Shows "Not Impl." for stochastic differential tests
- ❌ Missing benchmark return calculation for beta estimation
- ❌ No implementation of equation (4) from paper

#### 4. **Simplified Risk Calculation**
**Paper Methodology:**
> Should compute market betas using weighted average of top 10 cryptocurrencies as benchmark

**Code Reality:**
- ❌ Uses simple standard deviation of daily profits
- ❌ No benchmark index construction
- ❌ No market beta calculation for individual assets

#### 5. **Trading Execution Differences**
**Paper Says:**
> "Day trading" with opening/closing prices, but with mean reversion signals

**Code Reality:**
- ✅ Uses open/close prices correctly
- ❌ No mean reversion entry/exit logic
- ❌ Holds fixed position entire period

## Why Numbers Could Be Off

### 1. **No Mean Reversion Signals**
- **Impact**: Massive profit/loss differences
- **Issue**: Code takes position immediately and holds, rather than waiting for optimal entry points
- **Paper expectation**: Only trade when spread diverges beyond threshold

### 2. **Missing Market-Neutral Component**
- **Impact**: Higher risk exposure than intended
- **Issue**: No proper market beta hedging
- **Paper expectation**: Market-neutral returns independent of overall market direction

### 3. **Incorrect Position Sizing**
- **Impact**: Suboptimal risk-return profile
- **Issue**: Not using minimum variance hedge ratio correctly
- **Paper expectation**: Positions sized to minimize portfolio variance

### 4. **No Transaction Cost Consideration**
- **Impact**: Overstated profits
- **Issue**: Paper likely assumes some transaction costs in realistic scenarios
- **Paper expectation**: Net profits after trading costs

### 5. **Data Frequency Mismatch**
- **Impact**: Different profit volatility patterns
- **Issue**: Paper uses daily data but code might not align properly with paper's specific timing
- **Paper expectation**: Consistent 6-month formation/trading periods

## Specific Code Improvements Needed

### 1. Implement Proper Spread-Based Trading
```python
# Calculate spread
spread = price1 - beta * price2
spread_zscore = (spread - spread.mean()) / spread.std()

# Entry signals
long_entry = spread_zscore < -2.0  # Spread too low, expect reversion up
short_entry = spread_zscore > 2.0   # Spread too high, expect reversion down

# Exit signals  
exit_signal = abs(spread_zscore) < 0.5  # Spread returned to normal
```

### 2. Add Market Beta Calculation
```python
# Create benchmark index (weighted avg of top cryptos)
benchmark_returns = create_crypto_benchmark_index()

# Calculate individual betas
for crypto in cryptos:
    beta = calculate_market_beta(crypto_returns, benchmark_returns)
```

### 3. Implement Stochastic Differential Method
```python
# Calculate stochastic differential returns per equation (4)
stoch_diff = return_x - return_y - delta * market_return
```

## Expected Impact on Results

### If Properly Implemented:
1. **Lower absolute profits** (more selective trading)
2. **Better risk-adjusted returns** (proper market-neutral positioning)
3. **More consistent performance** across market conditions
4. **Results closer to paper's Table 6** values

The current implementation appears to be a simplified approximation that captures the basic cointegration concept but misses the sophisticated mean reversion trading strategy that makes pairs trading profitable and market-neutral.