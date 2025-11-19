# Known Issues and Limitations

**Last Updated**: 2025-11-16
**Status**: Critical problems identified - system not suitable for real trading

---

## Executive Summary

This document provides an honest assessment of NeuroVest's critical flaws. While the system demonstrates software engineering practices and serves as an educational project, **it contains fundamental problems that make it completely unsuitable for real-money trading**.

**Primary Verdict**: The combination of severe overfitting, inadequate validation, and unrealistic assumptions means this system would likely result in significant financial losses if deployed with actual capital.

---

## 🚨 CATASTROPHIC PROBLEMS (Disqualifying)

### 1. Severe Model Overfitting

**Issue**: Models report 96-97% accuracy

**Why This is Impossible**:
- No legitimate trading model achieves 96%+ prediction accuracy
- Renaissance Technologies (the most successful quant fund ever) doesn't claim such accuracy
- Professional quant funds typically aim for 51-55% accuracy with excellent risk management
- Financial markets are inherently noisy and unpredictable

**What This Indicates**:
- ✓ Training on future data (look-ahead bias)
- ✓ Testing on training data (data leakage)
- ✓ Cherry-picking parameters that fit historical data perfectly
- ✓ Not accounting for realistic transaction costs
- ✓ Overly complex models memorizing noise rather than learning patterns

**Expected Reality**:
- Accuracy will drop to 45-50% in live trading (worse than random)
- All backtested "profits" will evaporate
- Classic case of curve-fitting to historical data

**Impact**: 🔴 **CRITICAL** - Makes all performance metrics meaningless

---

### 2. Original -96% Maximum Drawdown

**Issue**: System originally reported -96.11% maximum drawdown before protection was added

**What This Means**:
- At some point during backtesting, the system lost 96% of its capital
- A $100,000 account would have dropped to $3,890
- Near-total wipeout of capital

**Why This Happened**:
- Stop losses weren't actually enforced in backtest code
- Gap risk not accounted for (overnight market crashes)
- Leverage compounded losses exponentially (3x leverage × -32% market move = -96% loss)
- System may have doubled down on losing positions
- No circuit breaker to halt trading during severe drawdowns

**Mathematical Impossibility**:
```
System claims:
- Stop Losses: 4% (stocks), 8% (crypto)
- Daily Loss Limit: 2% of portfolio
- Leverage: 1-2x (stocks), 1-3x (crypto)

Maximum possible loss with 4% stops:
- 10 positions × 4% each = 40% maximum loss
- With 2x leverage: 80% maximum loss
- Still doesn't explain -96% drawdown

Conclusion: Risk controls didn't actually work
```

**Current Status**: Added drawdown protection claiming to reduce this to -15%, but root cause (overfitting) remains unaddressed

**Impact**: 🔴 **CRITICAL** - Would result in complete account loss

---

### 3. No Proper Validation Methodology

**Missing Critical Elements**:

❌ **No walk-forward testing**
- Uses static backtest on one historical period
- Doesn't prove system adapts to changing market regimes
- Can't demonstrate out-of-sample performance

❌ **No out-of-sample testing**
- Likely trained and tested on the same data
- Classic beginner mistake in machine learning
- Results are meaningless without train/test separation

❌ **No Monte Carlo simulation**
- Can't assess probability distribution of outcomes
- Don't know if results are skill or random luck
- No confidence intervals on performance metrics

❌ **No regime analysis**
- Doesn't test performance in different market conditions:
  - Bear markets (2008, 2022)
  - High volatility (VIX > 30)
  - Low volatility (VIX < 15)
  - Crisis periods (2020 COVID crash)
  - Different interest rate environments

❌ **Sample size too small**
- Only 475 trades backtested
- Need 1000+ trades for statistical significance
- Could easily be random luck rather than systematic edge

**What Proper Validation Requires**:
1. Train/Test split (at least 70/30)
2. Walk-forward analysis (rolling windows)
3. Out-of-sample testing on unseen data
4. Cross-validation across different time periods
5. Stress testing on historical crashes
6. Monte Carlo simulation (1000+ iterations)
7. Statistical significance testing

**Impact**: 🔴 **CRITICAL** - Can't trust any reported metrics

---

### 4. Impossibly High Sharpe Ratio

**Issue**: Reported Sharpe Ratio of 7.47

**Reality Check**:
| Entity | Sharpe Ratio |
|--------|--------------|
| NeuroVest (claimed) | 7.47 |
| Warren Buffett | ~0.76 |
| Renaissance Medallion (best fund ever) | ~2.5 |
| Excellent hedge fund | 1.5-2.0 |

**Why This is Absurd**:
- Means you get 7.47 units of return per unit of risk
- Better than every hedge fund in recorded history
- Even Medallion only achieves ~2.5 with massive proprietary data and PhD quants
- **This number alone proves severe overfitting to historical data**

**The Contradiction**:
```
High Sharpe Ratio (7.47) means: Low volatility, consistent returns
High Max Drawdown (-96%) means: Extreme volatility, huge losses

These are contradictory!

Real explanation: Cherry-picked metrics from different time periods
                  or backtest methodology errors
```

**Impact**: 🔴 **CRITICAL** - Proves results are not reliable

---

## 💥 SEVERE PROBLEMS (Would Fail in Live Trading)

### 5. Missing Transaction Costs

**Unaccounted Expenses**:

**Stock Trading Costs**:
- Commission fees (even "$0 commission" has SEC fees, exchange fees)
- Bid-ask spread (typically 0.05-0.5% per trade)
- Slippage (market impact when executing orders)
- Market data fees
- SEC Section 31 fees ($0.0000278 per dollar sold)

**Crypto Trading Costs**:
- Exchange trading fees (0.1-0.5% per trade)
- Withdrawal fees (moving funds between exchanges)
- Funding rates (cost of holding leveraged positions overnight)
- Gas fees (for on-chain transactions)
- Network congestion fees

**Impact Calculation**:
```
475 backtested trades = ~950 buy/sell executions
Conservative cost estimate: 0.2% per trade
Total cost: 950 × 0.2% = 190% of capital over backtest period

Claimed: 21.22% annual return over ~15 years = +354% total
After realistic costs: Possibly NEGATIVE returns

Even at 0.1% per trade:
950 × 0.1% = 95% of profits gone
```

**Impact**: 🟠 **SEVERE** - Would eliminate most/all profits

---

### 6. No Live Trading Validation

**Problem**: All results are backtested simulations only
- Zero real-world performance data
- No paper trading results
- No live trading track record
- Explicitly states "For educational purposes only"

**Typical Progression**:
```
Stage 1 - Backtest:     96% accuracy, 21% returns
Stage 2 - Paper trading: 60% accuracy, 8% returns
Stage 3 - Live trading:  45% accuracy, -5% returns

Reason: Overfitting to historical data is revealed
```

**Problems Revealed Only in Live Trading**:
- Order execution delays and slippage
- Partial fills and rejections
- Market impact (your orders move the price)
- Psychological pressure of real money
- Broker outages and technical failures
- Unexpected market conditions
- News events and black swan events
- Correlation breakdown during stress

**Impact**: 🟠 **SEVERE** - Unknown real-world performance

---

### 7. Data Quality Issues

**Survivorship Bias**:
- System only includes currently traded assets
- Doesn't account for delisted stocks or dead cryptocurrency projects
- Thousands of stocks delisted over the years (bankruptcies, mergers)
- Hundreds of cryptocurrencies have gone to zero
- Testing only on "survivors" inflates historical returns significantly

**Look-ahead Bias**:
- Using future information to make past predictions
- Examples:
  - Using adjusted prices (incorporates future stock splits)
  - Using end-of-day data to predict intraday
  - Features calculated using future data points
- Would explain the impossible 96% accuracy

**Data Snooping Bias**:
- Testing hundreds of parameter combinations
- Selecting only the best-performing configuration
- That configuration worked by chance on historical data
- Won't work going forward (random luck doesn't repeat)

**Point-in-Time Issues**:
- Financial data gets revised after initial release
- Using final revised data in backtest (cheating)
- Real trading only has preliminary data

**Cumulative Impact**:
```
Claimed returns: 21.22% annualized
After correcting for survivorship bias: ~15%
After correcting for look-ahead bias: ~8%
After correcting for data snooping: ~5%
After transaction costs: ~0% or negative

This is why 95% of backtested strategies fail live.
```

**Impact**: 🟠 **SEVERE** - Results are artificially inflated

---

### 8. Leverage Misuse

**Reported Usage**: 1-2x (stocks), 1-3x (crypto)

**The Dangers**:
- Leverage amplifies both gains AND losses proportionally
- 3x leverage on a -32% market move = -96% portfolio loss
- Most retail traders blow up accounts due to leverage misuse
- Requires sophisticated risk management (which this system clearly lacks)
- Margin calls can force liquidation at worst possible time

**What Likely Happened**:
```
Scenario: Crypto market crash
1. System took 3x leveraged position in crypto
2. Crypto market dropped 33% (not unusual)
3. Position lost 99% (3x × 33% = 99%)
4. Account nearly wiped out
5. -96% drawdown achieved in single event
```

**Additional Problems**:
- Overnight funding costs compound daily
- Forced liquidations below certain equity levels
- Reduced margin during volatility
- Correlation risk (all leveraged positions lose simultaneously)

**Impact**: 🟠 **SEVERE** - Catastrophic loss potential

---

## 🔧 SERIOUS PROBLEMS (Professional Gaps)

### 9. Model Architecture Flaws

**Problem**: All models are tree-based (XGBoost, LightGBM, Random Forest)

**Why This Matters**:
- Ensemble models work when component models are DIVERSE
- All decision tree-based = highly correlated predictions
- Doesn't provide true diversification benefit
- Like buying 3 different S&P 500 index funds (not actual diversification)

**Missing Elements**:
- No feature engineering documentation
- No model retraining strategy
- No feature importance analysis
- No interpretability or explainability
- Black box system (dangerous for real money)

**What Proper Systems Need**:
1. Diverse model types (trees + linear + neural nets)
2. Different feature sets for each model
3. Different timeframes (short/medium/long)
4. Regular retraining (weekly/monthly)
5. Feature importance monitoring
6. Model performance decay detection
7. Automatic model retirement when performance degrades

**Impact**: 🟡 **SERIOUS** - System won't adapt to changing markets

---

### 10. Asset Allocation Flaws

**Claimed Correlation Analysis**:
- TLT (bonds): -0.25 correlation to SPY (supposed hedge)
- GLD (gold): 0.10 correlation (supposed diversifier)
- Crypto: 0.50 correlation to stocks (supposed independent alpha)

**Fatal Flaw**: Correlations are NOT stable
- Correlations change dramatically during market stress
- During crashes, "uncorrelated" assets all drop together
- This is called "correlation breakdown" or "tail risk"

**Historical Evidence**:
```
March 2020 COVID crash (1 month):
- S&P 500: -34%
- Gold: -12%
- Bitcoin: -50%
- Bonds (TLT): -10%

All "diversified" assets fell together!
Correlations went from 0.0-0.5 to 0.9+ during crisis.
```

**The 80/20 Allocation Problem**:
- Appears arbitrary (no theoretical justification)
- Likely chosen AFTER seeing backtest results (cherry-picking)
- No optimization framework described
- No sensitivity analysis
- Suggests overfitting to historical period

**Impact**: 🟡 **SERIOUS** - Diversification fails when needed most

---

### 11. No Market Microstructure Understanding

**Missing Real-World Constraints**:

**Order Execution Assumptions**:
- Assumes instant fills at desired price (impossible)
- Reality: Slippage occurs
- Reality: Partial fills
- Reality: Order rejections
- Large orders move the market against you

**Liquidity Constraints**:
- Doesn't check if enough trading volume exists
- PPLT, PALL (platinum, palladium ETFs) are extremely illiquid
- Example: PPLT average volume ~50,000 shares/day
- A $10,000 order might be 2% of daily volume (huge market impact)

**Market Hours Mismatch**:
- Crypto trades 24/7
- Stocks only trade 6.5 hours/day
- Can't simultaneously enter/exit stock and crypto positions
- After-hours stock trading is extremely illiquid

**Overnight Risk**:
- Markets can gap up or down overnight
- Stop losses DON'T protect against gaps
- Stock opens 10% lower → stop at -4% never executes → immediate -10% loss
- This explains how -96% drawdown occurred despite "4% stops"

**Example Failure**:
```
Backtest assumption:
- See signal to sell PPLT at $70.00
- Assume instant fill at $70.00

Reality:
- Submit sell order
- Bid-ask spread is $69.50 / $70.50
- Sell hits bid at $69.50 (-0.7% slippage)
- Only partially filled (5,000 of 10,000 shares)
- Price drops to $69.00 while waiting
- Eventually fill at average of $69.25
- Total slippage: -1.1%

Do this across 475 trades = massive profit reduction
```

**Impact**: 🟡 **SERIOUS** - Real performance much worse than backtest

---

## 🔧 TECHNICAL DEBT PROBLEMS

### 12. No Testing Infrastructure

**Complete Absence of Software Testing**:
- ❌ No unit tests
- ❌ No integration tests
- ❌ No validation tests
- ❌ No regression tests
- ❌ No performance tests
- ❌ No edge case tests

**Impact of Missing Tests**:
- Can't verify code works correctly
- Changes could break everything silently
- No confidence in system reliability
- Bugs could cause catastrophic trading losses
- Can't safely refactor or improve code

**Examples of Undetected Bugs**:
```python
# Bug 1: Off-by-one error in position sizing
position_size = account_value * 0.1  # Should be 0.01
# Result: 10x larger positions, 10x the risk

# Bug 2: Logic error in stop loss
if price < stop_loss:
    buy()  # Should be sell()!
# Result: Buy more when losing, catastrophic losses

# Bug 3: Data type error
leverage = "3"  # String instead of integer
position = capital * leverage  # Results in string concatenation
# Result: System crashes or bizarre behavior
```

**Impact**: 🟡 **SERIOUS** - Code reliability unknown

---

### 13. No Production Infrastructure

**Missing Production Readiness**:
- ❌ No deployment strategy
- ❌ No monitoring/alerting
- ❌ No error handling
- ❌ No logging
- ❌ No database backups
- ❌ No failover/redundancy

**Real-World Failure Scenario**:
```
Day 1, 2:00 AM:
- System crashes due to unhandled API error
- No monitoring, so you don't know
- Market moves against open positions
- Stop losses don't execute (system is down)
- Wake up at 9 AM to see -15% loss

Day 3, 3:30 PM:
- Database file corrupts (SQLite not enterprise-grade)
- Lose all position tracking data
- Don't know what you own anymore
- Can't close positions safely

Result: Thousands of dollars lost due to infrastructure failures
```

**Impact**: 🟡 **SERIOUS** - System can't run reliably 24/7

---

### 14. Security & Compliance Issues

**Critical Security Vulnerabilities**:
- ❌ No API key management (likely hardcoded or plaintext)
- ❌ No authentication/authorization
- ❌ No encryption
- One security breach = attacker drains your account

**Regulatory Compliance Problems**:
- ❌ Pattern Day Trader rules not enforced
- ❌ Wash sale rules not tracked
- ❌ Margin requirements not monitored
- ❌ No audit trail
- Could result in IRS penalties or SEC investigation

**Worst-Case Scenario**:
```
1. API keys leaked in GitHub repository
2. Attacker finds keys, accesses broker account
3. Attacker executes unauthorized trades
4. Drains account with large losing positions
5. You're liable for all losses
6. Broker won't reimburse (your keys, your fault)
7. Potentially face SEC investigation
```

**Impact**: 🟡 **SERIOUS** - Legal and financial liability

---

## Priority-Ranked Summary

### CRITICAL (System is completely unusable):
1. ⛔ Severe model overfitting (96-97% accuracy impossible)
2. ⛔ Original -96% maximum drawdown
3. ⛔ No proper validation methodology
4. ⛔ Impossibly high Sharpe ratio (7.47)

### SEVERE (Would fail immediately in live trading):
5. 🔴 No transaction costs modeled
6. 🔴 No live trading validation whatsoever
7. 🔴 Data quality issues (survivorship, look-ahead, snooping bias)
8. 🔴 Leverage misuse

### SERIOUS (Professional gaps):
9. 🟠 Model architecture flaws
10. 🟠 Asset allocation flaws
11. 🟠 No market microstructure understanding

### IMPORTANT (Technical debt):
12. 🟡 No testing infrastructure
13. 🟡 No production infrastructure
14. 🟡 Security & compliance issues

---

## The Bottom Line

**This system has ONE fatal flaw that makes all others irrelevant:**

> The combination of severe overfitting (96-97% accuracy), inadequate validation, and the original -96% drawdown means that **if you use this system with real money, you will almost certainly lose most or all of your capital.**

**Even if every infrastructure problem was fixed, the fundamental strategy is not viable because it's overfitted to historical data and has never been validated in real markets.**

---

## What This Project Actually Demonstrates

**Positive Aspects**:
- ✅ Good software organization and documentation
- ✅ Understanding of trading concepts
- ✅ Ability to implement complex systems
- ✅ Integration of multiple technologies
- ✅ Clear code structure

**Educational Value**:
- ✅ Demonstrates what NOT to do in algorithmic trading
- ✅ Shows importance of proper validation
- ✅ Illustrates common pitfalls (overfitting, unrealistic assumptions)
- ✅ Provides foundation for learning

**NOT Demonstrated**:
- ❌ A profitable trading system
- ❌ Proper quantitative research methodology
- ❌ Real-world trading viability
- ❌ Statistical rigor

---

## Path Forward

### If Continuing as Educational Project:

1. **Keep as-is** with honest disclaimers ✅ (Done)
2. **Add this documentation** to show self-awareness ✅ (Done)
3. **Create LESSONS_LEARNED.md** documenting what you learned
4. **Consider building V2** with proper methodology:
   - Simple strategy (moving average crossover)
   - Proper train/test split
   - Transaction costs included
   - Realistic expectations (51-55% accuracy, Sharpe 0.5-1.5)
   - Walk-forward validation
   - 6-12 months paper trading before any real capital

### If Pursuing Real Trading (NOT Recommended):

1. Complete rebuild required - current system is not salvageable
2. Proper quantitative research methodology
3. 6-12 months minimum paper trading
4. Start with $1-10k maximum after extensive validation
5. Consult with licensed financial professionals
6. Accept that most retail algorithmic traders lose money

---

## Conclusion

This is a well-executed **learning project** that demonstrates technical skills but contains fundamental flaws making it unsuitable for real trading. Use it to learn, build portfolio credibility, and understand algorithmic trading concepts.

**Never use it with real money.**

---

**Document End**
