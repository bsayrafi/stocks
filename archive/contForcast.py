import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Fetch Market Data
# Downloading 4 years of S&P 500 ETF data
print("Fetching data...")
data = yf.download('SPY', start='2020-01-01', end='2024-01-01')
df = data[['Close']].copy()

# 2. Define the Ensemble
# We will create 16 pairs where the slow MA is 5x the fast MA
# e.g., (10, 50), (12, 60) ... up to (40, 200)
pairs = [(f, f * 5) for f in range(10, 41, 2)]
num_pairs = len(pairs)

# 3. Calculate Votes for Every Pair
votes = pd.DataFrame(index=df.index)

for fast, slow in pairs:
    fast_ma = df['Close'].rolling(window=fast).mean()
    slow_ma = df['Close'].rolling(window=slow).mean()
    
    # Vote +1 if Fast > Slow (Bullish), -1 if Fast < Slow (Bearish)
    votes[f'{fast}_{slow}'] = np.where(fast_ma > slow_ma, 1, -1)

# 4. Generate the Continuous Forecast Signal
# Sum all votes and divide by total pairs to normalize between -1.0 and 1.0
df['Continuous_Signal'] = votes.sum(axis=1) / num_pairs

# 5. Proportional Position Sizing & Backtest
df['Market_Returns'] = df['Close'].pct_change()

# CRITICAL: Shift the signal by 1 day to prevent look-ahead bias
# You trade today based on yesterday's closing signal
df['Allocated_Position'] = df['Continuous_Signal'].shift(1)

# Multiply daily market return by our fractional position size
df['Strategy_Returns'] = df['Allocated_Position'] * df['Market_Returns']

# Calculate cumulative returns
df['Cumulative_Market'] = (1 + df['Market_Returns']).cumprod()
df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()

# 6. Visualize the Results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

# Top Chart: Performance
ax1.plot(df.index, df['Cumulative_Market'], label='Buy & Hold SPY', color='gray')
ax1.plot(df.index, df['Cumulative_Strategy'], label='Continuous Ensemble Strategy', color='blue')
ax1.set_title('Strategy Performance vs Buy & Hold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom Chart: Continuous Signal (Position Size)
ax2.plot(df.index, df['Allocated_Position'], color='purple', label='Position Size (-1.0 to 1.0)')
ax2.fill_between(df.index, df['Allocated_Position'], 0, where=(df['Allocated_Position'] >= 0), color='green', alpha=0.3)
ax2.fill_between(df.index, df['Allocated_Position'], 0, where=(df['Allocated_Position'] < 0), color='red', alpha=0.3)
ax2.set_title('Continuous Forecasting Signal')
ax2.set_ylabel('Exposure')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()