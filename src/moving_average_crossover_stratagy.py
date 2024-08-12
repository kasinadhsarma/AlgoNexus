import jax.numpy as jnp
from jax import jit, lax
import pandas as pd
import matplotlib.pyplot as plt

MAX_WINDOW_SIZE = 200  # Maximum expected window size

@jit
def moving_average(data, window_size):
    cumsum = jnp.cumsum(jnp.pad(data, (MAX_WINDOW_SIZE - 1, 0), mode='edge'))
    indices = jnp.arange(data.shape[0])

    def body_fun(i, result):
        start = lax.dynamic_slice(cumsum, (i,), (1,))
        end = lax.dynamic_slice(cumsum, (i + window_size,), (1,))
        return result.at[i].set((end[0] - start[0]) / window_size)

    result = jnp.zeros(data.shape[0])
    return lax.fori_loop(0, data.shape[0], body_fun, result)

@jit
def crossover_strategy(prices, short_window, long_window):
    short_ma = moving_average(prices, short_window)
    long_ma = moving_average(prices, long_window)

    def body_fun(i, signals):
        signal = jnp.where(short_ma[i] > long_ma[i], 1, -1)
        return signals.at[i].set(signal)

    signals = jnp.zeros(prices.shape[0])
    return lax.fori_loop(0, prices.shape[0], body_fun, signals)

@jit
def backtest_strategy(prices, signals):
    returns = jnp.diff(jnp.log(prices))
    strategy_returns = signals[:-1] * returns
    cumulative_returns = jnp.cumsum(strategy_returns)
    return cumulative_returns

def load_data(filename):
    df = pd.read_csv(filename)
    return jnp.array(df['Close'].values)

def plot_results(prices, signals, cumulative_returns):
    plt.figure(figsize=(12, 8))
    plt.plot(prices, label='Price')
    plt.plot(jnp.where(signals == 1, prices, jnp.nan), '^', markersize=10, color='g', label='Buy Signal')
    plt.plot(jnp.where(signals == -1, prices, jnp.nan), 'v', markersize=10, color='r', label='Sell Signal')
    plt.title('Moving Average Crossover Strategy')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('ma_crossover_signals.png')

    plt.figure(figsize=(12, 8))
    plt.plot(cumulative_returns, label='Strategy Returns')
    plt.title('Cumulative Returns of Moving Average Crossover Strategy')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.savefig('ma_crossover_returns.png')

def main():
    # Load data
    prices = load_data('/home/ubuntu/AAPL_data_20240811.csv')

    # Set parameters
    short_window = 10
    long_window = 50

    # Generate signals
    signals = crossover_strategy(prices, short_window, long_window)

    # Backtest strategy
    cumulative_returns = backtest_strategy(prices, signals)

    # Plot results
    plot_results(prices, signals, cumulative_returns)

    print(f"Final cumulative return: {cumulative_returns[-1]:.2f}")

if __name__ == "__main__":
    main()

