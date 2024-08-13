import unittest
from unittest.mock import patch, MagicMock
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from src.moving_average_crossover_stratagy import moving_average, crossover_strategy, backtest_strategy, load_data, plot_results, main  # Replace 'your_module' with the actual module name

class TestTradingStrategies(unittest.TestCase):
    
    def setUp(self):
        # Mock data
        self.prices = jnp.array([100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150], dtype=jnp.float32)
        self.short_window = 3
        self.long_window = 5

    def test_moving_average(self):
        result = moving_average(self.prices, self.short_window)
        expected_result = jnp.array([105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0])
        jnp.testing.assert_allclose(result, expected_result, atol=1e-6)

    def test_crossover_strategy(self):
        signals = crossover_strategy(self.prices, self.short_window, self.long_window)
        expected_signals = jnp.array([-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1])
        jnp.testing.assert_allclose(signals, expected_signals, atol=1e-6)

    def test_backtest_strategy(self):
        signals = crossover_strategy(self.prices, self.short_window, self.long_window)
        cumulative_returns = backtest_strategy(self.prices, signals)
        expected_returns = jnp.cumsum(signals[:-1] * jnp.diff(jnp.log(self.prices)))
        jnp.testing.assert_allclose(cumulative_returns, expected_returns, atol=1e-6)

    @patch('pandas.read_csv')
    def test_load_data(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({'Close': [100, 105, 110, 115, 120]})
        prices = load_data('dummy_path.csv')
        expected_prices = jnp.array([100, 105, 110, 115, 120], dtype=jnp.float32)
        jnp.testing.assert_allclose(prices, expected_prices, atol=1e-6)

    @patch('matplotlib.pyplot.savefig')
    def test_plot_results(self, mock_savefig):
        # Mock data
        signals = jnp.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
        cumulative_returns = jnp.array([0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0])
        
        # Run plot_results function
        plot_results(self.prices, signals, cumulative_returns)
        
        # Check if savefig was called correctly
        mock_savefig.assert_called()
        self.assertEqual(mock_savefig.call_count, 2)

    @patch('your_module.load_data', return_value=jnp.array([100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150], dtype=jnp.float32))
    @patch('your_module.plot_results')
    def test_main(self, mock_plot_results, mock_load_data):
        with patch('builtins.print') as mock_print:
            main()
            mock_load_data.assert_called_with('/home/ubuntu/AAPL_data_20240811.csv')
            mock_plot_results.assert_called()
            mock_print.assert_called_with(f"Final cumulative return: {jnp.cumsum(crossover_strategy(self.prices, self.short_window, self.long_window)[:-1] * jnp.diff(jnp.log(self.prices)))[-1]:.2f}")

if __name__ == "__main__":
    unittest.main()

