import sys
sys.path.append('/home/ubuntu/AlgoNexus/src')
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import gym
from agent_management import TradingEnvironment
from moving_average_crossover_strategy import moving_average_jax, load_data

class TestTradingEnvironment(unittest.TestCase):

    def setUp(self):
        # Mock data
        self.mock_prices = np.array([100, 105, 110, 115, 120, 125, 130], dtype=np.float32)
        
        # Mock the load_data function
        self.mock_load_data = MagicMock(return_value=self.mock_prices)
        
        # Initialize the environment
        self.env = TradingEnvironment(csv_file_path='dummy_path.csv')

    @patch('moving_average_crossover_strategy.load_data')
    def test_initialization(self, mock_load_data):
        mock_load_data.return_value = self.mock_prices
        self.assertEqual(self.env.initial_balance, 10000)
        self.assertEqual(self.env.transaction_fee, 0.001)
        self.assertEqual(self.env.action_space.n, 3)
        self.assertEqual(self.env.observation_space.shape, (4,))
        self.assertTrue(np.all(self.env.prices == self.mock_prices))

    @patch('moving_average_crossover_strategy.moving_average_jax')
    def test_step_buy(self, mock_moving_average):
        mock_moving_average.return_value = np.array([100, 105, 110, 115, 120, 125, 130])
        
        obs = self.env.reset()
        self.env.step(1)  # Buy action
        
        self.assertEqual(self.env.balance, 10000 - 100 * (1 + self.env.transaction_fee))
        self.assertEqual(self.env.shares_held, 10000 // 100)
        self.assertEqual(self.env.current_step, 1)

    @patch('moving_average_crossover_strategy.moving_average_jax')
    def test_step_sell(self, mock_moving_average):
        mock_moving_average.return_value = np.array([100, 105, 110, 115, 120, 125, 130])
        
        # Simulate buying first
        self.env.step(1)  # Buy action
        self.env.step(2)  # Sell action
        
        self.assertEqual(self.env.balance, (10000 // 100) * 100 * (1 - self.env.transaction_fee))
        self.assertEqual(self.env.shares_held, 0)
        self.assertEqual(self.env.current_step, 2)

    def test_calculate_reward(self):
        self.env.reset()
        self.env.step(1)  # Buy action
        self.env.step(2)  # Sell action
        reward = self.env._calculate_reward()
        
        expected_reward = (10000 // 100) * 100 * (1 - self.env.transaction_fee) - 10000
        self.assertAlmostEqual(reward, expected_reward)

    def test_reset(self):
        self.env.step(1)  # Perform some actions
        obs = self.env.reset()
        
        self.assertEqual(self.env.balance, 10000)
        self.assertEqual(self.env.shares_held, 0)
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(obs[0], 10000)
        self.assertEqual(obs[1], 0)

    @patch('builtins.print')
    def test_render(self, mock_print):
        self.env.reset()
        self.env.render()
        mock_print.assert_called_with(
            f'Step: {self.env.current_step}\nBalance: {self.env.balance}\nShares held: {self.env.shares_held}\nCurrent price: {self.env.prices[self.env.current_step]}\nTotal value: {self.env.balance + self.env.shares_held * self.env.prices[self.env.current_step]}'
        )

if __name__ == "__main__":
    unittest.main()
