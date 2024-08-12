import unittest
from unittest.mock import MagicMock
import numpy as np
import jax.numpy as jnp
import tensorflow as tf

from fetchai.ledger.crypto import Entity
from fetchai.ledger.contract import Contract
from fetchai.ledger.api import LedgerApi
from fetchai.ledger.api.token import TokenTxFactory
from src.trading_environment import TradingEnvironment
from moving_average_crossover_strategy import crossover_strategy_jax, crossover_strategy_tf
from trading_agent import TradingAgent  # Assume your code is in trading_agent.py

class TestTradingAgent(unittest.TestCase):
    def setUp(self):
        # Mock the Fetch.ai components
        self.entity = MagicMock(spec=Entity)
        self.ledger_api = MagicMock(spec=LedgerApi)
        self.contract = MagicMock(spec=Contract)
        self.agent = TradingAgent(self.entity, self.ledger_api, self.contract)

        # Mock TradingEnvironment
        self.agent.environment = MagicMock(spec=TradingEnvironment)
        self.agent.environment.prices = np.array([100, 105, 110, 115])
        self.agent.environment.current_step = 0
        self.agent.environment.balance = 1000
        self.agent.environment.shares_held = 0
        self.agent.environment.reset.return_value = (self.agent.environment.balance, self.agent.environment.shares_held, 100, 105)
        self.agent.environment.step.return_value = (self.agent.environment.balance, 0, True, {})

    def test_make_decision_buy_signal(self):
        # Mock the crossover strategies
        crossover_strategy_jax = MagicMock(return_value=jnp.array([1]))
        crossover_strategy_tf = MagicMock(return_value=tf.convert_to_tensor([1]))
        
        # Test buy decision
        action = self.agent.make_decision((1000, 0, 100, 105))
        self.assertEqual(action, 1)  # Buy action
        
    def test_make_decision_sell_signal(self):
        # Mock the crossover strategies
        crossover_strategy_jax = MagicMock(return_value=jnp.array([-1]))
        crossover_strategy_tf = MagicMock(return_value=tf.convert_to_tensor([-1]))
        
        # Test sell decision
        self.agent.environment.shares_held = 10
        action = self.agent.make_decision((1000, 10, 100, 105))
        self.assertEqual(action, 2)  # Sell action
        
    def test_execute_trade_buy(self):
        self.agent.environment.balance = 1000
        self.agent.environment.prices = np.array([100])
        self.agent.execute_trade(1)  # Buy action
        # Check if transfer was called with the correct parameters
        self.ledger_api.sync.assert_called_once()
        # Add more specific assertions if needed

    def test_execute_trade_sell(self):
        self.agent.environment.shares_held = 10
        self.agent.environment.prices = np.array([100])
        self.agent.execute_trade(2)  # Sell action
        # Check if transfer was called with the correct parameters
        self.ledger_api.sync.assert_called_once()
        # Add more specific assertions if needed

    def test_run(self):
        # Mock the methods in the TradingEnvironment
        self.agent.environment.step.return_value = (1000, 0, False, {})  # Run one step
        self.agent.run(num_episodes=1)
        self.agent.environment.reset.assert_called()
        self.agent.environment.step.assert_called()
        # Add more specific assertions if needed

if __name__ == "__main__":
    unittest.main()
