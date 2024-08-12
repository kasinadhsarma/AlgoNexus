from fetchai.ledger.crypto import Entity
from fetchai.ledger.contract import Contract
from fetchai.ledger.api import LedgerApi
from fetchai.ledger.api.token import TokenTxFactory

import numpy as np
import jax.numpy as jnp
import tensorflow as tf
from src.trading_environment import TradingEnvironment
from moving_average_crossover_strategy import crossover_strategy_jax, crossover_strategy_tf

class TradingAgent:
    def __init__(self, entity: Entity, ledger_api: LedgerApi, contract: Contract):
        self.entity = entity
        self.ledger_api = ledger_api
        self.contract = contract
        self.environment = TradingEnvironment('data/AAPL_data_20240811.csv')
        self.short_window = 10
        self.long_window = 50

    def make_decision(self, observation):
        # Extract relevant information from the observation
        balance, shares_held, short_ma, long_ma = observation

        # Use both JAX and TensorFlow crossover strategies to generate signals
        price = self.environment.prices[self.environment.current_step]
        prices_jax = jnp.array([price])
        prices_tf = tf.constant([price])

        signal_jax = crossover_strategy_jax(prices_jax, self.short_window, self.long_window)[-1]
        signal_tf = crossover_strategy_tf(prices_tf, self.short_window, self.long_window)[-1].numpy()

        # Combine signals (e.g., take the average)
        combined_signal = (signal_jax + signal_tf) / 2

        # Convert combined signal to action
        if combined_signal > 0.5 and balance > 0:
            return 1  # Buy
        elif combined_signal < -0.5 and shares_held > 0:
            return 2  # Sell
        else:
            return 0  # Hold

    def execute_trade(self, action):
        # Execute the trade on the Fetch.ai ledger
        current_price = self.environment.prices[self.environment.current_step]
        if action == 1:  # Buy
            shares_to_buy = int(self.environment.balance // current_price)
            if shares_to_buy > 0:
                tx = TokenTxFactory.transfer(
                    self.entity,
                    self.contract.address,
                    amount=shares_to_buy * current_price,
                    fee=100000,
                )
                tx.sign(self.entity)
                self.ledger_api.sync(tx)
                print(f"Bought {shares_to_buy} shares at {current_price}")
        elif action == 2:  # Sell
            if self.environment.shares_held > 0:
                tx = TokenTxFactory.transfer(
                    self.contract.address,
                    self.entity.address,
                    amount=self.environment.shares_held * current_price,
                    fee=100000,
                )
                tx.sign(self.entity)
                self.ledger_api.sync(tx)
                print(f"Sold {self.environment.shares_held} shares at {current_price}")

    def run(self, num_episodes=1000):
        for episode in range(num_episodes):
            observation = self.environment.reset()
            done = False
            while not done:
                action = self.make_decision(observation)
                self.execute_trade(action)
                observation, reward, done, _ = self.environment.step(action)

def setup_fetch_ai():
    # Set up Fetch.ai ledger connection
    entity = Entity()
    ledger_api = LedgerApi('testnet')
    contract = Contract(...)  # Load your smart contract here
    return entity, ledger_api, contract

def main():
    entity, ledger_api, contract = setup_fetch_ai()
    agent = TradingAgent(entity, ledger_api, contract)
    agent.run()


if __name__ == "__main__":
    main()
