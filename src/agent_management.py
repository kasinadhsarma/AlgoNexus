from fetchai.ledger.crypto import Entity
from fetchai.ledger.contract import Contract
from fetchai.ledger.api import LedgerApi
from fetchai.ledger.api.token import TokenTxFactory

import numpy as np
from trading_environment import TradingEnvironment
from moving_average_crossover_strategy import crossover_strategy

class TradingAgent:
    def __init__(self, entity: Entity, ledger_api: LedgerApi, contract: Contract):
        self.entity = entity
        self.ledger_api = ledger_api
        self.contract = contract
        self.environment = TradingEnvironment('/home/ubuntu/AAPL_data_20240811.csv')
        self.short_window = 10
        self.long_window = 50

    def make_decision(self, observation):
        # Extract relevant information from the observation
        balance, shares_held, short_ma, long_ma = observation

        # Use the crossover strategy to generate a signal
        prices = np.array([short_ma, long_ma])  # Use the moving averages as a proxy for prices
        signal = crossover_strategy(prices, self.short_window, self.long_window)[-1]

        # Convert signal to action
        if signal == 1 and balance > 0:
            return 1  # Buy
        elif signal == -1 and shares_held > 0:
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
