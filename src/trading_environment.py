import gym
import numpy as np
import pandas as pd
from gym import spaces
from moving_average_crossover_strategy import moving_average as moving_average_jax, load_data

class TradingEnvironment(gym.Env):
    def __init__(self, csv_file_path, initial_balance=10000, transaction_fee=0.001):
        super(TradingEnvironment, self).__init__()

        # Load and preprocess data
        self.data = load_data(csv_file_path)
        self.prices = self.data
        self.returns = np.diff(np.log(self.prices))

        # Trading params
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee

        # RL params
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.short_window = 10
        self.long_window = 50

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def _get_observation(self):
        short_ma = moving_average_jax(self.prices[:self.current_step + 1], self.short_window)[-1]
        long_ma = moving_average_jax(self.prices[:self.current_step + 1], self.long_window)[-1]

        obs = np.array([
            self.balance,
            self.shares_held,
            short_ma if not np.isnan(short_ma) else self.prices[self.current_step],
            long_ma if not np.isnan(long_ma) else self.prices[self.current_step]
        ])
        return obs

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        done = self.current_step >= len(self.prices) - 1
        obs = self._get_observation()

        reward = self._calculate_reward()

        return obs, reward, done, {}

    def _take_action(self, action):
        current_price = self.prices[self.current_step]

        if action == 1:  # Buy
            shares_to_buy = self.balance // current_price
            cost = shares_to_buy * current_price * (1 + self.transaction_fee)
            self.balance -= cost
            self.shares_held += shares_to_buy
        elif action == 2:  # Sell
            sell_value = self.shares_held * current_price * (1 - self.transaction_fee)
            self.balance += sell_value
            self.shares_held = 0

    def _calculate_reward(self):
        return self.balance + self.shares_held * self.prices[self.current_step] - self.initial_balance

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        return self._get_observation()

    def render(self, mode='human'):
        # Render the environment to the screen
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Current price: {self.prices[self.current_step]}')
        print(f'Total value: {self.balance + self.shares_held * self.prices[self.current_step]}')
