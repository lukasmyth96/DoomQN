from abc import ABC, abstractmethod
import random

import numpy as np


class Policy(ABC):

    @abstractmethod
    def get_action_idx(self, q_values, episode, *args, **kwargs):
        """
        Implements policy to return action idx
        Parameters
        ----------
        q_values: np.ndarray
            of shape (num_actions,)
        episode: int
        args
        kwargs

        Returns
        -------
        action_idx: int
        """

        return 0


class EpsilonGreedy(Policy):

    def __init__(self, fixed_epsilon=None, initial_eps=None, min_eps=None, decay_factor=None):

        self.fixed_epsilon = fixed_epsilon

        self.initial_eps = initial_eps
        self.min_eps = min_eps
        self.decay_factor = decay_factor

    def get_action_idx(self, q_values, episode=0, *args, **kwargs):
        """
        Returns action_idx with highest predicted q-value with p=(1-epsilon) or a random action with p=epsilon
        """

        if self.fixed_epsilon:
            epsilon = self.fixed_epsilon
        else:
            epsilon = self.get_exploration_rate(episode)

        assert 0 <= epsilon <= 1
        if random.random() < epsilon:
            action_idx = random.randint(0, len(q_values)-1)
        else:
            action_idx = np.argmax(q_values)

        return action_idx

    def get_exploration_rate(self, episode):
        """
        get epsilon value as a function of the episode - decays exponentially
        Parameters
        ----------
        episode: int

        Returns
        -------
        epsilon: float
        """
        epsilon = max(self.min_eps, self.initial_eps * (self.decay_factor ** episode))
        return epsilon


class Greedy(Policy):
    """ Return action_idx with the highest predicted probability"""
    def get_action_idx(self, q_values, episode, *args, **kwargs):
        return np.argmax(q_values)

