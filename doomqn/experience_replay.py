from collections import deque
from random import sample

import numpy as np


class Transition(object):
    def __init__(self, preprocessed_curr, action_idx, reward, preprocessed_next, is_terminal):
        """

        Parameters
        ----------
        preprocessed_curr: np.ndarray
        action_idx: int
        reward: float
        preprocessed_next: np.ndarray
        is_terminal: bool
        """
        self.preprocessed_curr = preprocessed_curr
        self.action_idx = action_idx
        self.reward = reward
        self.preprocessed_next = preprocessed_next
        self.is_terminal = is_terminal


class ReplayBuffer():
    """
    Stores memory in terms of (current_s, next_s, current_a, current_reward, is_terminal)
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque([], maxlen=capacity)

    @property
    def size(self):
        return len(self.buffer)

    def append(self, transition):
        """
        add transition object to buffer
        Parameters
        ----------
        transition: _vizdoom.experience_replay.Transition
        """
        self.buffer.append(transition)

    def get_minibatch(self, batch_size, prioritized=False):
        """
        returns minibatch of desired size
        Parameters
        ----------
        batch_size: int
        prioritized: bool
            if True - samples are chosen with probability proportional to their reward

        Returns
        -------
        s_curr: np.ndarray
        action_indices: np.ndarray
        s_next: np.ndarray
        is_terminal: np.ndarray
        r: np.ndarray
        """
        if prioritized:
            rewards = [tran.reward for tran in self.buffer]
            if min(rewards) < 0:
                rewards += (np.abs(min(rewards)) + 1e-10)  # scale so probs are non-negative
            probs = np.array(rewards) / sum(rewards)
            indices = np.random.choice(list(range(self.size)), batch_size, p=probs)
        else:
            indices = sample(list(range(self.size)), batch_size)

        transitions = [self.buffer[idx] for idx in indices]
        s_curr = np.array([tran.preprocessed_curr for tran in transitions])
        action_indices = np.array([tran.action_idx for tran in transitions])
        s_next = np.array([tran.preprocessed_next for tran in transitions])
        r = np.array([tran.reward for tran in transitions])
        is_terminal = np.array([tran.is_terminal for tran in transitions])

        return s_curr, action_indices, s_next, r, is_terminal


