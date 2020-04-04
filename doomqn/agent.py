import os
import numpy as np

from keras.models import load_model

from doomqn.utils.data_preprocess import preprocess
from doomqn.q_model import QModel
import random


class DQNAgent():
    def __init__(self, config):
        self.config = config
        self.input_shape = config.INPUT_SHAPE
        self.actions = config.ACTIONS
        self.num_actions = len(self.actions)
        self.double = config.USE_DOUBLE_DQN
        self.discount_factor = config.DISCOUNT_FACTOR



        if self.double:
            self.model_1 = QModel(input_shape=self.input_shape, num_actions=self.num_actions)
            self.model_2 = QModel(input_shape=self.input_shape, num_actions=self.num_actions)
        else:
            self.model = QModel(input_shape=self.input_shape, num_actions=self.num_actions)

    def select_action(self, game_state, episode, policy):
        """
        Select and return action based on trained_models and exploration policy
        Parameters
        ----------
        game_state: vizdoom.vizdoom.GameState
        episode: int
            which episode are we on in training - used to determine epsilon
        policy: _vizdoom.exploration_policy.Policy
            sub-class of the Policy base class

        Returns
        -------
        action: list[bool]
        """

        preprocessed_curr = preprocess(game_state.screen_buffer, resolution=self.config.STATE_DIMS, convert_to_gray=self.config.CONVERT_TO_GRAYSCALE)
        preprocessed_curr = np.expand_dims(preprocessed_curr, axis=0)  # add batch dim
        q_values = self.model.predict_q_values(preprocessed_curr)

        action_idx = policy.get_action_idx(q_values, episode=episode)
        action = self.actions[action_idx]

        return action

    # TODO: add batch into the configuration, and consider using multiple last frames for the CNN trained_models.
    def update_policy(self, batch):
        """
        Perform single update step on a randomly sampled minibatch from the replay buffer
        Parameters
        ----------
        batch: tuple
            single mini-batch sampled from replay buffer using the get_minibatch() method
            (s_curr, action_indices, s_next, r, is_terminal)

        Returns
        -------
        loss: float
        """
        s_curr, action_indices, s_next, r, is_terminal = batch

        if self.double:
            # FIXME I have not yet applied the bug fix on the bellman equation to the DDQN
            which_model_to_update = random.randint(0, 1)

            if which_model_to_update == 0:
                q_target, next_state_max_q = self.retrieve_current_target_double(s_curr, s_next, self.model_1, self.model_2)
                q_target[:, action_indices] = r + self.discount_factor * (1 - is_terminal) * next_state_max_q
                loss = self.model_1.train_on_batch(s_curr, q_target)
            else:
                q_target, next_state_max_q = self.retrieve_current_target_double(s_curr, s_next, self.model_2, self.model_1)
                q_target[:, action_indices] = r + self.discount_factor * (1 - is_terminal) * next_state_max_q
                loss = self.model_1.train_on_batch(s_curr, q_target)

        else:

            # TODO this can almost certainly be done more efficiently
            # get the predicted highest q value of any action at at the new state
            next_state_max_q = np.max(self.model.predict_q_values(s_next), axis=1)  # (batch_size,)

            action_indices_array = np.eye(self.num_actions)[action_indices]  # (batch_size, num_actions)

            # update q_taget at the position of the actions taken using the Bellman equation
            current_q_values = self.model.predict_q_values(s_curr)  # (batch_size, num_actions)

            q_update = np.expand_dims(r + self.discount_factor * (1 - is_terminal) * next_state_max_q, axis=-1)  # (batch_size, 1)

            q_target = (current_q_values * (1 - action_indices_array)) + (action_indices_array * q_update)  # (batch_size, num_actions)

            loss = self.model.train_on_batch(s_curr, q_target)

        return loss

    def retrieve_current_target_double(self, s1, s2, model_to_update, model_for_target):
        a_max = model_to_update.predict_best_action(s2)
        q_max = model_for_target.predict_q_values(s2)[a_max]
        q_target = model_to_update.predict_q_values(s1)
        return q_target, q_max

    def load_model(self, model_dir):
        self.model = QModel()
        self.model.keras_model = load_model(os.path.join(model_dir, 'highest_reward.h5'))
        print('\n Keras model succesfully loaded from: ', model_dir)




