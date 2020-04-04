import os
import numpy as np
import time
from tqdm import tqdm

from doomqn.agent import DQNAgent
from doomqn.intrinsic_curiosity_module import ICM
from doomqn.environment import Environment
from doomqn.experience_replay import ReplayBuffer, Transition
from doomqn.utils.tensorboard import TensorboardLogger
from doomqn.utils.data_preprocess import preprocess
from doomqn.utils.common import create_directory_path_with_timestamp
from doomqn.utils.io import pickle_save
from doomqn.exploration_policy import Greedy, EpsilonGreedy
from doomqn.config import Config


class Trainer:

    def __init__(self, config):

        self.config = config
        self.environment = Environment(level=config.LEVEL, cfg_filepath=config.CFG_FILEPATH, visible=config.VISIBLE)

        self.config.ACTIONS = self.environment.actions  # so the actions can be restored during testing
        self.agent = DQNAgent(config=config)
        self.curiosity_module = ICM(input_shape=config.INPUT_SHAPE, num_actions=self.environment.num_actions)

        self.replay_buffer = ReplayBuffer(capacity=config.BUFFER_SIZE)
        self.evaluation_buffer = ReplayBuffer(capacity=config.EVAL_BUFFER_SIZE)

    def game_step(self, episode, policy):
        """


        Parameters
        ----------
        episode: int
        policy: _vizdoom.exploration_policy.Policy
            sub-class of the Policy base class

        Returns
        -------
        current_state
        action: list[bool]
        next_state
        game_over: bool
        reward: float
        """
        current_state = self.environment.game.get_state()
        action = self.agent.select_action(current_state, episode, policy=policy)
        next_state, reward, game_over = self.environment.step(action, skiprate=self.config.SKIP_RATE)
        return current_state, action, next_state, game_over, reward

    def evaluate_model(self, episodes, max_timestemps, policy=Greedy(), sleep=0):
        """
        Run agent in greedy policy mode for number of episodes and return average overall reward from those episodes
        Parameters
        ----------
        episodes: int
        max_timestemps: int
            max timesteps per episodes
        policy: _vizdoom.exploration_policy.Policy
            sub-class of Policy
        sleep: float
            time to sleep between steps - used during evaluation to prevent too fast to see

        Returns
        -------
        average_reward: float
            the average total reward recieved across the episodes
        """
        if not self.environment.game_initialised:
            self.environment.initialise_game()

        episode_rewards = []  # store final rewards from each episode
        for episode in range(episodes):
            print('======== EPISODE {} ======='.format(episode))
            timestep = 0
            self.environment.game.new_episode()
            while (not self.environment.game.is_episode_finished()) and (timestep < max_timestemps):
                _ = self.game_step(episode=0, policy=policy)
                timestep += 1

                if sleep > 0:
                    time.sleep(sleep)  # sleep is used during evaluation to make actions visible

            if self.config.VISIBLE:
                time.sleep(1)  # to have visible separation between episodes

            episode_rewards.append(self.environment.game.get_total_reward())

            print('======= TOTAL REWARD: {} ======='.format(self.environment.game.get_total_reward()))

        return np.mean(episode_rewards)

    def compute_mean_q_on_eval_buffer(self):
        """
        Calculate the mean maximum return that the current model thinks it can achieve on all the transitions in the
        eval buffer
        Returns
        -------
        mean_max_q: float
        """
        batch = self.evaluation_buffer.get_minibatch(batch_size=self.evaluation_buffer.size)[0]
        q_values = self.agent.model.predict_q_values(state_batch=batch)
        mean_max_q = np.mean(np.max(q_values, axis=1), axis=0)
        return mean_max_q

    def train(self):

        if not self.environment.game_initialised:
            self.environment.initialise_game()

        # Prepare logging
        timestamp_dir = create_directory_path_with_timestamp(destination_dir=self.config.OUTPUT_DIR)
        print('\n Logs will be saved in: ', timestamp_dir)
        pickle_save(os.path.join(timestamp_dir, 'config.pkl'), self.config)
        self.environment.save_cfg(timestamp_dir)  # save level cfg
        tensorboard_logger = TensorboardLogger(log_dir=timestamp_dir)
        evaluation_rewards = []

        try:
            for episode in tqdm(range(self.config.TOTAL_EPISODES)):

                self.environment.game.new_episode()
                tstep = 0
                while (not self.environment.game.is_episode_finished()) and (tstep < self.config.MAX_STEPS_PER_EPISODE):

                    # 1) Take action in game
                    current_state, chosen_action, new_state, is_terminal, reward = self.game_step(episode, policy=self.config.EXPLORATION_POLICY)
                    preprocessed_curr = preprocess(current_state.screen_buffer,
                                                   resolution=self.config.STATE_DIMS,
                                                   convert_to_gray=self.config.CONVERT_TO_GRAYSCALE)
                    preprocessed_next = preprocess(new_state.screen_buffer,
                                                   resolution=self.config.STATE_DIMS,
                                                   convert_to_gray=self.config.CONVERT_TO_GRAYSCALE) if not is_terminal else preprocessed_curr
                    chosen_action_idx = self.environment.actions.index(chosen_action)

                    # 2) Compute prediction error of the forward dynamics model in the ICM - i.e. the intrinsic reward
                    #    and add it to the extrinsic reward
                    intrinsic_reward = self.curiosity_module.train_on_batch(batch=[preprocessed_curr,
                                                                                   chosen_action,
                                                                                   preprocessed_next])
                    reward += self.config.INTRINSIC_REWARD_WEIGHT * intrinsic_reward

                    # 3) Store transition in replay buffer
                    transition = Transition(preprocessed_curr=preprocessed_curr,
                                            action_idx=chosen_action_idx,
                                            reward=reward,
                                            preprocessed_next=preprocessed_next,
                                            is_terminal=is_terminal)

                    # add transitions to eval buffer until that's full then start adding to replay buffer
                    if self.evaluation_buffer.size < self.evaluation_buffer.capacity:
                        self.evaluation_buffer.append(transition)
                    else:
                        self.replay_buffer.append(transition)

                    # 4) Sample random mini-batch from buffer and train model on it
                    if self.replay_buffer.size > self.config.MIN_TIMESTEPS_BEFORE_TRAINING:
                        batch = self.replay_buffer.get_minibatch(self.config.BATCH_SIZE, prioritized=self.config.PRIORITIZED_SAMPLING)
                        self.agent.update_policy(batch)

                    tstep += 1

                if episode % self.config.EVAL_EVERY == 0:

                    # Calculate mean max q on eval buffer
                    mean_max_q = self.compute_mean_q_on_eval_buffer()
                    tensorboard_logger.log_scalar('mean_max_q', value=mean_max_q, step=episode)

                    # Evaluate current policy in greedy model every n episodes
                    mean_reward = self.evaluate_model(episodes=self.config.EVAL_EPISODES,
                                                      max_timestemps=self.config.MAX_STEPS_PER_EPISODE)
                    tensorboard_logger.log_scalar('evaluation_mean_reward', value=mean_reward, step=episode)
                    print('\n\n Episode {} - Mean Eval Reward: {}'.format(episode, mean_reward))

                    # Save model if performance has improved over best so far
                    current_best = max(evaluation_rewards) if evaluation_rewards else -np.inf
                    if (episode == 0) or (mean_reward > current_best):
                        self.agent.model.save(output_path=os.path.join(timestamp_dir, 'highest_reward.h5'))
                        print('Eval reward improved from {} to {} - saving model \n'.format(current_best, mean_reward))
                        evaluation_rewards.append(mean_reward)

            self.environment.game.close()

        except KeyboardInterrupt:
            self.environment.game.close()


if __name__ == "__main__":

    trainer = Trainer(config=Config())
    trainer.train()