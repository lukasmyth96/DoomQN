import argparse
import os
import time

from pynput.keyboard import Key, Listener
import vizdoom as vzd
from tqdm import tqdm

from doomqn.environment import Environment
from doomqn.experience_replay import ReplayBuffer, Transition
from doomqn.utils.data_preprocess import preprocess
from definition import ROOT_DIR, SCENARIOS_DIR


class Teacher:

    def __init__(self, level, cfg_filepath, timesteps=1000, img_dims=(64, 64), grayscale=True):
        """

        Parameters
        ----------
        level: str
            level name - a .wad file with that name must exist in the scenarios folder
        cfg_filepath: str
            path to .cfg configuration file
        """
        self.level = level
        self.cfg_filepath = cfg_filepath

        self.timesteps = timesteps
        self.img_dims = img_dims
        self.grayscale = grayscale

        self.environment = Environment(level=level, cfg_filepath=cfg_filepath)
        self.environment.change_screen_resolution(vzd.RES_1280X960)
        self.environment.initialise_game()

        self.replay_buffer = ReplayBuffer(capacity=timesteps)
        self.current_key = None
        self.listener = Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        self.listener.start()

        self.current_keys = {0: False,
                             1: False,
                             2: False}

        self.key_to_action_idx = {Key.left: 0,
                                  Key.right: 1,
                                  Key.up: 2}

    def on_key_press(self, key):
        """ add key to key log"""
        key_idx = self.key_to_action_idx.get(key)
        if key_idx is not None:
            self.current_keys[key_idx] = True

    def on_key_release(self, key):
        key_idx = self.key_to_action_idx.get(key)
        if key_idx is not None:
            self.current_keys[key_idx] = False

    def get_action_from_user(self):
        action = [self.current_keys[idx] for idx in range(self.environment.num_actions)]

        return action

    def game_step(self, action):
        """
        Take action in game
        Parameters
        ----------
        action: list[bool]

        Returns
        -------
        current_state
        action: list[bool]
        next_state
        game_over: bool
        reward: float
        """
        current_state = self.environment.game.get_state()
        next_state, reward, game_over = self.environment.step(action, skiprate=1)
        return current_state, action, next_state, game_over, reward

    def collect_demonstrations(self):

        self.environment.game.new_episode()
        t = 0
        while t < self.timesteps:

            try:
                action = self.get_action_from_user()
                if any(action):

                    current_state, chosen_action, new_state, is_terminal, reward = self.game_step(action)

                    # preprocessed_curr = preprocess(current_state.screen_buffer,
                    #                                resolution=self.img_dims,
                    #                                convert_to_gray=self.grayscale)
                    # preprocessed_next = preprocess(new_state.screen_buffer,
                    #                                resolution=self.img_dims,
                    #                                convert_to_gray=self.grayscale) if not is_terminal else preprocessed_curr
                    #chosen_action_idx = self.environment.actions.index(chosen_action)

                    self.replay_buffer.append(Transition(preprocessed_curr=current_state.screen_buffer,
                                                         action_idx=0,  # FIXME
                                                         reward=reward,
                                                         preprocessed_next=new_state.screen_buffer if new_state is not None else current_state.screen_buffer,
                                                         is_terminal=is_terminal))

                    time.sleep(1.0 / vzd.DEFAULT_TICRATE)

                    t += 1
            except vzd.vizdoom.ViZDoomErrorException:
                self.environment.game.new_episode()

            if self.environment.game.is_episode_finished():
                self.environment.game.close()
                self.environment.initialise_game()
                self.environment.game.new_episode()

        self.environment.game.close()


if __name__ == '__main__':
    """
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, default='my_way_home')
    parser.add_argument('--cfg_filepath', type=str, default=os.path.join(SCENARIOS_DIR, 'my_way_home.cfg'))
    pargs = parser.parse_args()

    _teacher = Teacher(level=pargs.level, cfg_filepath=pargs.cfg_filepath)
    _teacher.collect_demonstrations()