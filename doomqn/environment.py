import os
import vizdoom as vzd
import numpy as np
import shutil

from definition import SCENARIOS_DIR


class Environment(object):
    def __init__(self, level='basic', cfg_filepath=None, visible=True):
        """

        Parameters
        ----------
        level: str
            name of level - .wad file with that name must exist in scenarios folder
        cfg_filepath: str
            path to .cfg file
        visible: bool, optional
        screen_resolution:
        """
        self.game = vzd.DoomGame()
        self.cfg_path = cfg_filepath
        self.game.load_config(self.cfg_path)
        self.game.set_doom_scenario_path(os.path.join(SCENARIOS_DIR, '{}.wad'.format(level)))
        self.game.set_doom_map("map01")
        self.game.set_window_visible(visible)

        self.num_actions = self.game.get_available_buttons_size()
        self.actions = [list(row) for row in np.eye(self.num_actions, dtype=bool)]
        self.screen_width = self.game.get_screen_width()
        self.screen_height = self.game.get_screen_height()

        self.game_initialised = False

    def initialise_game(self):
        self.game_initialised = True
        self.game.init()

    def change_screen_resolution(self, resolution):
        """
        must happen before game is initialised - used to increase resolution for evaluation viewing
        Parameters
        ----------
        resolution

        Returns
        -------

        """
        if self.game_initialised:
            raise Exception('cannot change screen resolution after game has been initialised')
        self.game.set_screen_resolution(resolution)

    def save_cfg(self, output_dir):
        """ Save cfg file inside model dir so that we can be sure we use the same cfg during testing"""
        output_path = os.path.join(output_dir, 'level_config.cfg')
        shutil.copy(self.cfg_path, output_path)

    def step(self, action, skiprate=1):
        """
        Take action within game
        Parameters
        ----------
        action: list[bool]
            action to be taken
        skiprate: int
            number of frames to repeat this action for

        Returns
        -------
        next_state: vizdoom.vizdoom.GameState
        reward: float
        game_over: bool

        """
        reward = self.game.make_action(action, skiprate)
        next_state = self.game.get_state()
        game_over = self.game.is_episode_finished()
        return next_state, reward, game_over

