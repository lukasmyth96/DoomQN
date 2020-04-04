import argparse
import os

import vizdoom as vzd

from definition import ROOT_DIR
from doomqn.utils.io import pickle_load
from doomqn.scripts.train import Trainer
from doomqn.exploration_policy import EpsilonGreedy, Greedy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str, nargs='?', default=os.path.join(ROOT_DIR, 'trained_models/my_way_home_icm_attempt_1'))
    parser.add_argument('episodes', type=int, nargs='?', default=10)
    pargs = parser.parse_args()

    config = pickle_load(os.path.join(pargs.model_dir, 'config.pkl'))
    config.VISIBLE = True
    config.LEVEL = 'my_way_home'
    config.CFG_FILEPATH = os.path.join(pargs.model_dir, 'level_config.cfg')  # ensures level cfg is same as training

    trainer = Trainer(config)

    # override any environemnt configs before training
    trainer.environment.change_screen_resolution(vzd.RES_1280X960)  # be careful to keep the aspect ratio same as training

    trainer.agent.load_model(pargs.model_dir)

    policy = EpsilonGreedy(fixed_epsilon=0.3)
    mean_reward = trainer.evaluate_model(episodes=pargs.episodes, max_timestemps=2100, policy=policy, sleep=0)
    trainer.environment.game.close()

    print('\n\n MEAN REWARD OVER {} EPISODES: {} \n\n'.format(pargs.episodes, mean_reward))