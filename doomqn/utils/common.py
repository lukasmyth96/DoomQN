import os
from datetime import datetime

import matplotlib.pyplot as plt


def create_action_from_index(action_index, num_actions):
    action = [False]*num_actions
    action[action_index] = True
    return action


def display_state(state):
    frames = state.shape[0]
    for frame in range(frames):
        plt.subplot(1, frames, frame + 1)
        plt.imshow(state[frame], cmap='Greys_r')
    plt.show()


def create_directory_path_with_timestamp(destination_dir, dir_prefix=''):
    directory_name = datetime.now().strftime("%Y_%m_%d_T%H_%M_%S")
    if dir_prefix != '':
        directory_name = dir_prefix + directory_name
    sub_dir = os.path.join(destination_dir, directory_name)
    os.mkdir(sub_dir)
    return sub_dir