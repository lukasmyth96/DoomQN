import os

from definition import ROOT_DIR, SCENARIOS_DIR

from doomqn.exploration_policy import EpsilonGreedy, Greedy


class Config:

    # Game settings
    LEVEL = 'my_way_home'  # name of level
    CFG_FILEPATH = os.path.join(SCENARIOS_DIR, 'my_way_home.cfg')
    VISIBLE = False

    # Experiment config
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'trained_models')

    TOTAL_EPISODES = 10000
    MAX_STEPS_PER_EPISODE = 2100
    SKIP_RATE = 8
    EVAL_EVERY = 25
    EVAL_EPISODES = 10

    # Exploration settings
    INITAL_EPS = 0.95
    EPS_DECAY_FACTOR = 0.9995  # factor by which to decay epsiolen each episode WARNING! small changes in this value can have a huge impact on the learning
    MIN_EPS = 0.2
    EXPLORATION_POLICY = EpsilonGreedy(initial_eps=INITAL_EPS, min_eps=MIN_EPS, decay_factor=EPS_DECAY_FACTOR)

    BUFFER_SIZE = 10000
    EVAL_BUFFER_SIZE = 10000
    BATCH_SIZE = 32
    PRIORITIZED_SAMPLING = True
    MIN_TIMESTEPS_BEFORE_TRAINING = 32  # minimum number of timesteps to play before starting training

    # Model config
    CONVERT_TO_GRAYSCALE = True  # if True will convert images to grayscale - else will use RGB channels
    STATE_DIMS = (64, 64)  # (w, h) dimensions of state after preprocessing
    colour_channels = 1 if CONVERT_TO_GRAYSCALE else 3
    INPUT_SHAPE = STATE_DIMS + (colour_channels,)

    USE_DOUBLE_DQN = False
    DISCOUNT_FACTOR = 0.9
    INTRINSIC_REWARD_WEIGHT = 50
    ACTIONS = None  # set during training

