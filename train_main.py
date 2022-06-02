from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from stable_baselines3 import TD3
from stable_baselines3.td3 import MlpPolicy as td3MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.cmd_util import make_vec_env # Module cmd_util will be renamed to env_util https://github.com/DLR-RM/stable-baselines3/pull/197
import numpy as np
import gym
import os
import time
from datetime import datetime
import argparse
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
import tensorflow as tf

AGGR_PHY_STEPS = 5
EPISODE_REWARD_THRESHOLD = -0 # Upperbound: rewards are always negative, but non-zero

if __name__ == "__main__":
     #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning experiments script')
    parser.add_argument('--obs',        default='kin',        type=ObservationType,                                                      help='Observation space (default: kin)', metavar='')
    parser.add_argument('--act',        default='one_d_rpm',  type=ActionType,                                                           help='Action space (default: one_d_rpm)', metavar='')
    parser.add_argument('--cpu',        default='1',          type=int,                                                                  help='Number of training environments (default: 1)', metavar='')        

    ARGS = parser.parse_args()

    sa_env_kwargs = dict(aggregate_phy_steps=AGGR_PHY_STEPS, obs=ARGS.obs, act=ARGS.act)

    ######################################## Save directory ########################################
    filename = os.path.dirname(os.path.abspath(__file__))+'/results/save-'+'HoverAviary'+'-'+'TD3'+'-'+ARGS.obs.value+'-'+ARGS.act.value+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    ######################################## Env ########################################
    train_env = make_vec_env(
        HoverAviary,
        env_kwargs=sa_env_kwargs,
        n_envs=ARGS.cpu,
        seed=0)

    eval_env = make_vec_env(
        HoverAviary,
        env_kwargs=sa_env_kwargs,
        n_envs=1,
        seed=0
        )

    offpolicy_kwargs = dict(net_arch=[256, 256, 128])

    # lr = 3e-4 {
        # 1: [512, 512, 256, 128]
        # 2: [512, 256, 128]
        # 3: [256, 256, 128]
    # }

    # [512, 512, 256, 128] {
        # lr = 1e-4
        # lr = 1e-3
    # }

    # [512, 256, 128] {
        # lr = 1e-3
        # lr = 1e-5
        # lr = 5e-4
    # }

     # [256, 256, 128] {
        # lr = 3e-4
        # lr = 1e-4
        # lr = 9e-5
    # }

    # [256, 256, 128], lr=3e-4 {
    #   tau = 0.001
    #   0.025
    #   0.075
    #   0.01
    # }

    model = TD3(
        td3MlpPolicy,
        train_env,
        gamma=0.99,
        learning_rate=3e-4,
        tau=0.01,

        policy_kwargs=offpolicy_kwargs,
        tensorboard_log=filename+'/tb/',
        verbose=1)

    print(model.policy)

    #### Train the model #######################################
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=filename+'-logs/', name_prefix='rl_model')
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=EPISODE_REWARD_THRESHOLD,
        verbose=1)
        
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=filename+'/',
        log_path=filename+'/',
        eval_freq=int(2000/ARGS.cpu),
        deterministic=True,
        render=False)
        
    model.learn(
        total_timesteps=120_000, #35000, #int(3e12)
        callback=eval_callback,
        log_interval=100)

    #### Save the model ########################################
    model.save(filename+'/success_model.zip')
    print(filename)

    #### Print training progression ############################
    # with np.load(filename+'/evaluations.npz') as data:
    #     for j in range(data['timesteps'].shape[0]):
    #         print(str(data['timesteps'][j])+","+str(data['results'][j][0][0]))