"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os


def main():
    env = gym.make('CustomHopper-source-v0')
    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper

    # - train a policy with stable-baselines3 on source env
    env = DummyVecEnv([lambda: env])
    PPO_path = os.path.join("training", "models", "PPO_1M")        
    model = PPO("MlpPolicy", env, verbose=1)
    # if os.path.exists("training/models/PPO_1M.zip"):
    #     print("el wiwi")
    #     model.load("training/models/PPO_1M.zip")
    # else:
    print("file not found. training...")
    model.learn(total_timesteps=1000000)
    model.save(PPO_path)
    print("Source environment results:")
    print(evaluate_policy(model, env, n_eval_episodes=30, render=True))
    target_env = gym.make('CustomHopper-target-v0')
    print('State space:', target_env.observation_space)  # state-space
    print('Action space:', target_env.action_space)  # action-space
    print('Dynamics parameters:', target_env.get_parameters())  # masses of each link of the Hopper
    target_env = DummyVecEnv([lambda: target_env])
    print("Target environment results:")
    print(evaluate_policy(model, target_env, n_eval_episodes=30, render=False))

    env.close()
    # - test the policy with stable-baselines3 on <source,target> envs

if __name__ == '__main__':
    main()