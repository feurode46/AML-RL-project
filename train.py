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

    target_env = gym.make('CustomHopper-target-v0')
    print('State space:', target_env.observation_space)  # state-space
    print('Action space:', target_env.action_space)  # action-space
    print('Dynamics parameters:', target_env.get_parameters())  # masses of each link of the Hopper
    target_env = DummyVecEnv([lambda: target_env])

    PPO_path = os.path.join("training", "models", "PPO_1M")  
    PPO_target_path = os.path.join("training", "models", "PPO_TARGET_1M")  

    print("--- TRAIN PPO ON SOURCE ENVIRONMENT --- ")      
    if os.path.exists("training/models/PPO_1M.zip"):
        print("Found source model!")
        model = PPO.load("training/models/PPO_1M", env=env)
    else:
        print("source model file not found. training...")
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=1000000)
        model.save(PPO_path)
    print("--- TRAIN PPO ON TARGET ENVIRONMENT --- ")      
    if os.path.exists("training/models/PPO_TARGET_1M.zip"):
        print("Found target model!")
        model_target = PPO.load("training/models/PPO_TARGET_1M", env=env)
    else:
        print("target model file not found. training...")
        model_target = PPO("MlpPolicy", target_env, verbose=1)
        model_target.learn(total_timesteps=1000000)
        model_target.save(PPO_target_path)
    
    print("Source-Source environment results:")
    print(evaluate_policy(model, env, n_eval_episodes=50, render=False))
    
    print("Source-Target environment results:")
    print(evaluate_policy(model, target_env, n_eval_episodes=50, render=False))

    print("Target-Target environment results:")
    print(evaluate_policy(model_target, target_env, n_eval_episodes=50, render=False))

    env.close()
    # - test the policy with stable-baselines3 on <source,target> envs

if __name__ == '__main__':
    main()