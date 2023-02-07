import gym
from env.custom_hopper import *
from stable_baselines3 import PPO, SAC
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os


def main():
    env = gym.make('CustomHopper-source-v0')
    env = DummyVecEnv([lambda: env])

    target_env = gym.make('CustomHopper-target-v0')
    target_env = DummyVecEnv([lambda: target_env])

    TRPO_path = os.path.join("training", "models", "TRPO_100k")  
    TRPO_target_path = os.path.join("training", "models", "TRPO_TARGET_100k")  

    print("--- TRAIN TRPO ON SOURCE ENVIRONMENT --- ")      
    if os.path.exists("training/models/TRPO_100k.zip"):
        print("Found source model!")
        model = TRPO.load("training/models/TRPO_100k", env=env)
    else:
        print("source model file not found. training...")
        model = TRPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=100000)
        model.save(TRPO_path)
    print("--- TRAIN TRPO ON TARGET ENVIRONMENT --- ")      
    if os.path.exists("training/models/TRPO_TARGET_100k.zip"):
        print("Found target model!")
        model_target = TRPO.load("training/models/TRPO_TARGET_100k", env=env)
    else:
        print("target model file not found. training...")
        model_target = TRPO("MlpPolicy", target_env, verbose=1)
        model_target.learn(total_timesteps=100000)
        model_target.save(TRPO_target_path)


    print("Source-Source environment results:")
    print(evaluate_policy(model, env, n_eval_episodes=50, render=False))
    
    print("Source-Target environment results:")
    print(evaluate_policy(model, target_env, n_eval_episodes=50, render=False))

    print("Target-Target environment results:")
    print(evaluate_policy(model_target, target_env, n_eval_episodes=50, render=False))
    
    env.close()


if __name__ == '__main__':
    main()