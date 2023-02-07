"""Sample script for training a control policy on the Hopper environment, using PPO algorithm.
We are searching for the optimal value for the learning rate
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os


def main():
    env = gym.make('CustomHopper-source-v0')
    env = DummyVecEnv([lambda: env])
    
    target_env = gym.make('CustomHopper-target-v0')
    target_env = DummyVecEnv([lambda: target_env])

    # hyperparameter: learning rate
    learning_rates = [1e-3, 3e-4, 1e-4, 1e-5, 1e-6]
    i=0
        
    for lr in learning_rates:    
        PPO_path = os.path.join("training", "models", "PPO", f"LR{i}")  
        PPO_target_path = os.path.join("training", "models", "PPO", f"TARGET_LR{i}")  

        print(f"--- TRAIN PPO ON SOURCE ENVIRONMENT (Learning Rate = {lr})--- ")      
        if os.path.exists(f"training/models/PPO/LR{i}.zip"):
            print("Found source model!")
            model = PPO.load(f"training/models/PPO/LR{i}", env=env)
        else:
            print("source model file not found. training...")
            model = PPO("MlpPolicy", env, verbose=1, learning_rate=lr)
            model.learn(total_timesteps=500000, progress_bar = True)
            model.save(PPO_path)
        print(f"--- TRAIN PPO ON TARGET ENVIRONMENT (Learning Rate = {lr})--- ")      
        if os.path.exists(f"training/models/PPO/TARGET_LR{i}.zip"):
            print("Found target model!")
            model_target = PPO.load(f"training/models/PPO/TARGET_LR{i}", env=target_env)
        else:
            print("target model file not found. training...")
            model_target = PPO("MlpPolicy", target_env, verbose=1, learning_rate=lr)
            model_target.learn(total_timesteps=500000, progress_bar=True)
            model_target.save(PPO_target_path)


        print("Source-Source environment results:")
        print(evaluate_policy(model, env, n_eval_episodes=50, render=False))
        
        print("Source-Target environment results:")
        print(evaluate_policy(model, target_env, n_eval_episodes=50, render=False))

        print("Target-Target environment results:")
        print(evaluate_policy(model_target, target_env, n_eval_episodes=50, render=False))
        i+=1

        env.close()

if __name__ == '__main__':
    main()