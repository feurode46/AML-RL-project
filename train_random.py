"""Training over randomized mass parameters.

   Algorithm used: PPO (for the time being)
"""
import gym
# from gym.wrappers.pixel_observation import PixelObservationWrapper
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
import utils

model_name = "PPO_random_100k"


def main():
    env = gym.make('CustomHopper-source-v0')
    initial_parameters = list(env.sim.model.body_mass)     
    # - train a policy with stable-baselines3 on source env
    env = DummyVecEnv([lambda: env])

    target_env = gym.make('CustomHopper-target-v0')
    target_env = DummyVecEnv([lambda: target_env])
    
    PPO_path = os.path.join("training", "models", model_name)

    randomized_environments = 100 # hyperparameter
    training_total_timesteps = 100000 # hyperparameter
    rand_proportion = 0.5 # hyperparameter
    timesteps_for_each = int(training_total_timesteps/randomized_environments)

    print("--- TRAIN PPO ON RANDOMIZED SOURCE ENVIRONMENT --- ")      
    if os.path.exists(f"training/models/{model_name}.zip"):
        print("Found source model!")
        model = PPO.load(f"training/models/{model_name}", env=env)
    else:
        print("source model file not found. training...")
        model = PPO("MlpPolicy", env, verbose=1)
        for i in range(randomized_environments):
            model.learn(total_timesteps=timesteps_for_each)
            # randomize parameters
            for param_idx in range(2, len(initial_parameters)):
                env.env_method("set_mass", param_idx, utils.randomize_parameter_uniform(initial_parameters[param_idx], rand_proportion), indices=0)
        model.save(PPO_path)

    print("Source-Source environment results:")
    print(evaluate_policy(model, env, n_eval_episodes=50, render=False))
    
    print("Source-Target environment results:")
    print(evaluate_policy(model, target_env, n_eval_episodes=50, render=False))

    print("Target-Target environment results:")
    # print(evaluate_policy(model_target, target_env, n_eval_episodes=50, render=False))

    env.close()

if __name__ == '__main__':
    main()