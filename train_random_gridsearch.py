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
    resultsfile = open("gridsearch_results.txt", "a")
    env = gym.make('CustomHopper-source-v0')
    initial_parameters = list(env.sim.model.body_mass)     
    env = DummyVecEnv([lambda: env])

    target_env = gym.make('CustomHopper-target-v0')
    target_env = DummyVecEnv([lambda: target_env])

    randomized_environments_vector = [10, 50, 100, 200, 500, 1000] # hyperparameter
    training_total_timesteps = 100000 # hyperparameter
    rand_proportion_vector = [0.3, 0.5, 0.7, 0.9] # hyperparameter   

    for randomized_environments in randomized_environments_vector:
        for rand_proportion in rand_proportion_vector:
                timesteps_for_each = int(training_total_timesteps/randomized_environments)
                PPO_path = os.path.join("training", "models", model_name + "_" + str(rand_proportion) + "_" + str(randomized_environments))
                env = gym.make('CustomHopper-source-v0')
                env = DummyVecEnv([lambda: env])
                print(f"--- TRAIN PPO ON RANDOMIZED SOURCE ENVIRONMENT ; rand_prop={rand_proportion}, envs = {randomized_environments}--- ")      
                if os.path.exists(f"training/models/{model_name}_{rand_proportion}_{randomized_environments}.zip"):
                    print("Found source model!")
                    model = PPO.load(f"training/models/{model_name}_{rand_proportion}_{randomized_environments}", env=env)
                else:
                    print("source model file not found. training...")
                    model = PPO("MlpPolicy", env, verbose=1)
                    for i in range(randomized_environments):
                        model.learn(total_timesteps=timesteps_for_each)
                        # randomize parameters
                        for param_idx in range(2, len(initial_parameters)):
                            env.env_method("set_mass", param_idx, utils.randomize_parameter_uniform(initial_parameters[param_idx], rand_proportion), indices=0)
                    model.save(PPO_path)

                resultsfile.write(f"Source-Source environment results (rand_prop={rand_proportion}, envs = {randomized_environments}):")
                result = evaluate_policy(model, env, n_eval_episodes=50, render=False)
                resultsfile.write(str(result))
                
                resultsfile.write(f"Source-Target environment results (rand_prop={rand_proportion}, envs = {randomized_environments}):")
                result2 = evaluate_policy(model, target_env, n_eval_episodes=50, render=False)
                resultsfile.write(str(result2))

                env.close()
    resultsfile.close()
if __name__ == '__main__':
    main()