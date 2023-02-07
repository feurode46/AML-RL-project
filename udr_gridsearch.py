"""Training over randomized mass parameters.

   Algorithm used: PPO
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os


def train_and_test(model_name, total_timesteps, rand_proportion_vector, eps_vector, resultsfile):

    resultsfile.write(f"\n##### {model_name} #####\n")

    for r in rand_proportion_vector:
        for rand_eps in eps_vector:
            dest_path = os.path.join("training", "models", model_name + "_random_r_" + str(r) + "_eps_" + str(rand_eps) + "_.zip")

            source_env = gym.make('CustomHopper-source-v0')
            source_env.enable_uniform_domain_randomization(rand_proportion = r, rand_eps=rand_eps)
            source_env = DummyVecEnv([lambda: source_env])

            model = PPO("MlpPolicy", source_env, verbose=1)
            model.learn(total_timesteps = total_timesteps)

            target_env = gym.make('CustomHopper-target-v0')
            target_env = DummyVecEnv([lambda: target_env])

            resultsfile.write(f"Source-Source environment results (rand_prop={r}, rand_eps={rand_eps}):\n")
            result = evaluate_policy(model, source_env, n_eval_episodes=50, render=False)
            resultsfile.write(str(result) + "\n")

            resultsfile.write(f"Source-Target environment results (rand_prop={r}, rand_eps={rand_eps}):\n")
            result2 = evaluate_policy(model, target_env, n_eval_episodes=50, render=False)
            resultsfile.write(str(result2) + "\n")

            model.save(dest_path)
            source_env.close()
            target_env.close()

resultsfile = open("__gridsearch_results_new.txt", "a")
rand_proportion_vector = [30, 50, 70]
n_eps_vector = [1, 5, 10, 50]


def main():
    train_and_test("PPO_100k",  100000,     rand_proportion_vector, n_eps_vector, resultsfile )
    train_and_test("PPO_500k",  500000,     rand_proportion_vector, n_eps_vector, resultsfile )
    train_and_test("PPO_1M",   1000000,     rand_proportion_vector, n_eps_vector, resultsfile )

if __name__ == "__main__":
    main()