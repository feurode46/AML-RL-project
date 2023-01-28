"""Training over randomized mass parameters.

   Algorithm used: PPO (for the time being)
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
import numpy as np
import utils

def main():
    RAND_PROPORTION = 0.5 # randomize parameters by +- 30%

    env = gym.make('CustomHopper-source-v0')
    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper
    initial_parameters = list(env.sim.model.body_mass) # save initial masses to randomize
    # print("*** Body mass parameters: ***")

    n_episodes = 10
    render = True
    for ep in range(n_episodes):
        done = False
        state = env.reset()  # Reset environment to initial state
        # randomize parameters (except the first one, which is torso mass)
        for i in range(2, len(env.sim.model.body_mass)):
            env.set_mass(i, utils.randomize_parameter_uniform(initial_parameters[i], RAND_PROPORTION))
        print('Dynamics parameters:', env.get_parameters())

        while not done:  # Until the episode is over
            action = env.action_space.sample()  # Sample random action

            state, reward, done, info = env.step(action)  # Step the simulator to the next timestep

            """Step 4: vision-based
            img_state = env.render(mode="rgb_array", width=224, height=224)
            """

            if render:
                env.render()
    env.close()

if __name__ == '__main__':
    main()