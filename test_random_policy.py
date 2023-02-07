"""Test a random policy on the Gym Hopper environment.
"""
import gym
from env.custom_hopper import *

def main():
    render = True

    # env = gym.make('CustomHopper-source-v0')  # [2.53429174 3.92699082 2.71433605 5.0893801 ]
    # env = gym.make('CustomHopper-target-v0')  # [3.53429174 3.92699082 2.71433605 5.0893801 ] 
    env = gym.make('CustomHopper-source-v0')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper

    n_episodes = 500

    for ep in range(n_episodes):  
        done = False
        state = env.reset()  # Reset environment to initial state

        while not done:  # Until the episode is over
            action = env.action_space.sample()  # Sample random action

            state, reward, done, info = env.step(action)  # Step the simulator to the next timestep

            if render:
                env.render()
    env.close()

if __name__ == '__main__':
    main()