import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

model_name = "PPO_1M"

def main():
    env = gym.make('CustomHopper-source-v0')
    target_env = gym.make('CustomHopper-target-v0')

    print("--- TEST PPO --- ")      
    if os.path.exists(f"training/models/{model_name}.zip"):
        print("Found source model!")
        model = PPO.load(f"training/models/{model_name}", env=env)
    
    render = False
    if render:
        # render 50 episodes
        n_episodes = 50
        for ep in range(n_episodes):  
            done = False
            state = env.reset()  # Reset environment to initial state

            while not done:  # Until the episode is over
                action, _ = model.predict(state)
                state, reward, done, info = env.step(action)  # Step the simulator to the next timestep
                if render:
                    env.render()
        env.close()
    
    # sb3 evaluate policy
    env = DummyVecEnv([lambda: env])
    target_env = DummyVecEnv([lambda: target_env])

    print("Source-Source environment results:")
    print(evaluate_policy(model, env, n_eval_episodes=50, render=False))
    
    print("Source-Target environment results:")
    print(evaluate_policy(model, target_env, n_eval_episodes=50, render=False))

    # print("Target-Target environment results:")
    # print(evaluate_policy(model_target, target_env, n_eval_episodes=50, render=False))

    env.close()
    # - test the policy with stable-baselines3 on <source,target> envs

if __name__ == '__main__':
    main()