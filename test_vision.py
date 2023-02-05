"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
import gym
from env.custom_hopper import *
from garage_wrappers.pixel_observation      import PixelObservationWrapper
from garage_wrappers.gray_scale_observation import Grayscale
from garage_wrappers.resize_observation     import Resize
from garage_wrappers.frame_stack            import StackFrames
from garage_wrappers.pytorch_observation    import ImageToPyTorch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

model_name = "er_mejo"

def main():
    env = gym.make('CustomHopper-source-v0')
    target_env = gym.make('CustomHopper-target-v0')

    pixel_env = PixelObservationWrapper(env)
    grayscale_env = Grayscale(pixel_env)
    resized_env = Resize(grayscale_env, 224, 224)
    frame_stack_env = StackFrames(resized_env, 4)
    pyTorch_env = ImageToPyTorch(frame_stack_env)

    target_pixel_env = PixelObservationWrapper(target_env)
    target_grayscale_env = Grayscale(target_pixel_env)
    target_resized_env = Resize(target_grayscale_env, 224, 224)
    target_frame_stack_env = StackFrames(target_resized_env, 4)
    target_pyTorch_env = ImageToPyTorch(target_frame_stack_env)

    print("--- TEST PPO --- ")      
    if os.path.exists(f"training/models/{model_name}.zip"):
        print("Found source model!")
        model = PPO.load(f"training/models/{model_name}", env=pyTorch_env)
    
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

                """Step 4: vision-based
                img_state = env.render(mode="rgb_array", width=224, height=224)
                """

                if render:
                    env.render()
        env.close()
    
    # sb3 evaluate policy
    env = DummyVecEnv([lambda: env])
    target_env = DummyVecEnv([lambda: target_env])

    print("Source-Source environment results:")
    print(evaluate_policy(model, pyTorch_env, n_eval_episodes=50, render=False))
    
    print("Source-Target environment results:")
    print(evaluate_policy(model, target_pyTorch_env, n_eval_episodes=50, render=False))

    # print("Target-Target environment results:")
    # print(evaluate_policy(model_target, target_env, n_eval_episodes=50, render=False))

    env.close()

if __name__ == '__main__':
    main()