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

model_name = "PPO_vision_based_UDR_100k"

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

                if render:
                    env.render()
        env.close()
    
    env = DummyVecEnv([lambda: env])
    target_env = DummyVecEnv([lambda: target_env])

    print("Source-Source environment results:")
    print(evaluate_policy(model, pyTorch_env, n_eval_episodes=50, render=False))
    
    print("Source-Target environment results:")
    print(evaluate_policy(model, target_pyTorch_env, n_eval_episodes=50, render=False))

    env.close()

if __name__ == '__main__':
    main()