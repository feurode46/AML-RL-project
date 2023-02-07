import gym
from env.custom_hopper import *
from garage_wrappers.pixel_observation      import PixelObservationWrapper
from garage_wrappers.gray_scale_observation import Grayscale
from garage_wrappers.resize_observation     import Resize
from garage_wrappers.frame_stack            import StackFrames
from garage_wrappers.max_and_skip           import MaxAndSkip
from garage_wrappers.pytorch_observation    import ImageToPyTorch
from custom_cnn                             import CustomCNN
from custom_vgg                             import CustomVGG
from custom_resnet18                        import CustomResNet18
from stable_baselines3 import PPO, TRPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

source_env = gym.make('CustomHopper-source-v0')
target_env = gym.make('CustomHopper-target-v0')


# print(source_env.observation_space)
# Output is Box([-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf], [inf inf inf inf inf inf inf inf inf inf inf], (11,), float64)
# observation space of the "Hopper-v4" environment is a 11-dimensional continuous space represented by a Box object. 
# - The lower bound of the space is [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf] 
#   and the upper bound is [inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf]. 
# - The Box object has a data type of float64. This means that the observation values are float numbers with a precision of 64 bits.

source_pixel_env = PixelObservationWrapper(source_env)
# print(source_pixel_env.observation_space)
# The source_pixel_env observation space is a Box object with shape (500, 500, 3) 
#   where the first two dimensions represent the height and width of the image, 
#   and the third dimension represents the RGB (Red, Green, Blue) color channels, 
#   and data type uint8.
# The values of the observations are between 0 and 255, with 0 representing black and 255 representing white.
# => This observation space represents a 500x500 pixel image.



source_grayscale_env = Grayscale(source_pixel_env)
# print(source_grayscale_env.observation_space)
# The source_grayscale_env observation space is a Box object with shape (500, 500)
# => This observation space is a 2D grayscale image with 500 rows and 500 columns

source_resized_env = Resize(source_grayscale_env, 224, 224)
# print(source_resized_env.observation_space)
# The source_resized_env observation space is a Box object with shape (64, 64)
# => This observation space is a 2D grayscale image with 64 rows and 64 columns

# source_frameskip_env = MaxAndSkip(source_resized_env, skip=2)
source_frame_stack_env = StackFrames(source_resized_env, 12)
# print(source_frame_stack_env.observation_space)
# Creates an environment source_frame_stack_env that stacks 4 consecutive frames from the resized_env environment 
# and returns the resulting 4-frame stack as a SINGLE observation
# The source_frame_stack_env observation space is a Box object with shape (64, 64, 4)
# => This observation space is the result of stacking 4 grayscale frames of the resized environment

import matplotlib.pyplot as plt
source_pyTorch_env = ImageToPyTorch(source_frame_stack_env)

# print(source_pyTorch_env.observation_space)
# The source_pyTorch_env observation space is a Box object with shape (4, 64, 64)
# => This observation space is the same as source_frame_stack_env, but with inverted dimensions
# from (height, width, channels) to (channels, height, width)

# target_pixel_env = PixelObservationWrapper(target_env)
# target_grayscale_env = Grayscale(target_pixel_env)
# target_resized_env = Resize(target_grayscale_env, 64, 64)
# target_frame_stack_env = StackFrames(target_resized_env, 4)
# target_pyTorch_env = ImageToPyTorch(target_frame_stack_env)

# Taken from https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
policy_kwargs = dict(
    features_extractor_class=CustomCNN, # to use other model: CustomCNN
    features_extractor_kwargs=dict(features_dim = 128), # was 128 in oldcnn
)

# policy_kwargs= dict(
#     features_extractor_class=MyResNet18,
#     features_extractor_kwargs=dict(features_dim=128),
# )

source_env.enable_uniform_domain_randomization(rand_proportion = 30)
model = TRPO("CnnPolicy", source_pyTorch_env, policy_kwargs = policy_kwargs, verbose = 1, batch_size = 32, learning_rate = 0.0003)
trained_model = model.learn(total_timesteps=100000, progress_bar=True)
trained_model.save(f"./training/models/vision_cnn_noskip")