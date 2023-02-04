import gym
import numpy as np
from env.custom_hopper import *
from garage_wrappers.pixel_observation      import PixelObservationWrapper
from garage_wrappers.gray_scale_observation import Grayscale
from garage_wrappers.resize_observation     import Resize
from garage_wrappers.frame_stack            import StackFrames


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
# The source_resized_env observation space is a Box object with shape (224, 224)
# => This observation space is a 2D grayscale image with 224 rows and 224 columns

source_frame_stack_env = StackFrames(source_resized_env, 4)
# print(source_frame_stack_env.observation_space)
# Creates an environment source_frame_stack_env that stacks 4 consecutive frames from the resized_env environment 
# and returns the resulting 4-frame stack as a SINGLE observation
# The source_frame_stack_env observation space is a Box object with shape (224, 224, 4)
# => This observation space is a the the result of stacking 4 grayscale frames of the resized environment


