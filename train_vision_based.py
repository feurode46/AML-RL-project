import gym
from env.custom_hopper import *
from gym.wrappers.pixel_observation         import PixelObservationWrapper
from gym.wrappers.gray_scale_observation    import GrayScaleObservation
from gym.wrappers.resize_observation        import ResizeObservation
from gym.wrappers.frame_stack               import FrameStack



source_env = gym.make('CustomHopper-source-v0')
target_env = gym.make('CustomHopper-target-v0')

print(source_env.observation_space)
