import numpy as np
import gym

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Takes environment env as input 
    and modifies the shape of the observation space 
    from (height, width, channels)
    to (channels, height, width)
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
            
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=255.0, 
                                                shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.uint8)

    def observation(self, observation):
        '''
        Called every time the environment returns an observation. 
        This is typically done in the step() or reset() methods of the environment. 
        The observation returned by the environment is passed as an argument to the observation method, 
        which can then be modified and returned to the agent as the final observation
        '''
        frame = np.swapaxes(observation, 2, 0)[None,:,:,:]
        # "swapaxes" -> Swaps the 2nd and 0th axes of the observation numpy array, 
        # effectively transforming it from (height, width, channels) shape 
        # to (channels, height, width) shape
        # +
        # "[None,:,:,:]" -> Adds a new dimension of size 1 at the beginning of the array. 
        # The resulting shape of the frame is (1, channels, height, width), 
        # which is the standard format for CNN inputs in PyTorch.
        return frame