import torch as th
import torch.nn as nn
from gym import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Example taken from https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    
    This custom CNN is designed to process input data 
    with a shape represented by the observation_space parameter, 
    which is a gym.Space object. 
    
    It is used to extract a set of features with a specified number of dimensions, 
    specified by the features_dim parameter (defaults to 256).
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume C x H x W images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),  # nn.Tanh() ??
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),  # nn.Tanh() ??
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )

        with th.no_grad():  # Preventing computation of gradients, as we don't need them for this single forward pass
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        # The purpose of the forward pass is to determine the number of features produced by the cnn network, 
        # which will then be used as input to the final linear layer (self.linear) 
        # to project the features down to features_dim dimensions.
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        '''
        Performs the forward pass of the CNN, 
        which takes as input a tensor observations 
        and returns a tensor of extracted features.
        '''
        return self.linear(self.cnn(observations))