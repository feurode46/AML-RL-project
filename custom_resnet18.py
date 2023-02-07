import torch as th
import torchvision
import torch.nn as nn
from gym import spaces
from torchvision.models import resnet18
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomResNet18(BaseFeaturesExtractor):
    def __init__(self, 
                    observation_space: spaces.Box, 
                    features_dim: int = 128,
                    augmentation = False
                    ):

        super().__init__(observation_space, features_dim)
        
        self.augmentation = augmentation
        if augmentation:
            self.aug_trans = nn.Sequential(
                nn.ReplicationPad2d(5), # pad the image...
                torchvision.transforms.RandomCrop(size=(224,224))) # ...then randomly crop it to fit

        n_input_channels = observation_space.shape[0]
        self.net = resnet18(weights=torchvision.models.ResNet18_Weights)
        self.net.conv1 = nn.Sequential(
                            nn.Conv2d(n_input_channels, 64, kernel_size = 7, stride = 2, padding = 3),
                            nn.BatchNorm2d(64),
                            nn.ReLU())
        self.net.fc = nn.Linear(in_features=512, out_features=128)


    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations
        x = self.net(x)
        return x