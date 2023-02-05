import torch as th
import torchvision
import torch.nn as nn
from gym import spaces
from torchvision.models import resnet18
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MyResNet18(BaseFeaturesExtractor):

    def __init__(self, 
                    observation_space: spaces.Box, 
                    features_dim: int = 256,
                    augmentation = False
                    ):

        super().__init__(observation_space, features_dim)
        
        self.augmentation = augmentation
        if augmentation:
            self.aug_trans = nn.Sequential(
                nn.ReplicationPad2d(5),  # 84x84 --> 94x94
                torchvision.transforms.RandomCrop(size=(84,84)) )  #Then crops it to 84x84 again

        n_input_channels = observation_space.shape[0]
        self.net = resnet18(weights=torchvision.models.ResNet18_Weights)
        self.net.conv1 = nn.Sequential(
                            nn.Conv2d(n_input_channels, 64, kernel_size = 7, stride = 2, padding = 3),
                            nn.BatchNorm2d(64),
                            nn.ReLU())
        self.net.fc = nn.Linear(in_features=512, out_features=128)


    def forward(self, observations: th.Tensor) -> th.Tensor:

        x = observations  #SHAPE [4, 3, 84, 84]

        # Code for iterative forward of 4 images
        # preds = []
        # for i in range(0, x.shape[1], 3): 
        #     frame = x[:, i: i+2 +1 , :, :]
        #     preds.append( self.net(frame) ) 

        # preds = th.cat(preds, dim=1)
        # return preds


        # Data Augmentation (DA)
        # x = self.aug_trans(x)
        
        x = self.net(x)
        return x