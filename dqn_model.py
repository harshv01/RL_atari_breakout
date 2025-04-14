#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #

        self.in_channels = in_channels
        self.num_actions = num_actions

        self.model_p1 = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.model_p2 = nn.Sequential(
            nn.Linear(in_features=3136, out_features=512),  # Assuming 84x84 image size input.
            nn.ReLU(),
            nn.Linear(512, self.num_actions)   
        )

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        x = x.permute(0, 3, 1, 2)
        # print("Shape of input: ", x.shape)
        # print("------two")

        #Normalize

        # x = x/torch.max(x)

        x=self.model_p1(x)  
        # x = x.view(x.size(0), -1)   # To Flatten
        x= torch.flatten(x, start_dim=1)
        x = self.model_p2(x)

        ###########################
        return x
