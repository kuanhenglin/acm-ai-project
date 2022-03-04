import torch
import torch.nn as nn
import torch.nn.functional as F
import constants

class StartingNetwork(nn.Module):
    def __init__(self):
        # Call nn.Module's constructor
        super().__init__()
        
        # Transfer resnet model (do later after data augmentation, batch normalization)
        self.model_a = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        # Remove last layer
        self.model_a = torch.nn.Sequential(*(list(self.model_a.children())[:-1]))
        # Output of ResNet: 512-d tensor

        # Feature caching: Save a "feature set" of all images after running through resnet18 with batch_size 1 (since it'll be the same every time)
        # Then we only have to train the 2 FC layers
        # Can also mess around with batch size

        self.fc1 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(constants.DROP_OUT)
        self.norm = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        # Forward propagation
        x = self.model_a(x)

        x = x.reshape((32, -1))
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x