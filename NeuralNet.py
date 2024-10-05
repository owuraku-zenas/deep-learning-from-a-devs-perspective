import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # The input size is 64x64 with 3 color channels, so the input to the first fully connected layer is 12288 (64*64*3)
        self.fc1 = nn.Linear(12288, 84)  # First fully connected layer
        self.fc2 = nn.Linear(84, 50)     # Second fully connected layer
        self.fc3 = nn.Linear(50, 2)      # Output layer (2 classes)
        
    def forward(self, x):  # Add the input argument 'x'
        """
        Forward pass of the neural network.
        View flattens the 3D Tensor to a 1D tensor.
        """
        # Flatten the input (batch_size, 3, 64, 64) -> (batch_size, 12288)
        x = x.view(x.size(0), -1)  # Keep batch size, flatten rest
        
        # Pass the input through the network
        x = F.relu(self.fc1(x))    # Apply ReLU after first layer
        x = F.relu(self.fc2(x))    # Apply ReLU after second layer
        # x = F.relu(self.fc3(x))
        x = self.fc3(x)            # Output layer (no softmax if using CrossEntropyLoss)
        return x


