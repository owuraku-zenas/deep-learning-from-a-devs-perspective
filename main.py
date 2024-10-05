import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from NeuralNet import NeuralNet
from CNNNet import CNNNet
import torch.optim as optim
import torch.nn as nn
import torch
import os
from PIL import Image, UnidentifiedImageError
import torch.nn.functional as F

# Define transformations for the data
data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to 64x64 (width x height)
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Define the data paths
train_data_path = "./images/train/"
val_data_path = "./images/val/"
test_data_path = "./images/test/"


def validate_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            try:
                img = Image.open(os.path.join(root, file))
                img.verify()  # Check if it's a valid image
            except (IOError, UnidentifiedImageError):
                print(f"Removing corrupted image: {file}")
                os.remove(os.path.join(root, file))  # Remove the file if it's corrupted

# Run the validation
validate_images(train_data_path)
validate_images(val_data_path)
validate_images(test_data_path)

# Load the datasets
train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=data_transforms)
val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=data_transforms)
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=data_transforms)

# Set the batch size
batch_size = 64

# Create DataLoaders for the datasets
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize the model, optimizer, and loss function
# model = NeuralNet()
model = CNNNet()


optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)
# optimizer = optim.Adagrad(model.parameters(), lr=0.01)
# optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9)

loss_fn = nn.CrossEntropyLoss()  # Define the loss function



if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    model.to(device)

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        
        # Training phase
        model.train()  # Set the model to training mode
        for batch in train_loader:
            optimizer.zero_grad()  # Reset gradients
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            output = model(inputs)  # Forward pass
            loss = loss_fn(output, targets)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            
            training_loss += loss.item()  # Accumulate training loss
        
        training_loss /= len(train_loader)  # Average training loss
        
        # Validation phase
        model.eval()  # Set the model to evaluation mode
        num_correct = 0
        num_examples = 0
        
        with torch.no_grad():  # No need to compute gradients during validation
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                output = model(inputs)  # Forward pass
                loss = loss_fn(output, targets)  # Compute validation loss
                valid_loss += loss.item()  # Accumulate validation loss
                
                # Compute accuracy
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
        
        valid_loss /= len(val_loader)  # Average validation loss
        
        # Print training and validation results for each epoch
        print(f'Epoch: {epoch + 1}/{epochs}, Training Loss: {training_loss:.2f}, 'f'Validation Loss: {valid_loss:.2f}, 'f'Accuracy: {num_correct / num_examples:.2f}')
    torch.save(model, "./model/simple_model")

train(model, optimizer, torch.nn.CrossEntropyLoss(),train_data_loader, test_data_loader)


