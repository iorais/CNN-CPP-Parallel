import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Layer 1: Convolution
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=0, bias=True)
        torch.nn.init.constant_(self.conv1.bias, 0.1)  # Set bias to 0.1
        
        # Layer 2: Convolution
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=3, stride=2, padding=0, bias=True)
        torch.nn.init.constant_(self.conv2.bias, 0.1)  # Set bias to 0.1
        
        # Layer 3: Dense
        self.fc1 = nn.Linear(72, 72, bias=True)
        torch.nn.init.constant_(self.fc1.bias, 1.0)  # Set bias to 1.0

        # Layer 4: Dense
        self.fc2 = nn.Linear(72, 10, bias=True)
        torch.nn.init.constant_(self.fc2.bias, 1.0)  # Set bias to 1.0

        # Activation Function
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.001)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten for dense layer
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def load_ubyte_data(file_path):
    """
    Load ubyte data into a PyTorch tensor.
    :param file_path: Path to the ubyte file.
    :return: PyTorch tensor with the loaded data.
    """
    with open(file_path, 'rb') as f:
        # Read header information
        magic_number = int.from_bytes(f.read(4), byteorder='big')  # Magic number
        num_items = int.from_bytes(f.read(4), byteorder='big')     # Number of items
        if magic_number == 2051:  # Image file
            num_rows = int.from_bytes(f.read(4), byteorder='big')  # Number of rows
            num_cols = int.from_bytes(f.read(4), byteorder='big')  # Number of columns
            # Read image data into a NumPy array
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, num_rows, num_cols)
        elif magic_number == 2049:  # Label file
            # Read label data into a NumPy array
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError("Invalid magic number in ubyte file!")
    
    # Convert to PyTorch tensor
    tensor_data = torch.tensor(data) # Convert to float tensor (optional)

    return tensor_data

def load_images(file_path, samples):
    images = load_ubyte_data(file_path)

    dims = images.shape
    images = images.reshape(dims[0], 1, dims[1], dims[2])
    images = images.to(dtype=torch.float)
    images /= 255.0

    return images[:samples]

def load_labels(file_path, samples):
    labels = load_ubyte_data(file_path)

    labels = labels.reshape(-1, 1)
    labels = labels.to(dtype=torch.long)

    return labels[:samples]
    

def main(args):
    # Hyperparameters
    learning_rate_conv = 0.1
    learning_rate_fc = 0.5
    num_epochs = args.num_epochs 
    batch_size = args.batch_size

    # Initialize the model, criterion, and optimizer
    model = CNN()

    # Split optimizer for convolution and dense layers
    optimizer = optim.SGD([
        {'params': model.conv1.parameters(), 'lr': learning_rate_conv},
        {'params': model.conv2.parameters(), 'lr': learning_rate_conv},
        {'params': model.fc1.parameters(), 'lr': learning_rate_fc},
        {'params': model.fc2.parameters(), 'lr': learning_rate_fc}
    ])

    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    # optimizer = optim.Adam(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()


    # Load data
    train_data = load_images("../datasets/train-images.idx3-ubyte", 60000)
    train_labels = load_labels("../datasets/train-labels.idx1-ubyte", 60000)

    trainset = TensorDataset(train_data, train_labels)
    trainLoader = DataLoader(trainset, batch_size=batch_size)

    if torch.cuda.is_available():
        print('GPU available')
        device = 'cuda'
    else:
        print("No GPU available")
        device = 'cpu'

    model.to(device)

    total = 0
    correct = 0

    # Training loop
    for epoch in range(num_epochs):
        for inputs, targets in tqdm(trainLoader, desc=f"[Training] Epoch #{epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)


            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(targets.size(0)))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            _, predicted = outputs.max(1)  # Get class with highest probability
            correct += predicted.eq(targets.squeeze()).sum().item()
            total += targets.size(0)

        train_accuracy = correct / total * 100
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        print(f"Train Accuracy: {train_accuracy:.2f}%")

    print("Training complete!")

    # Load test data
    test_data = load_images("../datasets/t10k-images.idx3-ubyte", 10000)
    test_labels = load_labels("../datasets/t10k-labels.idx1-ubyte", 10000)

    testset = TensorDataset(test_data, test_labels)
    testLoader = DataLoader(testset, batch_size=batch_size)

    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for testing
        for inputs, targets in tqdm(testLoader): 
            # Move data to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            # Compute loss
            loss = criterion(outputs, targets.view(targets.size(0)))
            test_loss += loss.item() * inputs.size(0)  # Accumulate loss
            
            # Compute accuracy
            _, predicted = outputs.max(1)  # Get class with highest probability
            correct += predicted.eq(targets.squeeze()).sum().item()
            total += targets.size(0)
    
    # Average loss and accuracy
    test_loss /= total
    test_accuracy = correct / total * 100

    print(total)
    print(correct)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse training parameters.")
    
    # Add arguments for num_epochs and batch_size
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,  # Default value if not provided
        help="Number of epochs for training."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,  # Default value if not provided
        help="Batch size for training."
    )
    
    # Parse the arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)