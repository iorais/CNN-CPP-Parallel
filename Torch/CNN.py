import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
        self.fc = nn.Linear(2 * 6 * 6, 10, bias=True)
        torch.nn.init.constant_(self.fc.bias, 1.0)  # Set bias to 1.0

    def forward(self, x):
        # Layer 1: Convolution + ReLU
        x = F.relu(self.conv1(x))
        # Layer 2: Convolution + ReLU
        x = F.relu(self.conv2(x))
        # Flatten the tensor for the dense layer
        x = x.view(x.size(0), -1)
        # Layer 3: Dense + Softmax
        x = F.softmax(self.fc(x), dim=1)
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
    

def main():
    # Hyperparameters
    learning_rate_conv = 0.5
    learning_rate_fc = 0.5
    num_epochs = 100
    batch_size = 32

    # Initialize the model, criterion, and optimizer
    model = CNN()

    # Split optimizer for convolution and dense layers
    optimizer = optim.SGD([
        {'params': model.conv1.parameters(), 'lr': learning_rate_conv},
        {'params': model.conv2.parameters(), 'lr': learning_rate_conv},
        {'params': model.fc.parameters(), 'lr': learning_rate_fc}
    ])

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

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

    # Training loop
    for epoch in range(num_epochs):
        for inputs, targets in trainLoader:
            inputs, targets = inputs.to(device), targets.to(device)


            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

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
        for inputs, targets in testLoader: 
            # Move data to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            # Compute loss
            loss = criterion(outputs, targets.squeeze())
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

if __name__ == "__main__":
    main()