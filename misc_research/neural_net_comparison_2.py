import torch
import torch.nn as nn
import torch.optim as optim

class FullyLinearNN(nn.Module):
    def __init__(self, input_dim=768):
        super(FullyLinearNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 10)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        return x

if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_dimension = 768

    # Instantiate network 1 (teacher) with random Gaussian weights
    network1 = FullyLinearNN(input_dim=input_dimension).to(device)
    network1.eval() # Teacher network is not training

    # Instantiate network 2 (student) with other random Gaussian weights
    network2 = FullyLinearNN(input_dim=input_dimension).to(device)

    print("Network 1 initialized:")
    # print(network1)
    print("Network 2 initialized:")
    # print(network2)

    # Hyperparameters for training network2
    learning_rate = 0.01 # Adjusted for SGD
    num_epochs = 100000
    batch_size = 64

    # Loss and optimizer
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(network2.parameters(), lr=learning_rate) # Old Adam optimizer
    optimizer = optim.SGD(network2.parameters(), lr=learning_rate, momentum=0.9) # Switched to SGD with momentum

    # Calculate and print initial loss before training
    with torch.no_grad():
        initial_inputs = torch.randn(batch_size, input_dimension).to(device)
        initial_targets = network1(initial_inputs)
        initial_outputs = network2(initial_inputs)
        initial_loss = criterion(initial_outputs, initial_targets)
        print(f"\nInitial Loss before training: {initial_loss.item():.6f}")

    print("\nStarting training: Network 2 learns from Network 1...")

    for epoch in range(num_epochs):
        # Generate random input data for this batch
        # Input dimensions: (batch_size, input_dimension)
        inputs = torch.randn(batch_size, input_dimension).to(device)

        # Get targets from network1 (teacher)
        with torch.no_grad(): # Ensure no gradients are computed for network1
            targets = network1(inputs)

        # Forward pass for network2 (student)
        outputs = network2(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization for network2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

    print("\nTraining finished.")

    # Example: Test with a new random input
    test_input = torch.randn(1, input_dimension).to(device)
    with torch.no_grad():
        output_network1 = network1(test_input)
        output_network2 = network2(test_input)

    print("\nTest after training:")
    print(f"Input: A random {input_dimension}-dim vector")
    print(f"Network 1 (Teacher) Output: \n{output_network1.cpu().numpy()}")
    print(f"Network 2 (Student) Output: \n{output_network2.cpu().numpy()}")
    final_loss = criterion(output_network2, output_network1)
    print(f"Final MSE between student and teacher on this test input: {final_loss.item():.6f}")

    # You can further verify by checking if the weights of network2 are close to network1
    # if the architecture is simple enough and optimization is perfect.
    # For deeper/more complex networks, matching outputs is the primary goal. 