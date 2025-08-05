import torch
import torch.nn as nn
import torch.optim as optim
import copy # For deepcopying model state

class FullyLinearNN(nn.Module):
    def __init__(self, input_dim=784):
        super(FullyLinearNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 768)
        self.layer2 = nn.Linear(768, 768)
        self.layer3 = nn.Linear(768, 768)
        self.layer4 = nn.Linear(768, 768)
        self.layer5 = nn.Linear(768, 10)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Ensure flattened input
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_dimension = 784
    network1 = FullyLinearNN(input_dim=input_dimension).to(device)
    network1.eval()
    network2 = FullyLinearNN(input_dim=input_dimension).to(device)

    print("Network 1 initialized.")
    print("Network 2 initialized.")

    initial_line_search_lr = 0.0000001 # User-defined starting LR for line search
    num_epochs = 100000
    batch_size = 64
    max_line_search_doublings = 20 # Max times to double the LR in line search

    criterion = nn.MSELoss()
    # We still need an optimizer instance for zero_grad, but step() won't be used directly
    optimizer = optim.SGD(network2.parameters(), lr=initial_line_search_lr) # LR here is just a placeholder

    # Calculate and print initial loss
    with torch.no_grad():
        initial_inputs = torch.randn(batch_size, input_dimension).to(device)
        initial_targets = network1(initial_inputs)
        initial_outputs = network2(initial_inputs)
        initial_loss_val = criterion(initial_outputs, initial_targets).item()
        print(f"\nInitial Loss before training: {initial_loss_val:.12f}")

    print("\nStarting training with line search...")

    for epoch in range(num_epochs):
        inputs = torch.randn(batch_size, input_dimension).to(device)
        
        optimizer.zero_grad()
        
        # Get targets from network1
        with torch.no_grad():
            targets = network1(inputs)

        # Calculate loss before line search for this batch
        current_outputs = network2(inputs)
        loss_before_step = criterion(current_outputs, targets)
        loss_before_step_val = loss_before_step.item()
        
        loss_before_step.backward() # Compute gradients

        # --- Custom Line Search ---
        original_params_state_dict = copy.deepcopy(network2.state_dict())
        
        best_lr_for_step = 0.0  # If this remains 0, no update improves loss
        current_best_line_search_loss = loss_before_step_val
        
        test_lr = initial_line_search_lr
        
        for i in range(max_line_search_doublings):
            # Apply prospective update based on original params for this step
            network2.load_state_dict(original_params_state_dict) # Reset to params before this line search iteration
            with torch.no_grad():
                for param in network2.parameters():
                    if param.grad is not None:
                        param.data.add_(param.grad.data, alpha=-test_lr) # p.data = p.data - test_lr * p.grad.data
            
            # Evaluate loss with this test_lr
            with torch.no_grad():
                loss_with_test_lr = criterion(network2(inputs), targets).item()

            if loss_with_test_lr < current_best_line_search_loss:
                current_best_line_search_loss = loss_with_test_lr
                best_lr_for_step = test_lr
                test_lr *= 2 # Try a larger step
            else:
                # Loss did not decrease, so the previous lr was better (or no lr was good if this is the first try)
                break 
        
        # Apply the best update found (if any)
        if best_lr_for_step > 0:
            network2.load_state_dict(original_params_state_dict) # Reset again
            with torch.no_grad():
                for param in network2.parameters():
                    if param.grad is not None:
                        param.data.add_(param.grad.data, alpha=-best_lr_for_step)
            actual_loss_after_step = current_best_line_search_loss
        else:
            # No step improved the loss, revert to original parameters for this batch
            network2.load_state_dict(original_params_state_dict)
            actual_loss_after_step = loss_before_step_val
        # --- End Custom Line Search ---

        if (epoch + 1) % 1 == 0: # Print every epoch
             print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {actual_loss_after_step:.12f}, LR used: {best_lr_for_step:.12e}')

    print("\nTraining finished.")

    # Test after training
    test_input = torch.randn(1, input_dimension).to(device)
    with torch.no_grad():
        output_network1 = network1(test_input)
        output_network2 = network2(test_input)
    final_loss_val = criterion(output_network2, output_network1).item()
    print("\nTest after training:")
    print(f"Input: A random {input_dimension}-dim vector")
    print(f"Network 1 (Teacher) Output: \n{output_network1.cpu().numpy()}")
    print(f"Network 2 (Student) Output: \n{output_network2.cpu().numpy()}")
    print(f"Final MSE on test input: {final_loss_val:.12f}")

    # You can further verify by checking if the weights of network2 are close to network1
    # if the architecture is simple enough and optimization is perfect.
    # For deeper/more complex networks, matching outputs is the primary goal. 