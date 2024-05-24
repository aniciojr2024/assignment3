import torch
from torch import nn

def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    """
    model = nn.Linear(input_size, output_size)
    return model

def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def fit_regression_model(X, y):
    """
    Train the model for the given number of epochs.
    """
    learning_rate = 0.01  # Pick a better learning rate if needed
    num_epochs = 1000  # Pick a better number of epochs if needed
    input_features = X.shape[1]  # extract the number of features from the input `shape` of X
    output_features = y.shape[1] if len(y.shape) > 1 else 1  # extract the number of features from the output `shape` of y
    model = create_linear_regression_model(input_features, output_features)
    
    loss_fn = nn.MSELoss()  # Use mean squared error loss

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previous_loss = float("inf")
    tolerance = 1e-6  # Tolerance for stopping condition

    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        
        # Print the loss every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        
        # Stop the training if the loss is not changing much
        if abs(previous_loss - loss.item()) < tolerance:
            print(f"Stopping early at epoch {epoch}")
            break
        
        previous_loss = loss.item()

    return model, loss

