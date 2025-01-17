import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter


def evaluate(loader, model, dimensions, device, loss_fn):
    model.eval()
    total, correct = 0, 0
    n_eval = 0
    total_loss = 0

    with torch.no_grad():

        for data in tqdm(loader):
            inputs, labels = data
            inputs = torch.reshape(inputs, dimensions).to(device)  # move to gpu
            labels = labels.to(device)  # move to gpu
            predictions = model(inputs)
            current_loss = loss_fn(predictions, labels)
            total_loss += current_loss
            total += len(labels)
            predictions = predictions.argmax(axis=1)
            correct += (predictions==labels).sum().item()
            n_eval += 1

    model.train()
    return correct / total, total_loss / n_eval



def starting_train(train_dataset, val_dataset, dimensions, model, hyperparameters, n_eval, device):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    #optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss().to(device)  # move to gpu

    # a learning rate of 0.1 is an ok place to start, but you should play around with different learning rates!
    # also a small weight_decay (l2 regularization) greater than 0 is a good idea to prevent overfitting
    optimizer = optim.SGD([{'params': model.fc1.parameters()}, # use default learning rate of 0.1 for fc layers
                            {'params': model.fc2.parameters()}, # use default learning rate of 0.1 for fc layers
                            {'params': model.model_a.parameters(), 'lr': 1e-3} # use smaller learning rate of 0.001 for the encoder (resnet)
                            ], lr=0.1, momentum=0.9, weight_decay=5e-4)

    # this lr scheduler multiplies the lr by 0.1 after 20, and 40 epochs respectively
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    
    train_total, train_correct, train_loss, train_num = 0, 0, 0, 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for data in tqdm(train_loader):
            batch_inputs, batch_labels = data
            batch_inputs = batch_inputs.to(device)  # move to gpu
            batch_labels = batch_labels.to(device)  # move to gpu

            optimizer.zero_grad()
            predictions = model(batch_inputs)
            current_loss = loss_fn(predictions, batch_labels)

            predictions = predictions.argmax(axis=1)
            train_correct += (predictions==batch_labels).sum().item()
            train_total += len(batch_labels)
            train_loss += current_loss
            train_num += 1

            current_loss.backward()
            optimizer.step()

            # learning rate scheduling
            # lr_scheduler.step()
        
        val_acc, val_loss = evaluate(val_loader, model, dimensions, device, loss_fn)
        print(f"Training accuracy: {train_correct / train_total}   Loss: {train_loss / train_num}")
        print(f"Validation accuracy: {val_acc}   Loss: {val_loss}")