import os

import constants
import torch
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train

def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # device = 'cpu'

    PATH = "data"

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    sample_factor = 1

    dimensions = [constants.BATCH_SIZE, 3, 224, 224]

    # Initalize dataset and model. Then train the model!
    
    # train_dataset = StartingDataset(0, (21000 * 0.8 // 32) * 32, PATH)
    # val_dataset = StartingDataset((21000 * 0.8 // 32) * 32, 21000 // 32 * 32, PATH)
    train_dataset = StartingDataset(True, PATH, sample_factor)
    val_dataset = StartingDataset(False, PATH, sample_factor)
    model = StartingNetwork()
    model = model.to(device)  # move to gpu if necessary

    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dimensions=dimensions,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
        device=device
    )

    # save model
    # torch.save(model.state_dict(), 'model.pth')

    torch.save(model, "cassavision-2model")

    # load with
    # model = StartingNetwork(args)
    # model.load_state_dict(torch.load('model.pth'))
    # model.eval()

    

if __name__ == "__main__":
    main()
