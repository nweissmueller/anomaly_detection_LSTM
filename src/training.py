import torch
from src.model_definitions import LSTM_AE


def train_model(model, train_dataset, val_dataset, n_epochs):
    """
    Trains model over n_epochs, keeps best model weights. Initializes optimizer, criterion,
    and model history dictionary.

    input: model architecture, taining dataset (normal cases), validation dataset (normal cases)
    number of epochs.

    outputs: trained model, and model training and validation error dictionary
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []

        # compute trainingset loss
        for seq in train_dataset:
            optimizer.zero_grad()
            seq = seq.to(device)
            seq_pred = model(seq)
            loss = criterion(seq_pred, seq)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # compute validation set loss
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for seq in val_dataset:
                seq = seq.to(device)
                seq_pred = model(seq)
                loss = criterion(seq_pred, seq)
                val_losses.append(loss.item())

        # compute mean training and validation loss and add to dict
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        # keep model weights if validation loss is lower than best previous value
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0:
            print(
                f'Epoch {epoch}: train loss {train_loss:.5f} val loss {val_loss:.5f}'
            )

    model.load_state_dict(best_model_wts)
    return model.eval(), history


def predict(model, dataset):
    """
    Generates predictions by applying the model to a dataset

    input: trained LSTM-model, time-series dataset
    output: predicted values, and cumulative L1 loss for the time-series
    """
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(device)

    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())

    return predictions, losses