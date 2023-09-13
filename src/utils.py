def create_dataset(df):
    """
    Creates stacked torch tensor from numpy df and returns dimensions.

    input: dataframe
    output: number of sequences, sequence length, number of features
    """

    sequences = df.astype(np.float32).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape

    return dataset, seq_len, n_features


def normalize_loss(losses, THRESHOLD=1):
    """
    Rescales cumulative L1 loss to range 0 to 1 relative to threshold,
    such that values below the manually selected threshold maps to 0.5 - 1.0
    and values above to 0.0 -0.5

    input: losses, manually selected threshold
    output: normalized loss
    """
    norm_x = list()
    x_min = np.min(losses)
    x_max = np.max(losses)
    for loss in losses:
        if loss >= THRESHOLD:
            # scale above, relative to threshold
            x = 0.5 * (loss - THRESHOLD) / (x_max - THRESHOLD)
        else:
            # scale below, relativce to threshold
            x = 1 - 0.5 * (loss - x_min) / (THRESHOLD - x_min)
        norm_x.append(x)
    return norm_x


def reset_model(model):
    """
    Resets the LSTM-AE model weights to enable de-novo model training

    input: LSTM-AE model
    """

    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def loss_to_prob(norm_losses_pos, norm_losses_neg):
    """
    Converts 0-1 normalized cumulative L1 loss per time-series and approximates
    probabilities by scaling loss relative to threshold.

    Returns y_predicted and y_test values for ROC curve
    """
    y_predicted_pos = norm_losses_pos
    y_test_pos = np.ones(len(y_predicted_pos))
    y_predicted_neg = norm_losses_neg
    y_test_neg = np.zeros(len(y_predicted_neg))
    y_predicted_np = np.concatenate(
        (np.array(y_predicted_pos), np.array(y_predicted_neg)), axis=0)
    y_test_np = np.concatenate((y_test_pos, y_test_neg), axis=0)
    a = torch.from_numpy(y_predicted_np)
    b = torch.from_numpy(y_test_np)

    return a.reshape(len(y_predicted_np), 1), b.reshape(len(y_test_np), 1)
