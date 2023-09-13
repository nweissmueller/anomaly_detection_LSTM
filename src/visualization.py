import seaborn as sns
import matplotlib.pyplot as plt


def plot_prediction(data, model, title, ax):
    """
    Plots predicted vs. actual time-series per graph

    input: LSTM-AE model, time-series dataset, title and axis
    output: plot
    """

    predictions, pred_losses = predict(model, [data])
    ax.plot(predictions[0], label='reconstructed')
    ax.plot(data, label='true')
    ax.set_title(f'{title} (loss: {np.around(pred_losses[0], 2)})')
    ax.legend()