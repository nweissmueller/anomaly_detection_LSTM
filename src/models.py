from torch import nn


class encoder(nn.Module):
    """
    encoder: initializes 3 LSTM RNNs, with input size and hidden size as
    function of embedding dimension.

    forward: implements forward pass through rnn layers

    output: hidden layer values of the encoder output
    """

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim1, self.hidden_dim2 = embedding_dim, 2 * embedding_dim, 4 * embedding_dim

        # first LSTM layer
        self.rnn1 = nn.LSTM(input_size=n_features,
                            hidden_size=self.hidden_dim2,
                            num_layers=1,
                            batch_first=True)

        # second LSTM layer
        self.rnn2 = nn.LSTM(input_size=self.hidden_dim2,
                            hidden_size=self.hidden_dim1,
                            num_layers=1,
                            batch_first=True)

        # third LSTM layer
        self.rnn3 = nn.LSTM(input_size=self.hidden_dim1,
                            hidden_size=embedding_dim,
                            num_layers=1,
                            batch_first=True)

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (_, _) = self.rnn2(x)
        x, (hidden_n, _) = self.rnn3(x)

        return hidden_n.reshape((self.n_features, self.embedding_dim))


class decoder(nn.Module):
    """
    decoder: initializes 3 LSTM RNNs, with input size and hidden size as
    function of embedding dimension and 1 linear layer that generates a 1-dim vector

    forward: implements forward pass through rnn layers and linear layer

    output: 1-dimensional reconstructed values from encoder input
    """

    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim1, self.hidden_dim2, self.n_features = 2 * input_dim, 4 * input_dim, n_features

        # first LSTM layer
        self.rnn1 = nn.LSTM(input_size=input_dim,
                            hidden_size=input_dim,
                            num_layers=1,
                            batch_first=True)

        # second LSTM layer
        self.rnn2 = nn.LSTM(input_size=input_dim,
                            hidden_size=self.hidden_dim1,
                            num_layers=1,
                            batch_first=True)

        # third LSTM layer
        self.rnn3 = nn.LSTM(input_size=self.hidden_dim1,
                            hidden_size=self.hidden_dim2,
                            num_layers=1,
                            batch_first=True)

        self.output_layer = nn.Linear(self.hidden_dim2, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x, (hidden_n, cell_n) = self.rnn3(x)
        x = x.reshape((self.seq_len, self.hidden_dim2))

        return self.output_layer(x)


class LSTM_AE(nn.Module):
    """
    Initializes encoder and decoder with given dimensions on device.

    forward: implements forward pass of input through encoder and decoder

    output: returns decoder output
    """

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTM_AE, self).__init__()
        self.encoder = encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x