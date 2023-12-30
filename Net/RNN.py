from torch import nn
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)

        # If the RNN layer returns a sequence of hidden states
        if len(out.shape) == 3:
            out = self.fc(out[:, -1, :])
        # If the RNN layer returns a single hidden state
        else:
            out = self.fc(out)

        return out
