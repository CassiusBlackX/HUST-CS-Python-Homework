from torch import nn

class AutoFCNet(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes):
        super(AutoFCNet, self).__init__()
        self.layers = nn.ModuleList()
        last_size = input_size
        for size in hidden_sizes:
            self.layers.append(nn.Linear(last_size, size))
            self.layers.append(nn.ReLU())
            last_size = size
        # output
        self.layers.append(nn.Linear(last_size, num_classes))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
