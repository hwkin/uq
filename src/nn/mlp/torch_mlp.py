import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, depth, act, dropout=False, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.act = act
        self.dropout = nn.Dropout(p=dropout_rate) if dropout else None

        # input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        
        # hidden layers
        for _ in range(depth - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # output layer
        self.layers.append(nn.Linear(hidden_size, num_classes))

    def forward(self, x, final_act=False):
        # hidden layers: Linear -> Activation -> Dropout
        for i in range(len(self.layers) - 1):
            x = self.act(self.layers[i](x))
            if self.dropout is not None:
                x = self.dropout(x)
        
        # output layer (no activation or dropout by default)
        x = self.layers[-1](x)
        if final_act:
            x = torch.relu(x)
        return x
