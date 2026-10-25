#e8_layer.py
import torch
import torch.nn as nn

class E8Layer(nn.Module):
    def __init__(self, input_dim=600, output_dim=248, activation=nn.ReLU()):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.act = activation

    def forward(self, x):
        return self.act(self.linear(x))
