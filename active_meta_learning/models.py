import torch.nn as nn
from torchmeta.modules import MetaModule, MetaSequential
from torchmeta.modules.utils import get_subdict


class MultiLayerPerceptron(MetaModule):
    def __init__(self, in_dim, out_dim, hidden_dim=40, num_layers=3):
        super(MultiLayerPerceptron, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Build layers
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append([nn.Linear(hidden_dim, out_dim)])

        self.output = MetaSequential(*layers)

    def forward(self, inputs, params=None):
        output = self.output(inputs, params=get_subdict(params, "output"))
        return output
