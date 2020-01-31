import torch.nn as nn
from torchmeta.modules import (
    MetaModule,
    MetaSequential,
    MetaConv2d,
    MetaBatchNorm2d,
    MetaLinear,
)
from torchmeta.modules.utils import get_subdict


class MultiLayerPerceptron(MetaModule):
    def __init__(self, in_dim, out_dim, hidden_dim=40):
        super(MultiLayerPerceptron, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.output = MetaSequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, inputs, params=None):
        output = self.output(inputs, params=get_subdict(params, "output"))
        return output
