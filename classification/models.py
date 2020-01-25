import torch.nn as nn
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)
from torchmeta.modules.utils import get_subdict

from utils import swish

class MultiLayerPerceptron(MetaModule):
    def __init__(self, in_dim, out_dim, num_layers=3, hidden_size=64, nonlinearity="relu"):
        super(MultiLayerPerceptron, self).__init__()
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.out_dim = out_dim

        if nonlinearity == "relu":
            self.activation = nn.ReLU
        elif nonlinearity == "swish":
            self.activation = swish
        elif nonlinearity == "sigmoid":
            self.activation = nn.sigmoid
        else:
            raise()

        self.layer_list = [
            nn.Flatten(),
            nn.Linear(in_dim, hidden_size),
            self.activation()
        ]
        for _ in range(num_layers):
            self.layer_list.extend([
                nn.Linear(hidden_size, hidden_size),
                self.activation()
            ])

        # Should be able to add variable layers
        self.features = MetaSequential(
            *self.layer_list
        )

        self.classifier = MetaLinear(hidden_size, out_dim)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=get_subdict(params, 'classifier'))
        return logits

def conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvolutionalNeuralNetwork(MetaModule):
    def __init__(self, in_channels, out_features, num_layers=3, hidden_size=64):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        # Should be able to add variable layers
        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

        self.classifier = MetaLinear(hidden_size, out_features)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=get_subdict(params, 'classifier'))
        return logits
