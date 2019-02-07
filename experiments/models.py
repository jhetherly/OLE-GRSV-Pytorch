import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleVGG(nn.Module):
    def __init__(self, image_shape, features=None, num_classes=10, model_version='vgg11', dropout=None):
        r"""
        """

        super(SimpleVGG, self).__init__()
        self.image_shape = image_shape
        self.features = features
        self.num_classes = num_classes
        self.model_version = model_version
        self.dropout = dropout
        if model_version.lower() in models.__dict__ and model_version.lower().startswith('vgg'):
            self.feature_extractor = models.__dict__[model_version.lower()](pretrained=False).features
        else:
            raise ValueError('"model_version" should be the name of a VGG torchvision model - got {}'.format(model_version))
        vgg_n_pooling_layers = 5
        self.output_conv_dim = 512*(image_shape[1]//2**vgg_n_pooling_layers)*(image_shape[2]//2**vgg_n_pooling_layers)
        if features is None:
            # NOTE: 0.163265306 is taken from the original "torchvision" ratio of 4096/25088
            intermediate_dim = int(0.163265306*self.output_conv_dim + 0.5)
            features = [intermediate_dim, intermediate_dim]
        feature_layers = []
        prev_feature = self.output_conv_dim
        for i, feature in enumerate(features):
            feature_layers.append(nn.Linear(prev_feature, feature))
            if i < len(features) - 1:
                feature_layers.append(nn.ReLU())
                if dropout is not None:
                    feature_layers.append(nn.Dropout(dropout))
            prev_feature = feature
        self._feature_vector = nn.Sequential(*feature_layers)
        if dropout is not None:
            self._classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(prev_feature, num_classes),
            )
        else:
            self._classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(prev_feature, num_classes),
            )

    def forward(self, x):
        x = self.feature_vector(x)
        return self._classifier(x)

    def feature_vector(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.output_conv_dim)
        return self._feature_vector(x)
