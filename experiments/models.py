import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleVGG(nn.Module):
    def __init__(self, image_dims, num_features=None, num_classes=10, model_version='vgg11', dropout=None):
        super(SimpleVGG, self).__init__()
        if model_version.lower() == 'vgg11':
            self.feature_extractor = models.vgg11(pretrained=False).features
        elif model_version.lower() == 'vgg13':
            self.feature_extractor = models.vgg13(pretrained=False).features
        elif model_version.lower() == 'vgg16':
            self.feature_extractor = models.vgg16(pretrained=False).features
        elif model_version.lower() == 'vgg19':
            self.feature_extractor = models.vgg19(pretrained=False).features
        vgg_n_pooling_layers = 5
        self.output_feature_dims = 512*(image_dims[0]//2**vgg_n_pooling_layers)*(image_dims[1]//2**vgg_n_pooling_layers)
        # NOTE: 0.163265306 is taken from the original ratio of 4096/25088
        intermediate_dim = int(0.163265306*self.output_feature_dims + 0.5)
        if num_features is None:
            num_features = intermediate_dim
        if dropout is not None:
            self._feature_vector = nn.Sequential(
                nn.Linear(self.output_feature_dims, intermediate_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, num_features),
            )
            self._classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes),
            )
        else:
            self._feature_vector = nn.Sequential(
                nn.Linear(self.output_feature_dims, intermediate_dim),
                nn.ReLU(),
                nn.Linear(intermediate_dim, num_features),
            )
            self._classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(num_features, num_classes),
            )

    def forward(self, x):
        x = self.feature_vector(x)
        return self._classifier(x)

    def feature_vector(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.output_feature_dims)
        return self._feature_vector(x)
