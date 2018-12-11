import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleVGG(nn.Module):
    def __init__(self, image_dims, num_features=84, num_classes=10, model_version='vgg11'):
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
        self.fc1 = nn.Linear(self.output_feature_dims, self.output_feature_dims//2)
        self.fc2 = nn.Linear(self.output_feature_dims//2, num_features)
        self.fc3 = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = F.relu(self.feature_vector(x))
        x = self.fc3(x)
        return x

    def feature_vector(self, x):
        x = F.relu(self.feature_extractor(x))
        x = x.view(-1, self.output_feature_dims)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x