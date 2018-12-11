import os
import json
from datetime import datetime

import h5py
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms

from utils import setup_logger
from dataloaders import create_CIFAR10_dataloaders
from models import SimpleVGG
from ole_grsv import OLEGRSV
from evaluators import compute_projection_matrices, evaluate


settings_filename = '/shared/jeff/Documents/sooh/nh02/artifacts/2018-12-08-00-06-50/settings.json'
with open(settings_filename, 'r') as f:
    experiment_settings = json.load(f)
override = True

base_dir = os.path.dirname(settings_filename)
features_filename = os.path.join(base_dir, 'features.hdf5')
model_checkpoint_filename = os.path.join(base_dir, experiment_settings['training']['checkpoint_filename'])

save_path = os.path.join(experiment_settings["artifacts_path"],
                         "eval_{:%Y-%m-%d-%H-%M-%S}".format(datetime.now()))
os.makedirs(save_path, exist_ok=True)
logger = setup_logger('eval_cifar10', log_file=os.path.join(save_path, 'eval.log'))

np_rng = np.random.RandomState(experiment_settings['rng'])

transform = transforms.Compose(
    [
    #  transforms.Resize(size=224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])
image_dims = (32, 32)

trainloader, valloader, testloader = create_CIFAR10_dataloaders(transform,
    experiment_settings['validation']['fraction'],
    experiment_settings['training']['batch_size'],
    experiment_settings['validation']['batch_size'],
    experiment_settings['testing']['batch_size'],
    np_rng)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

ole_grsv = OLEGRSV(classes=range(len(classes)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


model = SimpleVGG(image_dims, num_features=experiment_settings['model']['feature_size'],
                  num_classes=len(classes), model_version=experiment_settings['model']['version'])
# model = models.resnet18(num_classes=len(classes))
model.load_state_dict(torch.load(model_checkpoint_filename))
model.eval()
model.to(device)

if not os.path.isfile(features_filename) or override:
    features_dict, transforms = compute_projection_matrices(trainloader, model,
        experiment_settings['loss']['min_singular_value_fraction'])
    
    features_file = h5py.File(features_filename, "w")
    features_group = features_file.create_group("features")
    projections_group = features_file.create_group("projections")

    for label in sorted(features_dict.keys()):
        features_group[str(label)] = features_dict[label]
        projections_group[str(label)] = transforms[label]
    transforms = torch.from_numpy(transforms).to(device)

    features_file.close()
else:
    features_file = h5py.File(features_filename, "r")
    projections_group = features_file["projections"]

    proj_dict = {}
    for label in projections_group:
        proj_dict[int(label)] = projections_group[label][:]

    transforms_tensors = []
    for label in sorted(proj_dict.keys()):
        transforms_tensors.append(torch.from_numpy(proj_dict[label]))
    transforms = torch.stack(transforms_tensors).to(device)

    features_file.close()


y_true_train, y_pred_train = evaluate(trainloader, model, transforms, ole_grsv)
logger.info('training confusion matrix\n{}'.format(confusion_matrix(y_true_train, y_pred_train)))


y_true_val, y_pred_val = evaluate(valloader, model, transforms, ole_grsv)
logger.info('validation confusion matrix\n{}'.format(confusion_matrix(y_true_val, y_pred_val)))


y_true_test, y_pred_test = evaluate(testloader, model, transforms, ole_grsv)
logger.info('testing confusion matrix\n{}'.format(confusion_matrix(y_true_test, y_pred_test)))


