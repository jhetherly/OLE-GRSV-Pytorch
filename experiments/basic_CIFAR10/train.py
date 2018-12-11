import os
import json
from datetime import datetime

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
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


experiment_settings = {
    "rng": 1234567890, 
    "artifacts_path": "artifacts",
    "settings_filename": 'settings.json',
    "model": {
        "version": 'vgg11',
        "feature_size": 20,
    },
    "loss": {
        "min_singular_value_fraction": 0.1,
        "lambda": 5.0,
        "epsilon": 1e-3,
    },
    "training": {
        "checkpoint_filename": 'vgg11.ckpt',
        "batch_size": 10,
        "epochs": 300,
        "gradient_clipping_max_norm": 0.5,
        "use_out_of_set_validation": False,
        "val_every": 5,
    },
    "validation": {
        "batch_size": 5,
        "fraction": 0.1,
    },
    "testing": {
        "batch_size": 5,
    },
}

save_path = os.path.join(experiment_settings["artifacts_path"],
                         "train_{:%Y-%m-%d-%H-%M-%S}".format(datetime.now()))
os.makedirs(save_path, exist_ok=True)
logger = setup_logger('train_cifar10', log_file=os.path.join(save_path, 'train.log'))

np_rng = np.random.RandomState(experiment_settings['rng'])

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif hasattr(m, 'weight'):
        torch.nn.init.xavier_uniform_(m.weight)

base_transform = transforms.Compose(
    [
    #  transforms.Resize(size=224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])
training_transform = transforms.Compose(
    [
     transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1, hue=0.1),
     transforms.RandomAffine(degrees=15, shear=10),
     base_transform,
     ])
image_dims = (32, 32)

trainloader, valloader, testloader = create_CIFAR10_dataloaders(
    base_transform, training_transform,
    experiment_settings['validation']['fraction'],
    experiment_settings['training']['batch_size'],
    experiment_settings['validation']['batch_size'],
    experiment_settings['testing']['batch_size'],
    np_rng)

val_every = experiment_settings['training']['val_every']
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

ole_grsv = OLEGRSV(classes=range(len(classes)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


model = SimpleVGG(image_dims, num_features=experiment_settings['model']['feature_size'],
                  num_classes=len(classes), model_version=experiment_settings['model']['version'])
model.to(device)

model.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=experiment_settings['training']['epochs']//3, gamma=0.1)
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
#     factor=np.power(0.1, 1/3), patience=20//val_every, verbose=True)

model_save_filename = os.path.join(save_path,
    experiment_settings['training']['checkpoint_filename'])
settings_save_filename = os.path.join(save_path,
    experiment_settings['settings_filename'])
logger.info('Saving model to {}'.format(model_save_filename))
logger.info('Saving settings to {}'.format(settings_save_filename))
with open(settings_save_filename, 'w') as outfile:
    json.dump(experiment_settings, outfile)

print_every = 200
n_examples_per_class_g = experiment_settings['training']['batch_size']
n_examples_per_class_v = experiment_settings['validation']['batch_size']
geom_list = []
for iclass in range(len(classes)):
    for iexample in range(n_examples_per_class_g):
        if iexample < n_examples_per_class_g//2:
            geom_list.append(True)
        else:
            geom_list.append(False)
geom_mask = torch.Tensor(geom_list) == True
val_mask = torch.Tensor(geom_list) == False
geom_mask = geom_mask.to(device)
val_mask = val_mask.to(device)
min_singular_value_fraction = experiment_settings['loss']['min_singular_value_fraction']
valloader_iter = iter(valloader)
epsilon = torch.Tensor([experiment_settings['loss']['epsilon']]).to(device)
lambda_ = experiment_settings['loss']['lambda']
gradient_clipping_max_norm = experiment_settings['training']['gradient_clipping_max_norm']
use_out_of_set_validation = experiment_settings['training']['use_out_of_set_validation']
min_metric = np.finfo(float).max
max_metric = np.finfo(float).min
for epoch in range(experiment_settings['training']['epochs']):  # loop over the dataset multiple times
    model.train()

    running_loss = 0.0
    count = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        if use_out_of_set_validation:
            try:
                vdata = valloader_iter.next()
            except StopIteration:
                valloader_iter = iter(valloader)
                vdata = valloader_iter.next()
            except:
                raise
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            ginputs, glabels = inputs, labels
        else:
            ginputs, glabels = inputs[geom_mask], labels[geom_mask]
            vinputs, vlabels = inputs[val_mask], labels[val_mask]


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        z_g = model.feature_vector(ginputs)
        z_v = model.feature_vector(vinputs)

        loss = ole_grsv(z_g, glabels, z_v, vlabels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),
            max_norm=gradient_clipping_max_norm,
            norm_type=2)
        optimizer.step()

        if use_out_of_set_validation:
            running_loss += loss.item()
        else:
            running_loss += 0.5*loss.item()

            # NOTE: switch val and geom batches
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            z_g = model.feature_vector(ginputs)
            z_v = model.feature_vector(vinputs)

            loss = ole_grsv(z_v, vlabels, z_g, glabels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),
                max_norm=gradient_clipping_max_norm,
                norm_type=2)
            optimizer.step()

            running_loss += 0.5*loss.item()

        # print statistics
        if i % print_every == print_every - 1:    # print every 200 mini-batches
            logger.info('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss/(i + 1)))

        count += 1


    if epoch == 0 or epoch + 1 == experiment_settings['training']['epochs'] or (epoch + 1) % val_every == 0:
        model.eval()
        _, projections = compute_projection_matrices(trainloader, model,
            experiment_settings['loss']['min_singular_value_fraction'])
        projections = torch.from_numpy(projections).to(device)

        y_true_val, y_pred_val = evaluate(valloader, model, projections, ole_grsv)
        val_accuracy = accuracy_score(y_true_val, y_pred_val)
        logger.info('validation confusion matrix\n{}'.format(confusion_matrix(y_true_val, y_pred_val)))
        if max_metric < val_accuracy:
            max_metric = val_accuracy
            logger.info('new best validation accuracy found ({}), saving checkpoint'.format(val_accuracy))
            model.cpu()
            torch.save(model.state_dict(), model_save_filename)
            model.to(device)

        # lr_scheduler.step(val_accuracy)

    lr_scheduler.step()

logger.info('Finished Training')
