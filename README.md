# OLE-GRSV-Pytorch

Pytorch implementation of the [OLE-GRSV](https://arxiv.org/abs/1805.07291) loss function for classification.
This loss function tries to embed each feature vector in an orthogonal subspace relative to feature vectors of other categories while validating against samples that don't directly influence subspace assignment.
Unlike the training procedure described in the paper, I had to clip the gradients to regulate some issues with exploding gradients (the paper used weight regularization, which didn't help the gradient issues in my case).
I've not yet been able to reproduce the accuracy on CIFAR 10 with VGG 11, but I'm likely not using the proper optimizer, etc.
Also, there are likely a few efficiency improvements that could be made.

* [Installation](#installation)
* [Requirements](#requirements)
* [Usage](#usage)
  * [Loss](#loss)
  * [Experiments](#experiments)
* [References](#references)

<a name="installation"/>

## Installation

There is no need to pip-install anything to use this package (aside from the [requirements](#requirements)).
Clone the repo:
```bash
git clone https://github.com/jhetherly/OLE-GRSV-Pytorch.git
```
Then, put the `ole_grsv.py` Python file wherever you need access to this loss function.

<a name="requirements"/>

## Requirements

If you just need the loss function:

* Python 3.6.8
* Pytorch 1.0

If you want to reproduce my experiments (in addition to the above):

* TorchVision 0.2.1
* Numpy 1.15
* Scikit-learn 0.20
* h5py 2.9

<a name="usage"/>

## Usage

There are two components to this repository: the loss and my own experiments.

<a name="loss"/>

### Loss

The loss function is self-contained in the `ole_grsv.py` Python file.


<a name="experiments"/>

### Experiments

<a name="references"/>

## References
