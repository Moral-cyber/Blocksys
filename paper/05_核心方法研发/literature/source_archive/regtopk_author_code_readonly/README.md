# RegTopK
SourceCodes of the paper "Regularized Top-k: A Bayesian Framework for Gradient Sparsification"

## Modules
There are three modules, namely DistLearn.py and DistLearnLinearReg.py and SymthDataSet. The first two modules contain distributed
learning functions used for ResNet-18 and Linear Regression, respectively. The last module further generate the synthetic dataset
used in the Linear Regression Experiment presented in section "Numerical Validation" of the main paper, as well as Section 2 of 
the supplementary file.

## Implementations
There are four main implementations: Toy_Example.py, LinearRegression.py, ResNet18-CIFAR10.py, and ImagNette_distributed.ipynb

### Toy_Example.py
Toy_Example.py implements the toy example given in Section "Preliminaries", Sub-section "Learning Rate Scaling Property", Sub-sub-section "Motivational Example".

### LinearRegression.py
It provides the main implementation of the numerical experiment presented in Section "Numerical Validation" given in the paper.

### ResNet18-CIFAR10.py
It trains the ResNet-18 on CIFAR-10 with both top-k and RegTop-k.

### ImagNette_distributed.ipynb
It fine-tunes SqueezeNet, ShuffleNetV2, MobileNetV2, EfficientNet, and ResNet152 on ImageNette using both Top-k and RegTop-k