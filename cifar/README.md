# Training SkipNet on CIFAR-10 and CIFAR-100 

This folder contains the training code for original ResNet, SkipNet+SP and SkipNet+HRL+SP. 

## Prerequisite 
This code requires `Pytorch 2.0` and for the RL training, we only support single GPU (multi-GPU implementation will be 
included in the code for ImageNet). 

To install Pytorch, please refer to the docs at the [Pytorch website](http://pytorch.org/).


## Preparation
### Model Architecture
`models.py` includes the model definations and configurations for all architectures on the CIFAR datasets used in the paper.
We use the model config name to specify the model arch when training.  

For naming, 

- `cifar{10,100}\_resnet\_{#layers}`: original ResNets without routing modules. For example, `cifar10_resnet_38` means ResNet-38
on the CIFAR-10 dataset.
- `cifar{10,100}\_feedforward\_{#layers}`: SkipNet+SP with feedforward gates. 
- `cifar{10,100}\_rnn\_gate\_{#layers}`: SkipNet+SP with recurrent gates.
- `cifar{10,100}\_feedforward\_rl\_{#layers}`: SkipNet+HRL+SP with feedforward gates
- `cifar{10,100}\_rnn\_gate\_rl\_{#layers}`: SkipNet+HRL+SP with recurrent gates.

### Data Preparation
`data.py` includes the data preparation for the CIFAR-10 and CIFAR-100 datasets. 

## Training 

### Train original ResNets
`train_base.py` includes the training code with default hyper-parameters the same as the original [ResNet paper](https://arxiv.org/pdf/1512.03385.pdf).

To train the model, run  
```
python3 train_base.py train [ARCH] -d [DATASET] 
```

To test the trained model, run
```
python3 train_base.py test [ARCH] -d [DATASET] --resume [CHECKPOINT]
```

### Train SkipNet+SP
The supervised pre-training stage is to obtain a good intialization for the policy learning with hybrid reinforcement 
learning(HRL). `train_sp.py` includes the code to train the SkipNet+SP. By default, the code will save the checkpoints to 
`save_checkpoints/[ARCH]` which will be needed in HRL stage. 

To train the model, run 
```
python3 train_sp.py train [ARCH] -d [DATASET] 
```

To test the trained model, run
```
python3 train_sp.py test [ARCH] -d [DATASET] --resume [CHECKPOINT]
```

### Train SkipNet+HRL+SP
`train_rl.py` includes the code to learn the routing policy with hybrid objective function optimized with REINFORCE. As 
proposed in the paper, we use the trained SkipNet+SP as initialization. 

To train the model, run 
```
python3 train_rl.py train [ARCH] -d [DATASET] --resume [SP-CHECKPOINT]
```

To test the model, run 
```
python3 train_rl.py test [ARCH] -d [DATASET] --resume [HRL-CHECKPOINT]
```









