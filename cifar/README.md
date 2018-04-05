# Training SkipNet on CIFAR-10 and CIFAR-100 

This folder contains the training code and trained models for original ResNet, SkipNet+SP and SkipNet+HRL+SP. 

## Prerequisite 
This code requires `Pytorch 2.0` and for the RL training, we only support single GPU (multi-GPU implementation will be 
included in the code for ImageNet). 

To install Pytorch, please refer to the docs at the [Pytorch website](http://pytorch.org/).


## Preparation
### Model Architecture
`models.py` includes the model definations and configurations for all architectures on the CIFAR datasets used in the paper.
We use the model config name to specify the model arch when training.  

For naming, 

- `cifar{10,100}_resnet_{#layers}`: original ResNets without routing modules. For example, `cifar10_resnet_38` means ResNet-38
on the CIFAR-10 dataset.
- `cifar{10,100}_feedforward_{#layers}`: SkipNet+SP with feedforward gates. 
- `cifar{10,100}_rnn_gate_{#layers}`: SkipNet+SP with recurrent gates.
- `cifar{10,100}_feedforward_rl_{#layers}`: SkipNet+HRL+SP with feedforward gates
- `cifar{10,100}_rnn_gate_rl_{#layers}`: SkipNet+HRL+SP with recurrent gates.

### Data Preparation
`data.py` includes the data preparation for the CIFAR-10 and CIFAR-100 datasets. 


## Demo

We provide model checkpoints trained with supervised pretraining (SP) and 
hybrid reinforcement learning (HRL) for ResNet-38, ResNet-74, ResNet-110 
with recurrent gate design on CIFAR-10 as follows. The checkpoints trained 
with SP are used as initialization for HRL stage. 

* [ResNet-38-SP](http://people.eecs.berkeley.edu/~xinw/skipnet/resnet-38-rnn-sp-cifar10.pth.tar) and [ResNet-38-HRL](http://people.eecs.berkeley.edu/~xinw/skipnet/resnet-38-rnn-cifar10.pth.tar)
* [ResNet-74-SP](http://people.eecs.berkeley.edu/~xinw/skipnet/resnet-74-rnn-sp-cifar10.pth.tar) and [ResNet-74-HRL](http://people.eecs.berkeley.edu/~xinw/skipnet/resnet-74-rnn-cifar10.pth.tar)
* [ResNet-110-SP](http://people.eecs.berkeley.edu/~xinw/skipnet/resnet-110-rnn-sp-cifar10.pth.tar) and [ResNet-110-HRL](http://people.eecs.berkeley.edu/~xinw/skipnet/resnet-110-rnn-cifar10.pth.tar)

To evaluate the trained models,  you can download the model checkpoints and 
run the following commands. Take ResNet-110 as an example, 

```angular2html
python3 train_rl.py test cifar10_rnn_gate_rl_110 --resume resnet-110-rnn-cifar10.pth.tar --gate-type rnn
```

The expected results are 

|Model | Train Scheme | Top 1 Accuracy (%) | Computation Percentage (%)| FLOPs reduced % |
|-----------| :--------: | :------------------:| :---------------------:| :-----:|
| ResNet-38  | HRL |  92.750 | 70.272 | 29.728 |
| ResNet-74  | HRL |  92.770 | 52.340 | 47.660 |
| ResNet-110 | HRL |  93.300 | 49.526 | 50.474 |



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
python3 train_rl.py train [ARCH] -d [DATASET] --resume [SP-CHECKPOINT] --alpha [ALPHA] --gate-type [GATE-TYPE]
```

To test the model, run 
```
python3 train_rl.py test [ARCH] -d [DATASET] --resume [HRL-CHECKPOINT] --gate-type [GATE-TYPE]
```









