
# Training SkipNet on the ImageNet dataset

This folder contains the training code for original ResNet, SkipNet+SP and SkipNet+HRL+SP. 

## Prerequisite 
We support training with multiple GPUs with Pytorch 3.0. The requirement is the batch size is dividable
by the number of GPU used.

To install Pytorch, please refer to the docs at the [Pytorch website](http://pytorch.org/).

To prepare ImageNet dataset, please follow this [link](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset).

## Preparation
### Model Architecture
`models.py` includes the model definations and configurations for all architectures on the ImageNet datasets. We only 
use recurrent gates on the ImageNet dataset. We use the model config name to specify the model arch when training.  

For naming, 

- `resnet_{#layers}`: original ResNets without routing modules.
- `imagenet_rnn_gate_{#layers}`: SkipNet+SP with recurrent gates.
- `imagenet_rnn_gate_rl_{#layers}`: SkipNet+HRL+SP with recurrent gates.


## Demo 
We provide model checkpoints trained with [SP](http://people.eecs.berkeley.edu/~xinw/skipnet/resnet-101-rnn-sp-imagenet.pth.tar) and [HRL+SP](http://people.eecs.berkeley.edu/~xinw/skipnet/resnet-101-rnn-imagenet.pth.tar) 
on ResNet-101. 

To run this demo, you can first download the checkpoints and then run the following 
commands.

For just supervised pretraining (SP), run
```
python3 train_sp.py test imagenet_rnn_gate_101 -d [DATASET] --resume resnet-101-rnn-sp-imagenet.pth.tar
```
The expected results are 
```
Prec@1=77.486%, Prec@5 93.620%, Computation Percentage=84.976%, FLOPs reduction=15.024%
```

We also provide the trained checkpoint of ResNet-50 with supervised pretraining [here](http://people.eecs.berkeley.edu/~xinw/skipnet/resnet-50-rnn-sp-imagenet.pth.tar).
These checkpoints were used as initialization for training ResNet with hybrid reinforcement learning.

For hybrid reinforcement learning (HRL), run
```
python3 train_rl.py test imagenet_rnn_gate_rl_101 -d [DATASET] --resume resnet-101-rnn-imagenet.pth.tar
```

The expected results are 
```
Prec@1=76.942%,  Prec@5=93.420%,  Computation Percentage=70.058%, FLOPs reduction=29.942%
```

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
python3 train_rl.py train [ARCH] -d [DATASET] --resume [SP-CHECKPOINT] --alpha [ALPHA]
```

To test the model, run 
```
python3 train_rl.py test [ARCH] -d [DATASET] --resume [HRL-CHECKPOINT]
```
