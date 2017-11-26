# SkipNet: Learning Dynamic Routing in Convolutional Networks

This repository contains the code for [SkipNet paper]() under review at CVPR 2018.

SkipNet learns to route images through a sub-set of layers on a per-input basis. Challenging images are routed through more
layers than easy images. We talk about two model designs with either feedforward gates and reccurent gates which enables 
different levels of parameter sharing in the paper.  The model illustrations are as follows.
<p float="left">
  <img src="figs/skipnet_ff_structure.jpg" width="400" alt="SkipNet with feedforward gates" />
  <img src="figs/skipnet_rnn_structure.jpg" width="400" alt="SkipNet with recurrent gates"  /> 
</p>


## SkipNet on CIFAR datasets
All the model configuration and training/evaluation code are available under `./cifar`. If you want to train your own 
SkipNet, you can also follow the [document](cifar/README.md) under the same folder. 

## SkipNet on ImageNet datasets 
TBD




