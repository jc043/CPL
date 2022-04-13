# CPL: Continual Predictive Learning from Videos
A PyTorch implementation of our paper: Continual Predictive Learning from Videos. (The code is a bit messy right now, we will provide a cleaner version in the future.)

## Introduction
In this paper, we study a new continual learning problem in
the context of video prediction, and observe that most existing methods suffer from severe catastrophic forgetting in
this setup. To tackle this problem, we propose the continual
predictive learning (CPL) approach, which learns a mixture world model via predictive experience replay and performs test-time adaptation with non-parametric task inference.  Our approach is shown to effectively mitigate forgetting and remarkably outperform the naïve combinations of previous
art in video prediction and continual learning.
## Get Started

1. Install Python 3.8, PyTorch 1.9.0 for the code. 

2. Download data. [KTH action dataset](https://cloud.tsinghua.edu.cn/d/7d19372a621a4952b738/)

3. Train the CPL model. You can use train.sh/test.sh to train/test the CPL model. The learned model will be saved in the `--save_dir` folder. The generated future frames will be saved in the `--gen_frm_dir` folder.
```
cd script/
sh train.sh
```

4. To train the base model, use the following script.
```
cd script/
sh train_base.sh
```

## Acknowledgement
We appreciate the following github repos where we borrow code from:

https://github.com/thuml/predrnn-pytorch

https://github.com/AntixK/PyTorch-VAE
