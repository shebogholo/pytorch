#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import torch.nn as nn


# In[20]:


class Model(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dimension, output_dimension)  

    def forward(self, x):
        out = self.linear(x)
        return out
    
model = Model(28*28, 10)


# ## 1. torch.autograd

# `autograd` is a Pytorch package that we can use for `automatic differentiation` to automate the computation of backward passes in a Neural Network. Using `autograd,` the forward pass define a `computation graph` with nodes in the grapth being Tensors and edges being functions that produce output. 

# In[ ]:





# ## 2. torch.nn

# `torch.nn` is a Pytorch module for creating and training neural networks. It provides `Containers`, `Convolutional layers`, `Recurrent layers`, `Pooling layers`, `Padding layers`, `Non-linear activations`, `Loss functions`, `Normalization layers`, `Dropout layers` and `Utilities` among others.

# In[ ]:





# In[ ]:





# ## 3. torch.optim

# `torch.optim` package provides an interface for common optimization algorithms.

# <img src='images/optim.png' alt='Optimization package'/>

# In[17]:


optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, betas=(0.6,0.8), eps=1e-08)
optimizer = torch.optim.Adadelta(model.parameters(), lr=0.001, rho=0.9, eps=1e-05)


# Extras [Optimization Algorithms](https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/optimizers/)  [Learning Rate Scheduling](https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/lr_scheduling/)

# ## 5. torch.utils

# #### 5.1 Dataset

# In[ ]:





# #### 5.2 DataLoader

# In[ ]:





# ## 6. torch.random

# In[ ]:





# ## 7. TensorBoard

# In[ ]:





# ## 8. torch.jit

# In[ ]:





# ## References

# - [CS 231n CNN for Visual Recognition](http://cs231n.github.io/)
# - [PyTorch JavaPoint](https://www.javatpoint.com/pytorch)
# - [Deep Learning Wizard](https://www.deeplearningwizard.com/deep_learning/intro/)

# In[ ]:




