import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

class mySF1D_elementBased(nn.Module):    
    def __init__(self, left = -1.,  right = 1.):
        super().__init__()        
        
        self.left   = left
        self.right  = right

        # To easily transfer to CUDA or change dtype of whole model
        self.register_buffer('one', torch.tensor([1], dtype=torch.float32))

    def forward(self, x=None, training=False):  
        if training : x = (self.left + self.right) / torch.tensor(2., requires_grad=True) 
        sf1 = - (x - self.left) / (self.right - self.left) + self.one
        sf2 = (x - self.left)/(self.right - self.left)
        if training : return  sf1, sf2, self.right - self.left, x
        else : return  sf1, sf2


l, r    =  -0.9, 0.3
mySF    = mySF1D_elementBased(left = l, right = r)

XX      = torch.linspace(l,r,100)
s1, s2  = mySF(XX)
plt.plot(XX.data, s1.data,label='N1')
plt.plot(XX.data, s2.data,label='N2')
plt.grid()
plt.xlabel("x [mm]")
plt.ylabel("shape functions")
plt.legend()  
plt.show()
