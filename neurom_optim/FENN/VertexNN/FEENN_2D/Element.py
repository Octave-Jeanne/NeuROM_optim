import torch
from torch import nn


class Tri_2D_lin(nn.Module):
    def __init__(self, IntPrecision, FloatPrecision):
        super().__init__()

        self.register_buffer('ref_int', torch.tensor(1, dtype = IntPrecision))
        self.register_buffer('ref_float', torch.tensor(1, dtype = FloatPrecision))

        
        
        self.register_buffer('NNodes_per_element', torch.tensor(3, dtype = self.ref_int.dtype))
        self.register_buffer('dim', torch.tensor(2, dtype = self.ref_int.dtype))


        
    
    def SetQuad(self, quadrature_order):
        self.set_GP(quadrature_order)
        self.set_GW(quadrature_order)
        self.set_GS()
        self.set_Grad()
        

    def set_GP(self, quadrature_order):
        match quadrature_order:
            case "1"|1:
                gauss_coordinates =  torch.tensor([1/3, 1/3], dtype = self.ref_float.dtype).unsqueeze(0)
            
            case "2"|2:
                gauss_coordinates = torch.tensor([[1/6, 1/6],
                                                  [2/3, 1/6],
                                                  [1/6, 2/3]], dtype = self.ref_float.dtype)
            # etc
        
        self.register_buffer('gauss_coordinates', gauss_coordinates)

    def set_GW(self, quadrature_order):
        match quadrature_order:
            case "1"|1:
                gauss_weights = torch.tensor([1/2], dtype = self.ref_float.dtype).unsqueeze(-1)
            
            case "2"|2:
                gauss_weights = torch.tensor([1/6, 1/6, 1/6], dtype = self.ref_float.dtype).unsqueeze(-1)

            # etc
        
        self.register_buffer('gauss_weights', gauss_weights)

    
    def set_GS(self):
        shape_functions = torch.ones(size = (self.gauss_coordinates.shape[0], self.NNodes_per_element), dtype = self.ref_float.dtype)
        shape_functions[:, 0] = 1 - self.gauss_coordinates[:, 0] - self.gauss_coordinates[:, 1]
        shape_functions[:, 1] = self.gauss_coordinates[:, 0]
        shape_functions[:, 2] = self.gauss_coordinates[:, 1] 

        self.register_buffer('gauss_shape_functions', shape_functions)

    def set_Grad(self):
        grad_shape_functions = torch.ones(size = (self.gauss_coordinates.shape[0], self.NNodes_per_element, self.dim), dtype = self.ref_float.dtype)
        grad_shape_functions[:, 0, 0] = -1
        grad_shape_functions[:, 0, 1] = -1
        grad_shape_functions[:, 1, 0] = 1
        grad_shape_functions[:, 1, 1] = 0
        grad_shape_functions[:, 2, 0] = 0
        grad_shape_functions[:, 2, 1] = 1

        self.register_buffer('grad_gauss_shape_functions', grad_shape_functions)
        
    

    def forward(self, 
                local_coordinates   :   torch.tensor    =   None,
                train_mode          :   bool            =   True,
                grad_mode           :   bool            =   False):

        if train_mode:
            return self.train_forward(grad_mode)
        
        else:
            return self.eval_forward(local_coordinates, grad_mode)
        

    def train_forward(self,
                      grad_mode):
        

        if grad_mode:
            return self.grad_gauss_shape_functions

        else:
            return self.gauss_coordinates, self.gauss_weights, self.gauss_shape_functions

    def eval_forward(self,
                     local_coordinates, 
                     grad_mode):
        pass
  