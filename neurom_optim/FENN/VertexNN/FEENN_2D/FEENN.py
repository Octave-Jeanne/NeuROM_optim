from torch import nn
import torch
from .Grid import GridNN_2D
from .Interpolation import InterpolationNN_2D

class FEENN_2D(nn.Module):
    def __init__(self, Nodes, connectivity, n_components, element, mapping, IntPrecision, FloatPrecision):
        super().__init__()

        self.register_buffer('ref_int', torch.tensor(1, dtype = IntPrecision))
        self.register_buffer('ref_float', torch.tensor(1, dtype = FloatPrecision))

        

        self.register_buffer('NElem', torch.tensor(len(connectivity), dtype = self.ref_int.dtype))
        self.register_buffer('NNodes', torch.tensor(len(Nodes), dtype = self.ref_int.dtype))
        self.register_buffer('n_components', torch.tensor(n_components, dtype = self.ref_int.dtype))
        self.register_buffer('connectivity', torch.tensor(connectivity, dtype = self.ref_int.dtype))
        self.register_buffer('dim', torch.tensor(2, dtype = self.ref_int.dtype))
        self.X_interm = []
        self.U_interm = []

        self.grid = GridNN_2D(Nodes             = Nodes,
                              IntPrecision      = IntPrecision,
                              FloatPrecision    = FloatPrecision)
        

        self.interpolation = InterpolationNN_2D(NNodes = self.NNodes, 
                                                NElem = self.NElem,
                                                n_components = self.n_components,
                                                element = element,
                                                mapping = mapping,
                                                IntPrecision = IntPrecision,
                                                FloatPrecision = FloatPrecision)
        
        


    def SetBCs(self, 
               Fixed_nodal_coordinates_Ids  :   torch.Tensor    =   None,
               Fixed_nodal_values_Ids       :   torch.Tensor    =   None,
               Fixed_nodal_values_values    :   torch.Tensor    =   None):

        self.grid.SetBCs(Fixed_nodal_coordinates_Ids)

        self.interpolation.SetBCs(Fixed_Ids = Fixed_nodal_values_Ids,
                                  Fixed_Values = Fixed_nodal_values_values)
    
    def SetQuad(self, quadrature_order):
        self.interpolation.SetQuad(quadrature_order)

    def forward(self,
                train_mode          :   bool            =   True,
                grad_mode           :   bool            =   False, 
                global_coordinates  :   torch.Tensor    =   None):
        
        
        if train_mode:
            return self.train_forward(grad_mode)
        
        else:
            return self.eval_forward(global_coordinates, grad_mode)
        
    def train_forward(self, 
                      grad_mode):
        
        nodal_coordinates = self.grid(self.connectivity)
        return self.interpolation(nodal_coordinates = nodal_coordinates,
                                  connectivity = self.connectivity,
                                  train_mode = True,
                                  grad_mode = grad_mode)
    
        

    def eval_forward(self, global_coordinates):
        """
        To do
        """
        pass

    def StoreResults(self):
        self.X_interm.append(self.grid.GetCoord().cpu())
        self.U_interm.append(self.interpolation.GetValues().cpu())

    def getResults(self):
        return self.grid.GetCoord().cpu(), self.interpolation.GetValues().cpu()

    def Freeze(self,
               freeze_grid = True,
               freeze_interpolation = True):
        
        if freeze_grid:
            self.grid.Freeze()

        if freeze_interpolation:
            self.interpolation.Freeze()

    def UnFreeze(self,
                 unfreeze_grid = True,
                 unfreeze_interpolation = True):
        
        if unfreeze_grid:
            self.grid.UnFreeze()

        if unfreeze_interpolation:
            self.interpolation.UnFreeze()

    def reformat_value(self, value, dtype = None):
        if dtype is not None:
            if type(value) == torch.Tensor:
                value = value.clone().detach().to(dtype)

            else:
                value = torch.tensor(value, dtype = dtype)

        elif type(value) == int:
            value = torch.tensor(value, dtype = self.ref_int.dtype)

        elif type(value) == float:
            value = torch.tensor(value, dtype = self.ref_float.dtype)
            
        elif type(value) == bool:
            value = torch.tensor(value)
            
        elif type(value) == torch.Tensor:
            value = value.clone().detach()
        
        return value