import torch
from torch import nn
import vtk

class GridNN_2D(nn.Module):
    def __init__(self, Nodes, IntPrecision, FloatPrecision):
        super().__init__()

        self.register_buffer('ref_int', torch.tensor(1, dtype = IntPrecision))
        self.register_buffer('ref_float', torch.tensor(1, dtype = FloatPrecision))

        

        # Keep track of device and dtype used throughout the model
        self.register_buffer('dim', torch.tensor(2, dtype = self.ref_int.dtype))
        self.register_buffer('NNodes', self.reformat_value(torch.tensor(len(Nodes))))
        self.register_buffer('all_nodal_coordinates', self.reformat_value(Nodes))
        self.register_buffer('free', torch.tensor([True for i in self.all_nodal_coordinates]))
        
        self.nodal_coordinates =nn.ParameterDict({
                                                'free': self.all_nodal_coordinates,
                                                'imposed': torch.tensor([])
                                                })
        
        



    def forward(self, connectivity):
        nodal_coordinates = torch.ones_like(self.all_nodal_coordinates, dtype = self.ref_float.dtype)
        nodal_coordinates[self.free] = self.nodal_coordinates['free']
        nodal_coordinates[~self.free] = self.nodal_coordinates['imposed']
        
        return nodal_coordinates[connectivity]
    
    
        
    def SetBCs(self, Fixed_Ids):
        """
        Permanently fixes the nodes specified in the Boundary conditions
        """
        Fixed_Ids = self.reformat_value(Fixed_Ids)
        self.free[Fixed_Ids] = False
        self.nodal_coordinates['free'] = self.all_nodal_coordinates[self.free,:]
        self.nodal_coordinates['imposed'] = self.all_nodal_coordinates[~self.free,:]

        self.Freeze()
        self.UnFreeze()

    def Freeze(self):
        """
        This function prevents any modification of node coordinates during optimisation.
        """

        self.nodal_coordinates['free'].requires_grad = False
        self.nodal_coordinates['imposed'].requires_grad = False

    def UnFreeze(self):
        """
        Allows the free nodes to be trained
        """

        self.nodal_coordinates['free'].requires_grad = True
        self.nodal_coordinates['imposed'].requires_grad = False
    
    
    def GetCoord(self):
        """
        Returns the current mesh for future post-processing plots related use
        """
        coord = torch.ones_like(self.all_nodal_coordinates, dtype = self.ref_float.dtype)
        coord[self.free] = self.nodal_coordinates['free']
        coord[~self.free] = self.nodal_coordinates['imposed']
        return coord.detach().clone()
        

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

    