import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

class mySF1D_elementBased_vectorised(nn.Module):
    def __init__(self, connectivity):
        super(mySF1D_elementBased_vectorised, self).__init__()
        if connectivity.dim == 1:
            connectivity = connectivity[:,None]
        self.connectivity = connectivity
        self.register_buffer('GaussPoint',self.GP())
        self.register_buffer('w_g',torch.tensor(1.0))


    def UpdateConnectivity(self,connectivity):
        self.connectivity = connectivity.astype(int)

    def GP(self):
        "Defines the position of the intergration point(s) for the given element"

        return torch.tensor([[1/2, 1/2]], requires_grad=True)                                       # a1, a2, th 2 area coordinates

    def forward(self, 
                x               : torch.Tensor  = None  , 
                cell_id         : list          = None  , 
                coordinates     : torch.Tensor  = None  , 
                flag_training   : bool          = False):

        assert coordinates is not None, "No nodes coordinates provided. Aborting"

        cell_nodes_IDs  = self.connectivity[cell_id,:].T
        Ids             = torch.as_tensor(cell_nodes_IDs).to(coordinates.device).t()[:,:,None]      # :,:,None] usefull only in 2+D  
        nodes_coord     = torch.gather(coordinates[:,None,:].repeat(1,2,1),0, Ids.repeat(1,1,1))    # [:,:,None] usefull only in 2+D  Ids.repeat(1,1,d) with d \in [1,3]
        
        nodes_coord = nodes_coord.to(self.GaussPoint.dtype)

        if flag_training:
            refCoordg   = self.GaussPoint.repeat(cell_id.shape[0],1)
            Ng          = refCoordg
            x_g         = torch.einsum('enx,en->ex',nodes_coord,Ng)
            refCoord    = self.GetRefCoord(x_g,nodes_coord)
            N           = refCoord
            detJ        = nodes_coord[:,1] - nodes_coord[:,0]
            return N, x_g, detJ*self.w_g

        else:
            refCoord = self.GetRefCoord(x,nodes_coord)
            N = torch.stack((refCoord[:,0], refCoord[:,1]),dim=1) 
            return N

    
    def GetRefCoord(self,x, nodes_coord):
        InverseMapping          = torch.ones([int(nodes_coord.shape[0]), 2, 2], dtype=x.dtype, device=x.device)
        detJ                    = nodes_coord[:,0,0] - nodes_coord[:,1,0]
        InverseMapping[:,0,1]   = -nodes_coord[:,1,0]
        InverseMapping[:,1,1]   = nodes_coord[:,0,0]
        InverseMapping[:,1,0]   = -1*InverseMapping[:,1,0]
        InverseMapping[:,:,:]  /= detJ[:,None,None]
        x_extended = torch.stack((x, torch.ones_like(x)),dim=1)


        return torch.einsum('eij,ej...->ei',InverseMapping,x_extended.squeeze(1))
