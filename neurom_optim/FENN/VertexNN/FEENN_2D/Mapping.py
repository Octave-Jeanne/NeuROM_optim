import torch
from torch import nn

class Mapping_2D_Affine(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('dim', torch.tensor(2, dtype = torch.int))

    def get_direct_mapping_parameter(self, nodal_coordinates):
        # b ~ elemnent x dim
        b = nodal_coordinates[:,0,:self.dim]
        # A ~ element x dim x dim
        A = torch.ones(size = (len(nodal_coordinates), self.dim, self.dim), device = nodal_coordinates.device, dtype = nodal_coordinates.dtype)
        A[:,0,0] = nodal_coordinates[:,1,0] - nodal_coordinates[:,0,0]
        A[:,0,1] = nodal_coordinates[:,2,0] - nodal_coordinates[:,0,0]
        A[:,1,0] = nodal_coordinates[:,1,1] - nodal_coordinates[:,0,1]
        A[:,1,1] = nodal_coordinates[:,2,1] - nodal_coordinates[:,0,1]

        det = A[:, 0, 0]*A[:, 1, 1] - A[:, 0, 1]*A[:, 1, 0]

        return A, b, det

    def get_reverse_mapping_parameter(self, A, b, det):

        # Reverse the direct mapping
        reverse_A = torch.ones_like(A)
        reverse_A[:, 0, 0] = A[:, 1, 1]
        reverse_A[:, 0, 1] = -A[:, 0, 1]
        reverse_A[:, 1, 0] = -A[:, 1, 0]
        reverse_A[:, 1, 1] = A[:, 0, 0]
        reverse_A = reverse_A/det.unsqueeze(-1).unsqueeze(-1)
        
        reverse_b = -torch.einsum('eik,ek->ei', reverse_A, b)

        # reverse the direct determinant
        reverse_det = 1/det

        return reverse_A, reverse_b, reverse_det


    def get_mapping_parameters(self, nodal_coordinates, mode):
        A, b, det = self.get_direct_mapping_parameter(nodal_coordinates)

        match mode:
            case 'direct':
                return A, b, det
            
            case 'reverse'|'grad':
                return self.get_reverse_mapping_parameter(A, b, det)
            
            
        
        
    def forward(self, 
                nodal_coordinates   :   torch.tensor    =   None, 
                entity              :   torch.tensor    =   None, 
                mode                :   str             =   True):
        

        A, b, det = self.get_mapping_parameters(nodal_coordinates,
                                                mode)
        
        A, b, det, entity = self.reshape(A, b, det, entity, mode)

        match mode:
            case 'direct':
                mapped_coordinates = torch.einsum('epik,epk->epi', A, entity) + b
                return mapped_coordinates, det
            
            case 'reverse':
                mapped_coordinates = torch.einsum('pik,pk->pi', A, entity) + b
                return mapped_coordinates
            
            case 'grad':
                return torch.einsum('egni,egij->egnj', entity, A)

        

        
    def reshape(self, 
                A, 
                b,
                det, 
                entity,
                mode):
        
        NElem = A.shape[0]
        NPoints = entity.shape[0]

        if mode == 'direct' or mode == 'grad':
            entity = entity.unsqueeze(0)
            entity = entity.repeat([NElem] + [1 for _ in range(len(entity.shape) - 1)])

            A = A.unsqueeze(1)
            A = A.repeat(1, NPoints, 1, 1)

            b = b.unsqueeze(1)
            b = b.repeat(1, NPoints, 1)

            det = det.unsqueeze(1)
            det = det.repeat(1, NPoints)
            det = det.unsqueeze(-1)

        return A, b, det, entity

