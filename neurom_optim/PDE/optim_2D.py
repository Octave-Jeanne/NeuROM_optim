import torch
from torch import nn
from .operators import curl
import meshio
import os
import shutil
import numpy as np


class Optim_2D_loss(nn.Module):
    def __init__(self, 
                 budget, 
                 lagrange_multiplier, 
                 focus_multiplier, 
                 IntPrecision, 
                 FloatPrecision,
                 vtk_export = {'cell_data': [], 'point_data' : [], 'path_to_folder' : ''}):
        

        super().__init__()
        self.register_buffer('ref_int', torch.tensor(1, dtype = IntPrecision))
        self.register_buffer('ref_float', torch.tensor(1, dtype = FloatPrecision))
        self.register_buffer('budget', self.reformat_value(budget))
        self.register_buffer('lagrange_multiplier', self.reformat_value(lagrange_multiplier))
        self.register_buffer('focus_multiplier', self.reformat_value(focus_multiplier))
        self.vtk_export = vtk_export

        #if self.vtk_export['export_vtk']:
        #    self.clear_folder()


    def clear_folder(self):
        os.makedirs(self.vtk_export['path_to_folder'], exist_ok = True)
        if len(self.vtk_export['path_to_folder']) != 0:
            for filename in os.listdir(self.vtk_export['path_to_folder']):
                file_path = os.path.join(self.vtk_export['path_to_folder'], filename)

                if file_path.endswith('.vtk'):
                    os.remove(file_path)




    def forward(self, field, Mat, source = None, get_metrics = False):


        _, _, gauss_weights, mapping_det, _ = field()
        curl_field_values = curl(field)
        mat_values = Mat('nu',el_ids = torch.arange(0, field.NElem), NPoints = gauss_weights.shape[1])
        # source_values = source(el_ids = torch.arange(0, field.NElem), NPoints = gauss_weights.shape[1])
        choice = Mat('nu',el_ids = torch.arange(0, Mat.NElem), NPoints = gauss_weights.shape[1], return_choice = True)

        physics = self.physics_loss(curl_field_values, mat_values, mapping_det, gauss_weights)
        constraint = self.soft_constraint(choice, mapping_det, gauss_weights, self.budget)
        focus = self.focus_constraint(choice, mapping_det, gauss_weights)

        physics = physics.sum()
        constraint = constraint.sum()
        focus = focus.sum()

        loss = physics + self.lagrange_multiplier*constraint + self.focus_multiplier*focus


        if get_metrics:
            metrics = {'loss' : loss,
                       'constraint' : constraint,
                       'physics' : physics,
                       'focus': focus}
            
            return metrics

        else :
            return loss

    @staticmethod
    def physics_loss(curl_field_values, mat_values, mapping_det, gauss_weights):

        field_term = 0.5*mat_values*((curl_field_values**2).sum(dim = -1).unsqueeze(-1))
       

        potential = gauss_weights*mapping_det*field_term

        return potential
    
    @staticmethod
    def soft_constraint(choice, mapping_det, gauss_weights, budjet):
        domain_area = (mapping_det*gauss_weights).sum()

        # focus = ((choice>0.5)*torch.log10(1-choice) + (choice<=0.5)*torch.log10(choice)).sum()
        
        # return focus + ((choice.sum()/domain_area) - budjet)**2
    
        # return ((choice.sum()/domain_area) - budjet)**2

        return (mapping_det*choice).sum()/domain_area - budjet
    
    def focus_constraint(self, choice, mapping_det, gauss_weights):
        
        
        

        # mask = ~ ( (choice == 1) & (choice == 0) )

        # focus = choice[mask]*torch.log10(choice[mask]) + (1-choice[mask])*torch.log10(1-choice[mask])

        focus = gauss_weights*mapping_det*choice*(1-choice)

        return focus
    
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

        elif type(value) == np.float32:
            value = torch.tensor(value)
            
        elif type(value) == torch.Tensor:
            value = value.clone().detach()

       
        return value
    

    def export_vtk(self, epoch, field, Mat, source = None):

        if self.vtk_export['export_vtk']:

            _, _, gauss_weights, mapping_det, field_value = field()
            curl_field_values = curl(field)
            mat_values = Mat('nu',el_ids = torch.arange(0, field.NElem), NPoints = gauss_weights.shape[1])
            # source_values = source(el_ids = torch.arange(0, field.NElem), NPoints = gauss_weights.shape[1])
            choice = Mat('nu',el_ids = torch.arange(0, Mat.NElem), NPoints = 1, return_choice = True)

            physics = self.physics_loss(curl_field_values, mat_values, mapping_det, gauss_weights)
            constraint = self.soft_constraint(choice, mapping_det, gauss_weights, self.budget)
            focus = self.focus_constraint(choice, mapping_det, gauss_weights)

            cells = {'triangle' : field.connectivity.clone().detach().cpu().data.numpy()}
            point_data = {}
            cell_data = {}

            nodal_coordinates, A_values = field.getResults()
            for point_data_name in self.vtk_export['point_data']:
                match point_data_name:
                    case 'A':
                        point_data['A'] = A_values.clone().detach().squeeze().cpu().data.numpy()
            
            for cell_data_name in self.vtk_export['cell_data']:
                match cell_data_name:
                    case 'B':
                        cell_data['B'] = [curl_field_values.clone().detach().squeeze(1).cpu().data.numpy()]

                    case 'energy':
                        cell_data['energy'] = [(2*physics).clone().detach().squeeze(1).cpu().data.numpy()]
                    
                    case 'focus':
                        cell_data['focus'] = [focus.clone().detach().squeeze(1).cpu().data.numpy()]
                    
                    
                    case 'soft_choice':
                        cell_data['soft_choice'] = [choice.clone().detach().squeeze(1).cpu().data.numpy()]

                    case 'hard_choice':
                        cell_data['hard_choice'] = [(choice>=0.5).int().to(physics.dtype).clone().detach().squeeze(1).cpu().data.numpy()]

                    case 'nu':
                        cell_data['nu'] = [mat_values.clone().detach().squeeze(1).cpu().data.numpy()]

                    case 'mapping_det':
                        cell_data['mapping_det'] = [mapping_det.clone().detach().squeeze(1).cpu().data.numpy()]


            vtk_mesh = meshio.Mesh(points = nodal_coordinates.clone().detach().squeeze().cpu().data.numpy(),
                                    cells = cells,
                                    point_data = point_data,
                                    cell_data = cell_data
                                    )
            
            os.makedirs(self.vtk_export['path_to_folder'], exist_ok=True)
            path = os.path.join(self.vtk_export['path_to_folder'], f'result_{epoch}.vtk')
            vtk_mesh.write(path)

