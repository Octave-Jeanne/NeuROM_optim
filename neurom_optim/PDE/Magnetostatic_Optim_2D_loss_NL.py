import torch
from torch import nn
from .operators import curl
import meshio
import os
import shutil
import numpy as np


class Magnetostatic_Optim_2D_loss(nn.Module):
    def __init__(self, 
                 lagrange_multiplier, 
                 IntPrecision, 
                 FloatPrecision,
                 vtk_export = {'cell_data': [], 'point_data' : [], 'path_to_folder' : ''}):
        

        super().__init__()
        self.register_buffer('ref_int', torch.tensor(1, dtype = IntPrecision))
        self.register_buffer('ref_float', torch.tensor(1, dtype = FloatPrecision))
        self.register_buffer('lagrange_multiplier', self.reformat_value(lagrange_multiplier))
        self.vtk_export = vtk_export

        if self.vtk_export['export_vtk']:
            self.clear_folder()


    def clear_folder(self):
        os.makedirs(self.vtk_export['path_to_folder'], exist_ok = True)
        if len(self.vtk_export['path_to_folder']) != 0:
            for filename in os.listdir(self.vtk_export['path_to_folder']):
                file_path = os.path.join(self.vtk_export['path_to_folder'], filename)

                if file_path.endswith('.vtk'):
                    os.remove(file_path)




    def forward(self, field, Mat, source = None, get_metrics = False):

        _, _, gauss_weights, mapping_det, field_values = field()
        curl_field_values = curl(field)
        mat_values = Mat('nu',el_ids = torch.arange(0, field.NElem), NPoints = gauss_weights.shape[1], B = curl_field_values)

        if source is not None:
            source_values = source(el_ids = torch.arange(0, field.NElem), NPoints = gauss_weights.shape[1])
        else:
            source_values = torch.zeros_like(field_values)

        choice = Mat('nu',el_ids = torch.arange(0, Mat.NElem), NPoints = 1, return_choice = True)


        field_term = (gauss_weights*mapping_det*(0.5*mat_values*((curl_field_values**2).sum(dim = -1).unsqueeze(-1)))).sum()
        source_term = (gauss_weights*mapping_det*(source_values*field_values)).sum()

       

        potential = field_term - source_term

        domain_area = (mapping_det*gauss_weights).sum()
        soft_volume_term = (gauss_weights* mapping_det*choice).sum()/domain_area
        hard_volume_term = (gauss_weights* mapping_det*(choice>=0.5)).sum()/domain_area


        loss = potential + self.lagrange_multiplier*soft_volume_term 


        if get_metrics:
            metrics = {'loss' : loss,
                       'soft_volume_term' : soft_volume_term,
                       'hard_volume_term' : hard_volume_term,
                       'potential' : potential,
                       'field_term' : field_term,
                       'source_term' : source_term}
            
            return metrics

        else :
            return loss

    
    
    
    
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

            _, _, gauss_weights, mapping_det, field_values = field()
            curl_field_values = curl(field)
            mat_values = Mat('nu',el_ids = torch.arange(0, field.NElem), NPoints = gauss_weights.shape[1], B = curl_field_values)

            if source is not None:
                source_values = source(el_ids = torch.arange(0, field.NElem), NPoints = gauss_weights.shape[1])
            else:
                source_values = torch.zeros_like(field_values)

            field_term = gauss_weights*mapping_det*(0.5*mat_values*((curl_field_values**2).sum(dim = -1).unsqueeze(-1)))
            source_term = gauss_weights*mapping_det*(source_values*field_values)

       

            potential = field_term - source_term

            choice = Mat('nu',el_ids = torch.arange(0, Mat.NElem), NPoints = 1, return_choice = True)

            domain_area = (mapping_det*gauss_weights).sum()
            volume_term = (gauss_weights*mapping_det*choice)/domain_area


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

                    case 'field_term':
                        cell_data['field_term'] = [(field_term).clone().detach().squeeze(1).cpu().data.numpy()]

                    case 'source_term':
                        cell_data['source_term'] = [(source_term).clone().detach().squeeze(1).cpu().data.numpy()]

                    case 'potential':
                        cell_data['potential'] = [(potential).clone().detach().squeeze(1).cpu().data.numpy()]

                    case 'volume_term':
                        cell_data['volume_term'] = [(volume_term).clone().detach().squeeze(1).cpu().data.numpy()]
                    
                    
                    case 'soft_choice':
                        cell_data['soft_choice'] = [choice.clone().detach().squeeze(1).cpu().data.numpy()]

                    case 'hard_choice':
                        cell_data['hard_choice'] = [(choice>=0.5).int().to(potential.dtype).clone().detach().squeeze(1).cpu().data.numpy()]

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

