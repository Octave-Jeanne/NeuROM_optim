import torch
from .operators import grad
import numpy as np
import meshio
import os



def poisson_2D(field, Mat, source):
    # Get values of interest
    _, _, gauss_weights, mapping_det, field_values = field()
    grad_field_values = grad(field)
    mat_values = Mat('epsilon',el_ids = torch.arange(0, field.NElem), NPoints = gauss_weights.shape[1])
    source_values = source(el_ids = torch.arange(0, field.NElem), NPoints = gauss_weights.shape[1])

    field_term = 0.5*mat_values*((grad_field_values**2).sum(dim = -1))
    source_term = source_values*field_values

    potential = gauss_weights*mapping_det*(field_term - source_term)

    return potential.sum()




class Poisson_2D():
    def __init__(self, vtk_export, baseline, baseline_mesh):
        self.vtk_export = vtk_export
        self.baseline = baseline
        self.baseline_mesh = baseline_mesh

        if self.vtk_export['export_vtk']:
            self.clear_folder()


    def __call__(self, field, Mat, Source, get_metrics):
        _, _, gauss_weights, mapping_det, field_values = field()
        grad_field_values = grad(field)
        mat_values = Mat('epsilon',el_ids = torch.arange(0, field.NElem), NPoints = gauss_weights.shape[1])
        source_values = Source(el_ids = torch.arange(0, field.NElem), NPoints = gauss_weights.shape[1])

        field_term = mat_values*((grad_field_values**2).sum(dim = -1))
        source_term = source_values*field_values

        potential = gauss_weights*mapping_det*(0.5*field_term - source_term)
        loss = potential.sum()

        if get_metrics:
            metrics = {'loss' : loss}
            
            return metrics

        else :
            return loss

    
    def export_vtk(self, epoch, field, Mat, Source):
        _, _, gauss_weights, mapping_det, field_values = field()
        grad_field_values = grad(field)
        mat_values = Mat('epsilon',el_ids = torch.arange(0, field.NElem), NPoints = gauss_weights.shape[1])
        source_values = Source(el_ids = torch.arange(0, field.NElem), NPoints = gauss_weights.shape[1])

        field_term = mat_values*((grad_field_values**2).sum(dim = -1))
        source_term = source_values*field_values

        potential = gauss_weights*mapping_det*(0.5*field_term - source_term)

        nodal_coordinates, nodal_values = field.getResults()

        nodal_coordinates = nodal_coordinates.squeeze().data.numpy()
        nodal_values = nodal_values.squeeze().data.numpy()
        exact_values = np.array([self.baseline(self.baseline_mesh(*p)) for p in [(nodal_coordinates[i, 0], nodal_coordinates[i, 1]) for i in range(len(nodal_coordinates))]])

        cells = {'triangle': field.connectivity.numpy()}

        point_data = {}
        point_data['Baseline'] = exact_values
        point_data['predicted_values'] = nodal_values
        point_data['abs_error'] = abs(exact_values - nodal_values)

        relat_error = abs(exact_values - nodal_values)/abs(exact_values)
        # relat_error[relat_error==1] = 0
        # relat_error[exact_values==0] = 0
        point_data['relat_error'] = relat_error


        epsilon = Mat('epsilon', el_ids = torch.arange(0, field.NElem), NPoints = 1)
        rho = Source(el_ids = torch.arange(0, field.NElem), NPoints = 1)

        cell_data = {}
        cell_data['epsilon'] = [epsilon.squeeze().clone().detach().data.numpy()]
        cell_data['rho'] = [rho.squeeze().clone().detach().data.numpy()]
        cell_data['E'] = [grad_field_values.squeeze(1).clone().detach().data.numpy()]

        vtk_mesh = meshio.Mesh(points = nodal_coordinates,
                                cells = cells,
                                point_data = point_data,
                                cell_data = cell_data
                                )
        
        save_path = os.path.join(self.vtk_export['path_to_folder'], f'result_{epoch}.vtk')
        vtk_mesh.write(save_path)


    def clear_folder(self):
        if len(self.vtk_export['path_to_folder']) != 0:
            os.makedirs(self.vtk_export['path_to_folder'], exist_ok=True)
            for filename in os.listdir(self.vtk_export['path_to_folder']):
                file_path = os.path.join(self.vtk_export['path_to_folder'], filename)

                if file_path.endswith('.vtk'):
                    os.remove(file_path)

