import numpy as np
import meshio
import torch

def scalar_2D_to_vtk(baseline, baseline_mesh, predicted_field, Mat, Source):
    point_data = {}
    cell_data = {}

    evaluation_points = predicted_field.X_interm[-1].data.numpy()
    cells = {'triangle': predicted_field.connectivity.numpy()}


    exact_values = np.array([baseline(baseline_mesh(*p)) for p in [(evaluation_points[i, 0], evaluation_points[i, 1]) for i in range(len(evaluation_points))]])
    predicted_values = np.squeeze(predicted_field.U_interm[-1].data.numpy())

    point_data['Baseline'] = exact_values
    point_data['predicted_values'] = predicted_values
    point_data['abs_error'] = abs(exact_values - predicted_values)
    point_data['relat_error'] = abs(exact_values - predicted_values)/abs(exact_values)

    epsilon = Mat('epsilon', el_ids = torch.arange(0, predicted_field.NElem), NPoints = 1).squeeze().data.clone().detach().numpy()
    rho = Source(el_ids = torch.arange(0, predicted_field.NElem), NPoints = 1)

    cell_data['epsilon'] = [epsilon]
    cell_data['rho'] = [rho]


    vtk_mesh = meshio.Mesh(points = evaluation_points,
                           cells = cells,
                           point_data = point_data,
                           cell_data = cell_data
                           )
    
    return vtk_mesh


