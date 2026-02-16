import meshio
import torch
import numpy as np
import os

from ..Baseline.Poisson_2D_scalar_NGSolve import Poisson_2D_scalar_NGSolve_uniform_mat_constant_source



def export_VTK(model, Mat, baseline, baseline_mesh, config):
    model.to('cpu')
    Mat.to('cpu')
    evaluation_points = model.X_interm[-1].data.numpy()
    predicted_values = np.squeeze(model.U_interm[-1].data.numpy())

    pts = [(evaluation_points[i, 0], evaluation_points[i, 1]) for i in range(len(evaluation_points))]
    exact_values = np.array([baseline(baseline_mesh(*p)) for p in pts])

    # Tackle the numerical zeros
    exact_values[exact_values<1e-8] = 0

    abs_error = abs(exact_values - predicted_values)
    relat_error = abs(exact_values - predicted_values)/abs(exact_values)



    relat_error_ref_NeuROM = abs(exact_values - predicted_values)/abs(predicted_values)





    vtk_mesh = meshio.Mesh(points = model.X_interm[-1].numpy(),
                           cells = {'triangle': model.connectivity.numpy()},
                           point_data = {"U":predicted_values, "NgSolve_U":exact_values, "relat_error" : relat_error, "relat_error_ref_NeuROM": relat_error_ref_NeuROM, "abs_error" : abs_error},
                           cell_data = {"eps" : [Mat('epsilon', el_ids = torch.arange(0, Mat.NElem), NPoints = 1).squeeze().data.clone().detach().numpy()]}
                           )
    
    # vtk_mesh = meshio.Mesh(model.X_interm[-1].numpy(),
    #                        {'triangle': model.connectivity.numpy()},
    #                        point_data = {"U":predicted_values, "NgSolve_U":exact_values, "relat_error" : relat_error, "relat_error_ref_NeuROM": relat_error_ref_NeuROM, "abs_error" : abs_error},
    #                        cell_data = {"eps" : Mat('epsilon', el_ids = torch.arange(0, Mat.NElem), NPoints = 1).squeeze().data.clone().detach().numpy()}
    #                        )


    save_folder_path = config['postprocess']['save_folder_path']
    os.makedirs(save_folder_path, exist_ok = True)

    save_name = f'{config['problem']['type']}_source_{config['problem']['source_type']}_mat_{config['material'][0]['material_type']}_quad_{config['interpolation']['quadrature_order']}.vtk'

    save_path = os.path.join(save_folder_path, save_name)
    vtk_mesh.write(
        save_path, 
    )