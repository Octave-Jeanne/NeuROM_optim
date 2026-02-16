import numpy as np
import meshio
import torch
import os

from ...PDE.operators import curl

def optim_2D_to_vtk(A, Mat, video = False, path_to_folder = ''):
    if video:
        X_interm = A.X_interm
        U_interm = A.U_interm
        soft_choice_interm = Mat.properties['nu'].soft_choice_interm
        hard_choice_interm = Mat.properties['nu'].hard_choice_interm

        cells = {'triangle': A.connectivity.numpy()}

        for time_step, evaluation_points in enumerate(X_interm[1:]):
            point_data = {}
            cell_data = {}

            evaluation_points = evaluation_points.data.numpy()
            a_value = U_interm[time_step].data.numpy()
            soft_choice = soft_choice_interm[time_step].data.numpy()
            #hard_choice = hard_choice_interm[time_step].data.numpy()

            cell_data['choice'] = [soft_choice]
            point_data['A'] = np.squeeze(a_value)

            vtk_mesh = meshio.Mesh(points = evaluation_points,
                            cells = cells,
                            point_data = point_data,
                            cell_data = cell_data
                            )
            path = os.path.join(path_to_folder, f'result_{time_step}.vtk')
            vtk_mesh.write(path)

    else:


        
        point_data = {}
        cell_data = {}

        evaluation_points = A.X_interm[-1].data.numpy()
        cells = {'triangle': A.connectivity.numpy()}

        evaluation_points, A_values = A.getResults()

        point_data['A'] = A_values.clone().detach().squeeze().cpu().data.numpy()


        B = curl(A).squeeze(1)

        nu = Mat('nu', el_ids = torch.arange(0, Mat.NElem), NPoints = 1)
        nu = nu.squeeze(1).squeeze(-1)
        choice = Mat('nu',el_ids = torch.arange(0, Mat.NElem), NPoints = 1, return_choice = True).squeeze().data.clone().detach().numpy()

        energy_density = (0.5*((B**2).sum(-1))*nu).clone().detach().data.numpy()
        
        nu = nu.squeeze().data.clone().detach().numpy()

        _, _, _, mapping_det, _ = A()
        cell_data['mapping_det'] = [mapping_det.squeeze().squeeze()]

        cell_data['nu'] = [nu]
        cell_data['soft_choice'] = [choice]
        #cell_data['hard_choice'] = [(choice>=0.5).int()]
        cell_data['B']  = [B.clone().detach().data.numpy()]
        cell_data['local_energy_density'] = [energy_density]

        cell_data['global_energy_density'] = [mapping_det.squeeze().squeeze()*energy_density]

        # Debug
        _, _, gauss_weights, mapping_det, _ = A() 
        curl_field_values = curl(A)
        mat_values = Mat('nu',el_ids = torch.arange(0, A.NElem), NPoints = gauss_weights.shape[1])
        field_term = 0.5*mat_values*((curl_field_values**2).sum(dim = -1).unsqueeze(-1))
        potential = gauss_weights*mapping_det*field_term

        cell_data['field_term'] = [field_term.squeeze().squeeze().clone().detach().data.numpy()]
        cell_data['potential'] = [potential.squeeze().squeeze().clone().detach().data.numpy()]

        vtk_mesh = meshio.Mesh(points = evaluation_points,
                            cells = cells,
                            point_data = point_data,
                            cell_data = cell_data
                            )
        
        return vtk_mesh



