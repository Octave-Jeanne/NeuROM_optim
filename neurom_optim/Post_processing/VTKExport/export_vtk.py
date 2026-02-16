from .scalar_2D_to_vtk import scalar_2D_to_vtk
from .optim_2D_to_vtk import optim_2D_to_vtk
import os


def export_vtk(config, 
               baseline             =   None,
               baseline_mesh        =   None, 
               predicted_field      =   None,  
               Mat                  =   None, 
               Source               =   None,
               path_to_folder = ''):
    
    if predicted_field is not None:
        predicted_field.to('cpu')
    
    if Mat is not None:
        Mat.to('cpu')

    if Source is not None:
        Source.to('cpu')

    match config['Problem']['type']:
        case 'Poisson':
            match config['Field']['type']:
                case 'scalar_2D':
                    vtk_mesh = scalar_2D_to_vtk(baseline, baseline_mesh, predicted_field, Mat, Source)

        case 'Optim':
            vtk_mesh  = optim_2D_to_vtk(predicted_field, Mat, path_to_folder = path_to_folder)
        

    save_folder_path = config['postprocess']['path_to_folder']
    save_name = config['postprocess']['save_name']

    os.makedirs(save_folder_path, exist_ok = True)
    save_path = os.path.join(save_folder_path, f'{save_name}.vtk')

    if config['postprocess']['export_vtk']:
        vtk_mesh.write(
            save_path, 
        )