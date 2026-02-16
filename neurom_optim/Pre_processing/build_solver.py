from ..Solver.solver import Solver

def build_solver(solver_config, 
                 loss, 
                 model, 
                 Mat,
                 source = None):
    
    stop_criteria_definitions = {}
    hyperparameters = {}

    for key in solver_config.keys():
        match key:
            case 'optimizer':
                optimizer_name = solver_config[key]

            case 'n_epochs':
                stop_criteria_definitions['MaxEpochs'] = {'max_epoch': solver_config[key]}
            
            case 'learning_rate':
                hyperparameters['lr'] = solver_config[key]

            case 'loss_decrease_c':
                stop_criteria_definitions['Stagnation'] = {'loss_decrease_c': solver_config[key]}

    return Solver(loss = loss,
                  model = model,
                  Mat = Mat,
                  source = source,
                  optimizer_name = optimizer_name,
                  hyperparameters = hyperparameters,
                  stop_criteria_definitions = stop_criteria_definitions,
                  freeze_grid = solver_config.get('freeze_grid', False),
                  freeze_interpolation = solver_config.get('freeze_interpolation', False),
                  freeze_Mat = solver_config.get('freeze_Mat', False),
                  freeze_Source = solver_config.get('freeze_Source', False))

    
