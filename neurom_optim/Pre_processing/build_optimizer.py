from torch.optim import LBFGS, Adam

# from Custom_Optimizers.TC_optimizer import TC_optim


def build_optimizer(optimizer_name, hyperparameters, model, Mat, source):
    params = []

    if model is not None:
        params.extend(model.parameters())
    
    if Mat is not None:
        params.extend(Mat.parameters())

    if source is not None:
        params.extend(source.parameters())

    match optimizer_name:
        case 'lbfgs':
            optimizer = LBFGS(params = params,
                              lr = hyperparameters.get('lr', 1),
                              max_iter = hyperparameters.get('max_iter', 20),
                              max_eval = hyperparameters.get('max_eval', None),
                              tolerance_grad = hyperparameters.get('tolerance_grad', 1e-7),
                              tolerance_change = hyperparameters.get('tolerance_change', 1e-9),
                              history_size = hyperparameters.get('history_size', 100),
                              line_search_fn = hyperparameters.get('line_search_fn', None))
        
        case 'adam':
            optimizer = Adam(params = params,
                             lr = hyperparameters.get('lr', 0.001),
                             betas = hyperparameters.get('betas', (0.9, 0.999)),
                             eps = hyperparameters.get('eps', 1e-8),
                             weight_decay = hyperparameters.get('weight_decay', 0),
                             amsgrad= hyperparameters.get('amsgrad', False),
                             foreach=hyperparameters.get('foreach', None))

        # case 'step':
        #     optimizer = TC_optim(params = params,
        #                          initial_step = hyperparameters.get('initial_step', 1),
        #                          step_increase = hyperparameters.get('step_increase', 1.2),
        #                          step_decrease = hyperparameters.get('step_decrease', 0.5))
    return optimizer