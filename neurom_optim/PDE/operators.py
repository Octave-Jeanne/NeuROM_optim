import torch

def grad(model, 
         evaluation_points      :   torch.Tensor    =   None, 
         train_mode             :   bool            =   True):
    

    gradient = model(train_mode = train_mode,
                 grad_mode = True,
                 global_coordinates = evaluation_points)
    
    return gradient


def curl(model,
        evaluation_points     :    torch.Tensor   =    None,
        train_mode            :    bool           =    True):
    
     gradient = grad(model,
                    evaluation_points,
                    train_mode)
    
     if gradient.shape[-2] == 1:
        zero = torch.zeros_like(gradient)
        gradient = torch.cat([zero, zero, gradient], dim = -2)

     if gradient.shape[-1] == 2:
        zero = torch.zeros(size = (gradient.shape[0], gradient.shape[1], gradient.shape[2], 1), device = gradient.device, dtype = gradient.dtype)
        gradient = torch.cat([gradient, zero], dim = -1)
     
     curl_value = torch.zeros_like(gradient[:,:,:,0])

     curl_value[:,:,0] = gradient[:,:,2,1] - gradient[:,:,1,2]
     curl_value[:,:,1] = gradient[:,:,0,2] - gradient[:,:,2,0]
     curl_value[:,:,2] = gradient[:,:,1,0] - gradient[:,:,0,1]
    
     return curl_value