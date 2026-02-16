from torch import nn
import torch
from ..Pre_processing.build_optimizer import build_optimizer

class Optimizer(nn.Module):
    def __init__(self, 
                 loss, 
                 model, 
                 Mat,
                 optimizer_name,
                 hyperparameters,
                source = None,
                 freeze_grid = False,
                 freeze_interpolation = False,
                 freeze_Mat = False,
                 freeze_Source = False):
        
        super().__init__()
        self.loss = loss
        self.model = model
        self.Mat = Mat
        self.source = source
        self.optimizer_name = optimizer_name

        self.freeze_grid = freeze_grid
        self.freeze_interpolation = freeze_interpolation
        self.freeze_Mat = freeze_Mat
        self.freeze_Source = freeze_Source

        self.optimizer = build_optimizer(optimizer_name, hyperparameters, model, Mat, source)

    def closure(self):
        self.optimizer.zero_grad()
        loss = self.loss(self.model, self.Mat, self.source, get_metrics = False)
        loss.backward(retain_graph = True)
        return loss
    
    def step(self):
        match self.optimizer_name:
            case 'lbfgs'|'step':
                self.optimizer.step(self.closure)
                loss = self.closure()
            case 'adam':
                loss = self.closure()
                self.optimizer.step()



        metrics = self.loss(self.model, self.Mat, self.source, get_metrics = True)
        metrics = self.post_process_metrics(metrics)
        return metrics
    
    def Apply_freeze(self):
        if self.model is not None:
            self.model.UnFreeze()
            self.model.Freeze(freeze_grid = self.freeze_grid,
                              freeze_interpolation = self.freeze_interpolation)
            
        if self.Mat is not None:
            self.Mat.UnFreeze()
            if self.freeze_Mat:
                self.Mat.Freeze()
            
    @staticmethod
    def post_process_metrics(metrics):
        for key in metrics.keys():
            if type(metrics[key]) == torch.Tensor: 
                metrics[key] = metrics[key].clone().detach()
        return metrics
    
    def export_vtk(self, epoch):
        self.loss.export_vtk(epoch, self.model, self.Mat, self.source)

