from .optimizer import Optimizer
from .stop_criterion import Stop_criterion
from time import time
import matplotlib.pyplot as plt
import numpy as np
import torch

class Solver():
    def __init__(self,
                 loss,
                 model,
                 Mat,
                 optimizer_name,
                 hyperparameters,
                 stop_criteria_definitions,
                source = None,
                 freeze_grid = False,
                 freeze_interpolation = False,
                 freeze_Mat = False,
                 freeze_Source = False):    
        
        self.epochs = []
        self.times = []

        self.optimizer = Optimizer(loss = loss,
                                   model = model,
                                   Mat = Mat,
                                   source = source,
                                   optimizer_name = optimizer_name,
                                   hyperparameters = hyperparameters,
                                   freeze_grid = freeze_grid,
                                   freeze_interpolation = freeze_interpolation,
                                   freeze_Mat = freeze_Mat,
                                   freeze_Source = freeze_Source)
        
        self.stop_criterion = Stop_criterion(stop_criteria_definitions = stop_criteria_definitions)


    def solve(self, current_epoch = 0, current_time = 0):

        self.optimizer.Apply_freeze()

        local_epoch = 0
        self.metrics_evolution = []
        t0 = time()
        metrics = {}

        while not self.stop_criterion(local_epoch, metrics):
            local_epoch += 1
            
            metrics = self.optimizer.step()
            metrics['time'] = time() - t0 + current_time
            self.metrics_evolution.append(metrics)
            self.times.append(time() - t0 + current_time)
            self.epochs.append(local_epoch + current_epoch)
            self.export_vtk(local_epoch + current_epoch)

            if torch.isnan(metrics['loss']):
                raise ValueError(f"Loss is {metrics['loss']}") 

    def plot_metric(self, 
                    name = 'loss',
                    exact_value = None,
                    use_time = False):
        
        if exact_value is None :
            metric = [metric[name].cpu().data.numpy() for metric in self.metrics_evolution]
        
        else:
            metric = [metric[name].cpu().data.numpy() - exact_value for metric in self.metrics_evolution]

        if use_time :
            plt.plot(self.times, metric)

        else:
            plt.plot(self.epochs, metric)

    def get_metric(self, 
                    name = 'loss',
                    exact_value = None,
                    use_time = False):
        
        if exact_value is None :
            metric = [metric[name].cpu().data.numpy() for metric in self.metrics_evolution]
        
        else:
            metric = [metric[name].cpu().data.numpy() - exact_value for metric in self.metrics_evolution]

        if use_time :
            return self.times, metric

        else:
            return self.epochs, metric

    
    def export_vtk(self, epoch):
        self.optimizer.export_vtk(epoch)

        
        
