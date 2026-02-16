import torch
from tqdm import tqdm
import numpy as np

##########################################################################################################################
##########################################################################################################################
###                                                                                                                    ###
###                                                   Criteria wrapper                                                 ###
###                                                                                                                    ###
##########################################################################################################################
##########################################################################################################################

class Stop_criterion():
    def __init__(self, stop_criteria_definitions):
        self.criteria = []
        for criterion_name in stop_criteria_definitions.keys():
            self.criteria.append(self.build_criterion(criterion_name, stop_criteria_definitions[criterion_name]))

    

    def __call__(self, epoch, metrics):
        stop = False
        for criterion in self.criteria:
            stop = stop|criterion(epoch, metrics)
        
        return stop
    
    @staticmethod
    def build_criterion(criterion_name, criterion_definition):
        match criterion_name:
            case 'MaxEpochs':
                return MaxEpochs(max_epoch = criterion_definition.get('max_epoch', 100))

            case 'Stagnation':
                return Stagnation(loss_decrease_c = criterion_definition.get('loss_decrease_c', 0))

            # etc ...

    
##########################################################################################################################
##########################################################################################################################
###                                                                                                                    ###
###                                                   Specific riteria                                                 ###
###                                                                                                                    ###
##########################################################################################################################
##########################################################################################################################

class MaxEpochs():
    def __init__(self, max_epoch, show_all_metrics = False):
        self.max_epoch = max_epoch
        self.progress_bar = tqdm(total = max_epoch, desc = "Solving (worse case scenario)", leave = True)
        self.show_all_metrics = show_all_metrics

    def __call__(self, epoch, metrics):
        self.progress_bar.update(1)
        if self.show_all_metrics:
            self.progress_bar.set_postfix(metrics)
        
        else:
            self.progress_bar.set_postfix({'time' : metrics.get('time', 0)})

        return epoch >= self.max_epoch
    
class Stagnation():
    def __init__(self, loss_decrease_c):
        self.loss_old = None
        self.loss_decrease_c = loss_decrease_c

    def __call__(self, epoch, metrics):

        if metrics.get('loss', None) is None:
            return False

        else:
            current_loss = metrics['loss']
            if self.loss_old is None:
                self.loss_old = current_loss
                return False
            
            else:
                d_loss = 2*(torch.abs(current_loss - self.loss_old))/(torch.abs(current_loss + self.loss_old))
                self.loss_old = current_loss
                return d_loss < self.loss_decrease_c
        

