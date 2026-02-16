from torch import nn
import torch

from ..FENN.ConstantNN.constantNN import ConstantNN


class NL_nu(nn.Module):
    def __init__(self, nu_0, K, k, IntPrecision, FloatPrecision):
        super().__init__()
        self.nu_0 = ConstantNN(property_value = nu_0,
                               IntPrecision = IntPrecision,
                               FloatPrecision = FloatPrecision)
        
        self.K = ConstantNN(property_value = K,
                            IntPrecision = IntPrecision,
                            FloatPrecision = FloatPrecision)
        
        self.k = ConstantNN(property_value = k,
                            IntPrecision = IntPrecision,
                            FloatPrecision = FloatPrecision)
        
    def forward(self, el_ids, NPoints, B):
        nu_0 = self.nu_0(el_ids, NPoints)
        K = self.K(el_ids, NPoints)
        k = self.k(el_ids, NPoints)


        return nu_0 + K*torch.exp(k*((B**2).sum(-1).unsqueeze(-1)))


    def setBCs(self, parameter_name = None, is_fixed = True):
        if parameter_name is None:
            self.nu_0.setBCs(is_fixed = is_fixed)
            self.K.setBCs(is_fixed = is_fixed)
            self.k.setBCs(is_fixed = is_fixed)
        
        else:
            match parameter_name:
                case 'nu_0':
                    self.nu_0.setBCs(is_fixed = is_fixed)
                case 'K':
                    self.K.setBCs(is_fixed = is_fixed)
                case 'k':
                    self.k.setBCs(is_fixed = is_fixed)
    
    def Freeze(self, parameter_name = None):
        if parameter_name is None:
            self.nu_0.Freeze()
            self.K.Freeze()
            self.k.Freeze()
        
        else:
            match parameter_name:
                case 'nu_0':
                    self.nu_0.Freeze()
                case 'K':
                    self.K.Freeze()
                case 'k':
                    self.k.Freeze()


    def UnFreeze(self, parameter_name = None):
        if parameter_name is None:
            self.nu_0.UnFreeze()
            self.K.UnFreeze()
            self.k.UnFreeze()
        
        else:
            match parameter_name:
                case 'nu_0':
                    self.nu_0.UnFreeze()
                case 'K':
                    self.K.UnFreeze()
                case 'k':
                    self.k.UnFreeze()

    
    