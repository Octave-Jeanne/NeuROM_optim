from torch import nn
import torch

from ..FENN.ConstantNN.constantNN import ConstantNN

import numpy as np







 




class NL_nu_altair(nn.Module):
    def __init__(self, mur0, Jsat, a, IntPrecision, FloatPrecision):
        super().__init__()

        



        
        self.mur0 = ConstantNN(property_value = mur0,
                               IntPrecision = IntPrecision,
                               FloatPrecision = FloatPrecision)
        
        self.Jsat = ConstantNN(property_value = Jsat,
                            IntPrecision = IntPrecision,
                            FloatPrecision = FloatPrecision)
        
        self.a = ConstantNN(property_value = a,
                            IntPrecision = IntPrecision,
                            FloatPrecision = FloatPrecision)
        


    def forward(self, el_ids, NPoints, B):
        mur0 = self.mur0(el_ids, NPoints)
        Jsat = self.Jsat(el_ids, NPoints)
        a = self.a(el_ids, NPoints)

        slp = 1-1/mur0
        dab = slp/Jsat

        normB = (B**2).sum(-1).unsqueeze(-1)

        mu0 = 4e-7*np.pi
        return (normB - self.mScalar(normB, slp, dab, a, Jsat))/(mu0*normB+1e-8)


        # nu_0 = self.nu_0(el_ids, NPoints)
        # K = self.K(el_ids, NPoints)
        # k = self.k(el_ids, NPoints)
        # return nu_0 + K*torch.exp(k*((B**2).sum(-1).unsqueeze(-1)))


    def setBCs(self, parameter_name = None, is_fixed = True):
        if parameter_name is None:
            self.mur0.setBCs(is_fixed = is_fixed)
            self.Jsat.setBCs(is_fixed = is_fixed)
            self.a.setBCs(is_fixed = is_fixed)
        
        else:
            match parameter_name:
                case 'mur0':
                    self.mur0.setBCs(is_fixed = is_fixed)
                case 'Jsat':
                    self.Jsat.setBCs(is_fixed = is_fixed)
                case 'a':
                    self.a.setBCs(is_fixed = is_fixed)
    
    def Freeze(self, parameter_name = None):
        if parameter_name is None:
            self.mur0.Freeze()
            self.Jsat.Freeze()
            self.a.Freeze()
        
        else:
            match parameter_name:
                case 'mur0':
                    self.mur0.Freeze()
                case 'Jsat':
                    self.Jsat.Freeze()
                case 'a':
                    self.a.Freeze()


    def UnFreeze(self, parameter_name = None):
        if parameter_name is None:
            self.mur0.UnFreeze()
            self.Jsat.UnFreeze()
            self.a.UnFreeze()
        
        else:
            match parameter_name:
                case 'mur0':
                    self.mur0.UnFreeze()
                case 'Jsat':
                    self.Jsat.UnFreeze()
                case 'a':
                    self.a.UnFreeze()

    def mScalar(self, normB, slp, dab, a, Jsat) : 
        aa = dab*normB
        bb = aa + 1
        cc =  (1-a)
        return Jsat/(2*cc) * ( bb -torch.sqrt( bb**2 - 4*aa * cc ) )

  
        

    
    