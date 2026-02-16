from torch import nn
import torch

class BinarySelection(nn.Module):
    def __init__(self, 
                 property_1, 
                 property_2,
                 NElem,
                 IntPrecision,
                 FloatPrecision):
        
        super().__init__()


        self.register_buffer('ref_int', torch.tensor(1, dtype = IntPrecision))
        self.register_buffer('ref_float', torch.tensor(1, dtype = FloatPrecision))

        #self.register_buffer('all_soft_choice', -10*torch.ones(NElem, dtype = self.ref_float.dtype))
        self.register_buffer('all_soft_choice', 0.5*torch.ones(NElem, dtype = self.ref_float.dtype))
        self.register_buffer('free', torch.tensor([True for _ in self.all_soft_choice]))

        


        self.soft_choice = nn.ParameterDict({
                                                'free': self.all_soft_choice,
                                                
                                                })
        
        self.properties = nn.ModuleDict({'property_1' : property_1,
                                         'property_2' : property_2})
        
        self.sigmoid = nn.Sigmoid()

        self.soft_choice_interm = []
        self.hard_choice_interm = []
        self.soft_property_interm = []
        self.hard_property_interm = []

        self.hard_choice  = False


    def set_hard_choice(self, value : bool):
        self.hard_choice = value
    
    def forward(self, el_ids, NPoints, hard_choice = False, return_choice = False, *args, **kwargs):

        choice = torch.ones_like(self.all_soft_choice, dtype = self.ref_float.dtype)
        choice[self.free] = self.sigmoid(self.soft_choice['free'] - 0.5)
        choice[~self.free] = self.soft_choice['imposed']
        choice = choice[el_ids]
        if self.hard_choice:
            # In eval mode the Binary selection becomes "hard"
            choice = (choice>0.5).int()
        choice = choice.unsqueeze(1)
        choice = choice.repeat(1, NPoints)
        choice = choice.unsqueeze(-1)

        if return_choice:
            return choice
        
        else:
            return (1-choice)*self.properties['property_1'](el_ids, NPoints, *args, **kwargs) + choice*self.properties['property_2'](el_ids, NPoints, *args, **kwargs)
    
    def setBCs(self, Fixed_Ids, specific_value):
        """
        Permanently fixes the nodes specified in the Boundary conditions
        """
        Fixed_Ids = self.reformat_value(Fixed_Ids)
        specific_value = self.reformat_value(specific_value, dtype = self.ref_float.dtype)

        self.free[Fixed_Ids] = False

        

        self.all_soft_choice[Fixed_Ids] = specific_value


        self.soft_choice['free'] = self.all_soft_choice[self.free]
        self.soft_choice['imposed'] = self.all_soft_choice[~self.free]
        self.Freeze()
        self.UnFreeze()
    

    def Freeze(self):
        """
        This function prevents any modification of node coordinates during optimisation.
        """

        self.soft_choice['free'].requires_grad = False
        self.soft_choice['imposed'].requires_grad = False

    def UnFreeze(self):
        """
        Allows the free nodes to be trained
        """

        self.soft_choice['free'].requires_grad = True
        self.soft_choice['imposed'].requires_grad = False

    def get_choice(self):
        soft_choice = torch.ones_like(self.all_soft_choice, dtype = self.ref_float.dtype)
        soft_choice[self.free] = self.sigmoid(self.soft_choice['free'])
        soft_choice[~self.free] = self.soft_choice['imposed']

        hard_choice = (soft_choice>0.5).int()

        el_ids = torch.arange(0, len(soft_choice))
        NPoints = 1

        # soft_property = (1-soft_choice)*self.properties['property_1'](el_ids, NPoints) + soft_choice*self.properties['property_2'](el_ids, NPoints)
        # hard_property = (1-hard_choice)*self.properties['property_1'](el_ids, NPoints) + hard_choice*self.properties['property_2'](el_ids, NPoints)

        return soft_choice, hard_choice # , soft_property, hard_property
    
    def StoreResults(self):
        # soft_choice, hard_choice, soft_property, hard_property = self.get_choice()
        soft_choice, hard_choice = self.get_choice()
        soft_choice, hard_choice = self.get_choice()
        soft_choice = soft_choice.clone().detach().cpu()
        hard_choice = hard_choice.clone().detach().cpu()
        # soft_property = soft_property.clone().detach().cpu()
        # hard_property = hard_property.clone().detach().cpu()

        
        self.soft_choice_interm.append(soft_choice)
        self.hard_choice_interm.append(hard_choice)
        # self.soft_property_interm.append(soft_property)
        # self.hard_property_interm.append(hard_property)

    
    def reformat_value(self, value, dtype = None):
        if dtype is not None:
            if type(value) == torch.Tensor:
                value = value.clone().detach().to(dtype)

            else:
                value = torch.tensor(value, dtype = dtype)

        elif type(value) == int:
            value = torch.tensor(value, dtype = self.ref_int.dtype)

        elif type(value) == float:
            value = torch.tensor(value, dtype = self.ref_float.dtype)
            
        elif type(value) == bool:
            value = torch.tensor(value)
            
        elif type(value) == torch.Tensor:
            value = value.clone().detach()
        
        return value