from torch import nn
import torch

class Field_L2NN(nn.Module):
    """
    Module meant to manage material properties that take the form of a scalar field in the optic of material optimization
    """

    def __init__(self, 
                 property_values,
                 IntPrecision,
                 FloatPrecision):
        """
        Initializes the nn.Module dependencies and the property scalar field value

        Args : 
            property_values (Number of elements sized torch.tensor) : the material property value at each element of the mesh
        """


        super().__init__()

        # Keep track of device and dtype used throughout the model
        self.register_buffer('ref_int', torch.tensor(1, dtype = IntPrecision))
        self.register_buffer('ref_float', torch.tensor(1, dtype = FloatPrecision))

        self.register_buffer('property_all', self.reformat_value(property_values, dtype = self.ref_float.dtype))
        self.register_buffer('free', (torch.ones_like(self.property_all)==1))
        self.property =nn.ParameterDict({
                                                'free': self.property_all,
                                                'imposed': [],
                                                'mask':[]
                                                })
    
    def forward(self, el_ids, NPoints, *args, **kwargs):
        """
        Returns the material property at the specified element ids

        Args :
            el_ids (Number of specified elements ids sized torch.tensor) : The ids of the elements where the property value should be evaluated
        
        Returns :
            property_values (Number of specified elements ids sized torch.tensor) : The property value at the ids of the elements where the property value should be evaluated
        """

        property_all = torch.ones_like(self.property_all, dtype = self.ref_float.dtype)
        property_all[self.free] = self.property['free']
        property_all[~self.free] = self.property['imposed']

        property = property_all[el_ids]
        property = property.unsqueeze(1)
        property = property.repeat(1, NPoints)
        return property.unsqueeze(-1)

    def setBCs(self, Fixed_Ids, specific_value = None):
        """
        Permanently fixes the nodes specified in the Boundary conditions
        """

        Fixed_Ids = self.reformat_value(Fixed_Ids)
        specific_value = self.reformat_value(specific_value)

        self.free[Fixed_Ids] = False

        if specific_value is not None:
            self.property_all[~self.free] = specific_value


        self.property['free'] = self.property_all[self.free]
        self.property['imposed'] = self.property_all[~self.free]
        self.property['imposed'].requires_grad = False

    def Freeze(self):
        """
        This function prevents any modification of node coordinates during optimisation.
        """

        self.property['free'].requires_grad = False
        self.property['imposed'].requires_grad = False

    def UnFreeze(self):
        """
        Allows the free nodes to be trained
        """

        self.property['free'].requires_grad = True
        self.property['imposed'].requires_grad = False

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