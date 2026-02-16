from torch import nn
import torch

class ConstantNN(nn.Module):
    """
    Module meant to manage uniform material properies in the optic of material optimization
    """

    def __init__(self, 
                 property_value,
                 IntPrecision,
                 FloatPrecision):
        """
        Initializes the nn.Module dependencies and the material property value
        """
        super().__init__()

        # Keep track of device and dtype used throughout the model
        self.register_buffer('ref_int', torch.tensor(1, dtype = IntPrecision))
        self.register_buffer('ref_float', torch.tensor(1, dtype = FloatPrecision))

       
        
        self.property = nn.Parameter(self.reformat_value(property_value, dtype = self.ref_float.dtype))

    def forward(self, el_ids, NPoints, *args, **kwargs):
        """
        Returns the material property at the specified element ids

        Args :
            el_ids (Number of specified elements ids sized torch.tensor) : The ids of the elements where the property value should be evaluated
        
        Returns :
            property (Number of specified elements ids sized torch.tensor) : The property value at the ids of the elements where the property value should be evaluated
        """

        property = self.property*torch.ones(len(el_ids), dtype = self.ref_float.dtype, device = self.ref_float.device)
        property = property.unsqueeze(1)
        property = property.repeat(1, NPoints)
        return property.unsqueeze(-1)
        
    def setBCs(self, is_fixed):
        """
        Permanently fixes the property if it is specified
        """

        self.is_fixed = is_fixed
        if is_fixed :
            self.Freeze()
        
        else:
            self.UnFreeze()

    def Freeze(self):
        """
        This function prevents any modification of node coordinates during optimisation.
        """

        self.property.requires_grad = False

    def UnFreeze(self):
        """
        Allows the free nodes to be trained
        """

        if not self.is_fixed:
            self.property.requires_grad = True

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
       