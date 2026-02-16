import torch
from torch import nn



class PropertiesNN(nn.Module):
    """
    Module meant to manage material properties (i.e PDE conditionning parameters) in the optic of material optimization

    Stores and manages different material properties
    """

    def __init__(self, 
                 dim,
                 NElem,
                 IntPrecision,
                 FloatPrecision):
        """
        Initialize the nn.Module dependencies

        Stores whatever is necessary to manage material properties
        """

        super().__init__()

        # Keep track of device and dtype used throughout the model
        self.register_buffer('ref_int', torch.tensor(1, dtype = IntPrecision))
        self.register_buffer('ref_float', torch.tensor(1, dtype = FloatPrecision))

        self.properties = nn.ModuleDict()

        self.register_buffer('dim', torch.tensor(int(dim)))
        self.register_buffer('NElem', torch.tensor(NElem))

    

    def add_property(self,
                     property_name,
                     property):
        self.properties[property_name] = property

    def forward(self, 
                property_name, 
                el_ids,
                NPoints,
                **kwargs):
        """
        Returns the specified material property at the specified points or elements

        Args : 
            property_name (str) : The material property that should be considered
            x (Number of points x Number of coordinates sized torch.tensor) : The global coordinates of the points where the material property should be evaluated
            el_ids (Number of specified elements sized torch.tensor) : The ids of the elements where the material property should be evaluated

        Returns :
            property (Number of specified points or elements sized torch.tensor) : The material property value at each specified point or element
        """
        

        # Call the appropriated material property NN at the specified el_ids to get its values
        return self.properties[property_name](el_ids, NPoints, **kwargs)
        
        
    def setBCs(self, 
               properties_names   =   None, 
               Fixed_by_property  =   None):
        """
        Permanently fixes the properties values of a the specified property NN at the specified areas in the Boundary conditions

        Args : 
            properties_names (list of str) : The names of the properties where BCs should be enforced
            Fixed_by_property (list of either int or str) : How each BC should be enforced
                - case (list of int) : The indexes of the elements where the property value should be imposed (i.e permanently frozen)
                - case (list of str) : The name of the regions where the property value should be imposed (i.e permanently frozen)
        """

        for property_index, property_name in enumerate(properties_names):
            self.properties[property_name].setBCs(Fixed_by_property[property_index])
        
        
        

    def Freeze(self, property_name = None):
        """
        This function prevents any modification of properties values during optimisation.
        """

        if property_name is not None:
            self.properties[property_name].Freeze()
            
        else:
            for property in self.properties.values():
                property.Freeze()
       
    def UnFreeze(self, property_name = None):
        """
        Allows the free properties values to be trained
        """

        if property_name is not None:
            self.properties[property_name].UnFreeze()

        else:
            for property in self.properties.values():
                property.UnFreeze()

    def StoreResults(self):
        for property in self.properties.values():
                property.StoreResults()