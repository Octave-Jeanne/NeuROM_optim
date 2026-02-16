from torch import nn
import torch


class Regions_L2NN(nn.Module):
    """
    Module meant to manage region-based material properties (i.e property that is uniform by part) in the optic of material optimization
    """
    def __init__(self, 
                 regions_names, 
                 property_values, 
                 elements_in_regions, 
                 NElem,
                 IntPrecision,
                 FloatPrecision):
        """
        Initializes the nn.Module dependencies
        Initializes and stores each region of material property

        Args : 
            regions_names (str)
        """
        
        super().__init__()

        # Keep track of device and dtype used throughout the model
        self.register_buffer('ref_int', torch.tensor(1, dtype = IntPrecision))
        self.register_buffer('ref_float', torch.tensor(1, dtype = FloatPrecision))

        self.register_buffer('NElem', torch.tensor(NElem))
        
        self.regions = nn.ModuleDict()

        for region_index, region_name in enumerate(regions_names):
            self.regions[region_name] = Region_L2NN(elements_in_region = elements_in_regions[region_index],
                                                                    value = property_values[region_index],
                                                                    NElem = NElem,
                                                                    IntPrecision = IntPrecision,
                                                                    FloatPrecision = FloatPrecision)
    
    def forward(self, el_ids, *args, **kwargs):
        """
        Returns the material property at the specified element ids

        Args :
            el_ids (Number of specified elements ids sized torch.tensor) : The ids of the elements where the property value should be evaluated
        
        Returns :
            property_values (Number of specified elements ids sized torch.tensor) : The property value at the ids of the elements where the property value should be evaluated
        """
        property_values = torch.zeros(len(el_ids), dtype = self.ref_float.dtype)
        for region_name in self.regions.keys():
            property_values = self.regions[region_name](el_ids, property_values)
        return property_values
       

    def setBCs(self, region_names):
        """
        Permanently fixes the nodes specified in the Boundary conditions
        """
        for region_name in region_names:
            self.regions[region_name].setBCs()
            

    def Freeze(self):
        """
        This function prevents any modification of node coordinates during optimisation.
        """

        for region_name in self.regions.keys():
            self.regions[region_name].Freeze()

    def UnFreeze(self):
        """
        Allows the free nodes to be trained
        """
        
        for region_name in self.regions.keys():
            self.regions[region_name].UnFreeze()



class Region_L2NN(nn.Module):
    def __init__(self, elements_in_region, value, NElem, IntPrecision, FloatPrecision):
        super().__init__()
        self.register_buffer('ref_int', torch.tensor(1, dtype = IntPrecision))
        self.register_buffer('ref_float', torch.tensor(1, dtype = FloatPrecision))

        
        self.register_buffer('elements_in_region', elements_in_region)

        self.register_buffer('free', torch.tensor(True))
        self.register_buffer('NElem', torch.tensor(NElem))
        
        self.value = nn.Parameter(torch.tensor(value, requires_grad = True, dtype = self.ref_float.dtype))

    def setBCs(self):
        self.free = torch.tensor(False)
        self.value.requires_grad = False
    
    def forward(self, el_ids, property_values):
        """
        Returns the material property at the specified element ids

        Args :
            el_ids (Number of specified elements ids sized torch.tensor) : The ids of the elements where the property value should be evaluated
        
        Returns :
            property_values (Number of specified elements ids sized torch.tensor) : The property value at the ids of the elements where the property value should be evaluated
        """

        # Assign self value to the property assigned to the el_ids that are present in self.elements
        # self.elements_in_region[el_ids] --> The boolean value of el_ids in self.elements aka "is el_ids in the region ?"
        # property_values[self.elements_in_region[el_ids]] = self.value --> The property of the el_ids that are in the region is set to the region value

        property_values[self.elements_in_region[el_ids]] = self.value
        return property_values
        
    def Freeze(self):
        self.value.requires_grad = False

    def UnFreeze(self):
        if self.free:
            self.value.requires_grad = True