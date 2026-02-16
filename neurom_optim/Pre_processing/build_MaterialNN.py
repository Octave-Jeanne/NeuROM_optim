from ..FEENN.Material import MaterialNN
from .process_hardware import get_precision
import torch

#################################################################################################################################
#################################################################################################################################
###                                                                                                                           ###
###                                                        build MaterialNN                                                   ###
###                                                                                                                           ###
#################################################################################################################################
#################################################################################################################################


def build_Mat(mesh, config):
    dim = config['interpolation']['dim']
    NElem = len(mesh.elements[str(dim)]['connectivity'])
    IntPrecision, FloatPrecision = get_precision(config)

    Mat = MaterialNN(
        dim             = config['interpolation']['dim'], 
        NElem           = NElem,
        IntPrecision    = IntPrecision,
        FloatPrecision  = FloatPrecision)

    for material_config in config['material']:
        match material_config['material_type']:
            case 'region':
                add_region_properties(mesh, Mat, material_config, NElem)

            case 'field':
                add_field_properties(mesh, Mat, material_config, NElem)

            case 'constant':
                add_constant_properties(mesh, Mat, material_config, NElem)
    
    return Mat


def add_region_properties(mesh, Mat, material_config, NElem):
    elements_in_regions = []
    for region_name in material_config['regions_names']:
        elements_in_regions.append(get_element_in_region(region_name=region_name, mesh = mesh, NElem = NElem))

    for key in material_config.keys():
        if (key != 'regions_names') and (key != 'material_type') and (key != 'free'):
            Mat.add_property(property_name = key,
                             regions_names = material_config['regions_names'],
                             property_values = material_config[key],
                             elements_in_regions = elements_in_regions)
            
            # Might change later for custom variable Mat
            Mat.setBCs([key], [material_config['regions_names']])
            Mat.Freeze()
            

def add_field_properties(mesh, Mat, material_config, NElem):
    property_values = torch.zeros(NElem)
    
    for key in material_config.keys():
        if (key != 'regions_names') and (key != 'material_type') and (key != 'free'):
            for region_index, region_name in enumerate(material_config['regions_names']):
                boolean_elements_in_region = get_element_in_region(region_name=region_name, mesh = mesh, NElem = NElem)
                property_values[boolean_elements_in_region] = material_config[key][region_index]

            Mat.add_property(property_name = key, 
                            property_values = property_values)
            
            # Might change later for custom variable Mat
            Fixed_by_property = [torch.ones(NElem).bool()]
            Mat.setBCs(properties_names = [key],
                       Fixed_by_property = Fixed_by_property)
            
def add_constant_properties(mesh, Mat, material_config, NElem):
    for key in material_config.keys():
        if (key != 'regions_names') and (key != 'material_type') and (key != 'free'):
            Mat.add_property(property_name = key,
                             property_values = torch.tensor(material_config[key]))
            
            # Might change later for custom variable Mat
            Mat.setBCs([key], [material_config['free']])




def get_element_in_region(region_name, mesh, NElem):
    elemIDs = None
    element_in_region = torch.zeros(NElem).bool()
    for tag in mesh.PhysicalEntities.keys():
        entity_name = mesh.PhysicalEntities[tag]['name'].strip('""')
        if entity_name == region_name:

            elemIDs = mesh.PhysicalEntities[tag]['element_type_2']['elemIDs']

    if elemIDs is not None:
        element_in_region[elemIDs] = True

    return element_in_region
    

