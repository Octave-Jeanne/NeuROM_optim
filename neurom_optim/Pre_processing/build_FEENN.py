
from ..FEENN.VertexNN.FEENN_2D.FEENN import FEENN_2D
from ..FEENN.VertexNN.FEENN_2D.Element import Tri_2D_lin
from ..FEENN.VertexNN.FEENN_2D.Mapping import Mapping_2D_Affine



import torch

from .process_hardware import get_precision

#################################################################################################################################
#################################################################################################################################
###                                                                                                                           ###
###                                                        build FEENN_2D                                                     ###
###                                                                                                                           ###
#################################################################################################################################
#################################################################################################################################

def build_FEENN_2D(mesh, config):
    IntPrecision, FloatPrecision = get_precision(config)
    Nodes = mesh.nodes
    dim = config['interpolation']['dim']
    connectivity = mesh.elements[str(dim)]['connectivity']
    model =  FEENN_2D(Nodes = Nodes,
                      connectivity = connectivity,
                      n_components = config['interpolation']['n_components'],
                      element = get_element(config, IntPrecision, FloatPrecision),
                      mapping = get_mapping(config),
                      IntPrecision = IntPrecision,
                      FloatPrecision = FloatPrecision
                      )
    
    setBCs(model = model,
           mesh = mesh, 
           config = config,
           IntPrecision = IntPrecision,
           FloatPrecision = FloatPrecision)
    
    model.SetQuad(quadrature_order = config['interpolation']['quadrature_order'])

    return model

    

def get_element(config, IntPrecision, FloatPrecision):
    match config['interpolation']['element_type']:
        case 'Tri_2D_lin':
            return  Tri_2D_lin(IntPrecision, FloatPrecision)
        
def get_mapping(config):
    match config['interpolation']['mapping']:
        case 'Mapping_2D_Affine':
            return Mapping_2D_Affine()


def setBCs(model, mesh, config, IntPrecision, FloatPrecision):
    Fixed_nodal_coordinates_Ids = []
    Fixed_nodal_values_Ids = []
    Fixed_nodal_values_values = []

    # Might be changed later
    nodeIDs = torch.tensor([])
    values = torch.tensor([])
    for dirichletBC in config['DirichletDictionryList']:

        
        for key in mesh.PhysicalEntities[str(dirichletBC['Entity'])].keys():
            if key not in ['dim', 'name']:
                nodeIDs, values = update_while_processing_redundancies(nodeIDs = nodeIDs, 
                                                                       values = values, 
                                                                       new_IDs = mesh.PhysicalEntities[str(dirichletBC['Entity'])][key]['nodeIDs'], 
                                                                       dirichlet_value = dirichletBC['Value'], 
                                                                       tag = str(dirichletBC['Entity']),
                                                                       IntPrecision = IntPrecision,
                                                                       FloatPrecision = FloatPrecision)
                


    Fixed_nodal_coordinates_Ids = nodeIDs.to(IntPrecision)
    Fixed_nodal_values_Ids = nodeIDs.to(IntPrecision)
    Fixed_nodal_values_values = values.unsqueeze(1)

    model.SetBCs(Fixed_nodal_coordinates_Ids = Fixed_nodal_coordinates_Ids,
                 Fixed_nodal_values_Ids = Fixed_nodal_values_Ids,
                 Fixed_nodal_values_values = Fixed_nodal_values_values)


def update_while_processing_redundancies(nodeIDs, values, new_IDs, dirichlet_value, tag, IntPrecision, FloatPrecision):
    ideal_updated_nodeIDs = torch.hstack([nodeIDs, new_IDs])
    updated_nodeIDs = torch.unique(ideal_updated_nodeIDs)

    if len(ideal_updated_nodeIDs) == len(updated_nodeIDs):
        new_values = torch.tensor([dirichlet_value for _ in new_IDs], dtype = FloatPrecision)
        
    
    else:
        print(f'\nRedundancy found at Dirichlet tag {tag}. The redundant value will be ignored\n')
        new_values = torch.tensor([dirichlet_value for _ in range(len(new_IDs) - (len(ideal_updated_nodeIDs) - len(updated_nodeIDs)))], dtype = FloatPrecision)
    
    updated_values =  torch.hstack([values, new_values])
    return updated_nodeIDs, updated_values