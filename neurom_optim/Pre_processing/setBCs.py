import torch

def SetBcs(model, mesh, config):
    for dirichlet in config["DirichletDictionryList"]:
        for key in mesh.PhysicalNames.keys():
            if dirichlet['Entity'] == mesh.PhysicalNames[key]['tag']:
                Fixed_nodal_coordinates_Ids = torch.tensor(mesh.PhysicalNames[key]['element_tags'], dtype = torch.int)
                Fixed_nodal_values_Ids =  torch.tensor(mesh.PhysicalNames[key]['element_tags'], dtype = torch.int)