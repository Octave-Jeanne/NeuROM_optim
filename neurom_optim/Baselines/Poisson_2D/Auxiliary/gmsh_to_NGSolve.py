from netgen.read_gmsh import ReadGmsh
from ngsolve import Mesh

def gmsh_to_NGSolve(config):
    path_to_msh = config['Mesh']['path_to_msh']
    mesh = Mesh(ReadGmsh(path_to_msh))

    boundaries = mesh.GetBoundaries()
    dirichlet = ''
    for dirichlet_config in config['Dirichlet']:
        if dirichlet_config['name'] in boundaries:
            dirichlet += f"{dirichlet_config['name']}|"

    
    dirichlet = dirichlet[:-1]

    return mesh, dirichlet