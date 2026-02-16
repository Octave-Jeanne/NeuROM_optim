from ngsolve import H1, GridFunction, Integrate, dx, grad
from ..Auxiliary.gmsh_to_NGSolve import gmsh_to_NGSolve
from ..Auxiliary.process_source import process_source_term
from ..Auxiliary.process_field import process_field_term


def baseline(config):
    mesh, dirichlet = gmsh_to_NGSolve(config)
    fes = H1(mesh, order = 1, dirichlet = dirichlet)

    u = fes.TrialFunction()
    v = fes.TestFunction()

    # Change this to account for custom material properties
    bf = process_field_term(config = config, 
                            fes = fes, 
                            u = u, 
                            v = v)
    
    lf = process_source_term(config = config, 
                             fes = fes, 
                             v = v)
    bf.Assemble()
    lf.Assemble()

    # Solve
    sol = GridFunction(fes)
    sol.vec.data = bf.mat.Inverse(fes.FreeDofs(), inverse = "sparsecholesky")*lf.vec

    energy_source = process_source_term(config = config, 
                                        v = sol,
                                        mesh = mesh,
                                        get_energy = True)
    
    energy_field = process_field_term(config = config, 
                                       u = sol,
                                       mesh = mesh,
                                       get_energy = True)
    
    energy = energy_field - energy_source
    return mesh, sol, energy







def get_energy(config, sol, mesh):
    source_config = config['Source'][0]
    match source_config['type']:

        case 'homogeneous':
            return get_energy_homogeneous_source(config, sol, mesh) 
    pass


        
def get_energy_homogeneous_source(config, sol, mesh) :
    f = config['Source'][0]['Value']
    mat_value  = config['Material'][0]['Value']

    energy = Integrate((0.5 * mat_value *  grad(sol)*grad(sol) - f*sol)*dx, mesh)

    return energy

