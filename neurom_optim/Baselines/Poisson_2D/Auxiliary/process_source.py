from ngsolve import Integrate, dx, LinearForm

def process_source_term(config = None, fes = None, v = None, get_energy = False, mesh = None):
    source_config = config['Source'][0]
    match source_config['type']:

        case 'homogeneous':
            return process_homogeneous_source_term(config, fes, v, get_energy, mesh)
        
        

def process_homogeneous_source_term(config, fes, v, get_energy, mesh):
    if get_energy:
        energy = Integrate((config['Source'][0]['Value']*v)*dx, mesh)
        return energy
    
    else:
        lf = LinearForm(fes)
        lf += config['Source'][0]['Value']*v*dx
        return lf