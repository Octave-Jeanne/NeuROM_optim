from ngsolve import Integrate, dx, grad, BilinearForm

def process_field_term(config = None, fes = None, u = None, v = None, get_energy = False, mesh = None):

    match config['Material'][0]['type']:
        case 'field':
            return process_field_term_field_material(config, fes, u, v, get_energy, mesh)

        case 'constant':
            return process_field_term_constant_material(config, fes, u, v, get_energy, mesh)

    

def process_field_term_field_material(config, fes, u, v, get_energy, mesh):
    if get_energy:
        energy = 0
        for region_index, region_name in enumerate(config['Material'][0]['regions_names']): 
            energy += Integrate((0.5 * config['Material'][0]['Value'][region_index] *  grad(u)*grad(u))*dx(region_name), mesh)
        return energy

    else:
        bf = BilinearForm(fes)
        for region_index, region_name in enumerate(config['Material'][0]['regions_names']):
            bf += (config['Material'][0]['Value'][region_index]*grad(u)*grad(v))*dx(region_name)
        return bf


def process_field_term_constant_material(config, fes, u, v, get_energy, mesh):
    energy = 0
    if get_energy:
        mat_value = config['Material'][0]['Value']
        energy += Integrate((0.5 * mat_value *  grad(u)*grad(u))*dx, mesh)
        return energy

    else:
        mat_value = config['Material'][0]['Value']

        bf = BilinearForm(fes)
        bf += (mat_value*grad(u)*grad(v))*dx
        return bf