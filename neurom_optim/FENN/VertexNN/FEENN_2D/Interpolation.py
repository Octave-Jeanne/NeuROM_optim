from torch import nn
import torch



class InterpolationNN_2D(nn.Module):
    def __init__(self, NNodes, NElem, n_components, element, mapping, IntPrecision, FloatPrecision):
        super().__init__()

        self.register_buffer('ref_int', torch.tensor(1, dtype = IntPrecision))
        self.register_buffer('ref_float', torch.tensor(1, dtype = FloatPrecision))

        

        self.register_buffer('n_components', self.reformat_value(n_components))
        self.register_buffer('NNodes', self.reformat_value(NNodes))
        self.register_buffer('NElem', self.reformat_value(NElem))
        self.register_buffer('all_nodal_values',0.5*torch.ones((NNodes, n_components), dtype = self.ref_float.dtype))
        self.register_buffer('free', (torch.ones(self.NNodes)==1).bool())

        self.nodal_values =nn.ParameterDict({
                                                'free': self.all_nodal_values
                                                })
        
        self.element = element
        self.mapping = mapping


    def SetBCs(self, Fixed_Ids, Fixed_Values):

        Fixed_Ids = self.reformat_value(Fixed_Ids)
        Fixed_Values = self.reformat_value(Fixed_Values, dtype = self.ref_float.dtype)

        self.free[Fixed_Ids] = False
        self.all_nodal_values[Fixed_Ids] = Fixed_Values

        self.nodal_values['free'] = self.all_nodal_values[self.free]
        self.nodal_values['imposed'] = self.all_nodal_values[~self.free]

        self.Freeze()
        self.UnFreeze()

    def SetQuad(self, quadrature_order):
        self.element.SetQuad(quadrature_order)

    def forward(self, 
                nodal_coordinates, 
                connectivity,
                train_mode          :   bool            =   True,
                grad_mode           :   bool            =   True,
                global_coordinates  :   torch.Tensor    =   None):
        

        if train_mode:
            return self.train_forward(nodal_coordinates, connectivity, grad_mode)
        
        else:
            return self.eval_forward(nodal_coordinates, global_coordinates, grad_mode)
        
    def train_forward(self, nodal_coordinates, connectivity, grad_mode):

        nodal_values = torch.ones_like(self.all_nodal_values, dtype = self.ref_float.dtype)
        nodal_values[self.free] = self.nodal_values['free']
        nodal_values[~self.free] = self.nodal_values['imposed']

        


        if grad_mode:
            
            
            grad_shape_functions = self.element(grad_mode = True,
                                                train_mode = True)
            
            global_grad_shape_function = self.mapping(mode = 'grad',
                                                      nodal_coordinates = nodal_coordinates,
                                                      entity = grad_shape_functions)
            
            grad = torch.einsum('egnd,enc->egcd', global_grad_shape_function, nodal_values[connectivity])

            return grad




        else:
            gauss_coordinates, gauss_weights, shape_functions = self.element(grad_mode = False,
                                                                             train_mode = True)
            
            
        
            global_gauss_coordinates, det = self.mapping(mode = 'direct',
                                                         nodal_coordinates = nodal_coordinates,
                                                         entity = gauss_coordinates)
        
        
            shape_functions = shape_functions.unsqueeze(0)
            shape_functions = shape_functions.repeat(self.NElem, 1, 1)

            gauss_weights = gauss_weights.unsqueeze(0)
            gauss_weights = gauss_weights.repeat(self.NElem, 1, 1)

            # FEM interpolation (element x node x component, element x gauss x node ) -> element x gauss x component
            interpolation = torch.einsum('enc,egn->egc',nodal_values[connectivity], shape_functions)

            return global_gauss_coordinates, gauss_coordinates, gauss_weights, det, interpolation
    
    def eval_forward(self):
        """
        To do
        """
        pass


    def Freeze(self):
        """
        This function prevents any modification of nodal values during optimisation.
        """

        self.nodal_values['free'].requires_grad = False
        self.nodal_values['imposed'].requires_grad = False

    def UnFreeze(self):
        """
        Allows the free nodal values to be trained
        """

        self.nodal_values['free'].requires_grad = True
        self.nodal_values['imposed'].requires_grad = False



        

    def GetValues(self):
        """
        Stores the current nodal values for future post-processing plots related use
        """
        values = torch.ones_like(self.all_nodal_values, dtype = self.ref_float.dtype)
        values[self.free] = self.nodal_values['free']
        values[~self.free] = self.nodal_values['imposed']
        return values.clone().detach().cpu()
    
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
