import pyvista as pv
import numpy as np
import torch

def connectivity_to_pyvista_connectivity(connectivity):
        """
        Convert the Mesh connectivity to the pyvista format

        Args : 
            connectivity (Number of element x Number of points per element sized np.array) : The connectivity of the element in the mesh.
        
        Returns :
            pyvista_connectivity (Number of element *(1 + Number of points per element) sized np.array) : The connectivity in the pyvista format.
        """
         
        pyvista_connectivity = []
        connectivity = connectivity
        for element in connectivity:
            pyvista_connectivity += [len(element)] + list(element)
        
        return np.array(pyvista_connectivity)

def plot_GridNN_2D(grid, connectivity):
    connectivity = connectivity_to_pyvista_connectivity(connectivity.clone().detach().numpy())
    nodes = grid.all_nodal_coordinates.clone().detach().numpy()

    pv_mesh = pv.PolyData(nodes)
    pv_mesh.faces = connectivity

    free = grid.free.clone().detach().int()
    free[grid.free] = torch.tensor([grid.nodal_coordinates['free'].requires_grad for _ in grid.nodal_coordinates['free']]).int()
    free[~grid.free] = torch.tensor([grid.nodal_coordinates['imposed'].requires_grad for _ in grid.nodal_coordinates['imposed']]).int()


    pv_mesh['free'] = free.clone().detach().int().numpy()

    plotter = pv.Plotter()
    # plotter.add_mesh(mesh = pv_mesh,
    #                 scalars="free",
    #                 show_edges=True,
    #                 categories=True,
    #                 annotations={0: "Frozen", 1: "Free"},
    #                 #categorical=True,
    #                 #legend_labels=[("Frozen", "0"), ("Free", "1")]
    #                 interpolate_before_map=False
    #                 
    #                 )


    plotter.add_mesh(
                    pv_mesh,
                    scalars="free",
                    interpolate_before_map=False,
                    show_edges=True,
                    categories=True,
                    annotations={0: "Frozen", 1: "Free"},
                    cmap=["blue", "red"]
                    #scalar_bar_args={"title": "Status", "n_labels": 2, "color_mode": "discrete"},
                    )
    
    plotter.show()