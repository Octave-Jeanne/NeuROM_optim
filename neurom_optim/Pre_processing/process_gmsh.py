import torch
from tqdm import tqdm


class GMSH_object:
    def __init__(self):
        self.nodes = []
        self.elements = {}              # keys = str(etype)
        self.PhysicalEntities = {}
        self.line_type = None

    def __call__(self, path_to_msh, int_precision, float_precision):
        with open(path_to_msh) as f:
            nlines = sum(1 for _ in f)
        bar = tqdm(total=nlines, desc="Processing GMSH file")

        with open(path_to_msh) as f:
            for line in f:
                self.process_line(line)
                bar.update(1)

        self.process_overall_info(int_precision, float_precision)

    def process_line(self, line):
        if self.update_line_type(line):
            return
        line = line.strip().split()
        match self.line_type:
            case "PhysicalNames":
                self.process_physical_name(line)
            case "Nodes":
                self.process_node(line)
            case "Elements":
                self.process_element(line)

    def update_line_type(self, line):
        if line.startswith("$"):
            self.line_type = line.strip("$\n")
            return True
        return False

    def process_physical_name(self, line):
        if len(line) > 1:
            dim, tag, name = line
            self.PhysicalEntities[tag] = {
                "dim": dim,
                "name": name.strip('"')
            }

    def process_node(self, line):
        if len(line) == 1:
            self.NNodes = int(line[0])
        else:
            self.nodes.append([float(x) for x in line[1:]])

    def process_element(self, line):
        if len(line) < 4:
            return  # skip element-count line

        tag, etype, phys_tag, nodes = self.read_element_line(line)
        etype = str(etype)

        if etype not in self.elements:
            etype_name, edim = self.process_element_type_tag(etype)
            self.elements[etype] = {
                "dim": edim,
                "type": etype_name,
                "tags": [],
                "connectivity": [],
            }

        self.elements[etype]["tags"].append(tag)
        self.elements[etype]["connectivity"].append(nodes)

        if phys_tag in self.PhysicalEntities:
            key = f"element_type_{etype}"
            pe = self.PhysicalEntities[phys_tag]
            if key not in pe:
                pe[key] = {"tags": [], "nodeIDs": []}
            pe[key]["tags"].append(tag)
            pe[key]["nodeIDs"] += nodes

    @staticmethod
    def read_element_line(line):
        tag = int(line[0])
        etype = int(line[1])
        ntags = int(line[2])
        phys_tag = line[3]                     # first tag = physical
        nodes = [int(i) - 1 for i in line[3 + ntags:]]
        return tag, etype, phys_tag, nodes

    @staticmethod
    def process_element_type_tag(etype):
        match int(etype):
            case 1:
                return "2-node bar", 1
            case 2:
                return "3-node triangle", 2
            case 3:
                return "4-node quadrangle", 2
            case 4:
                return "4-node tetrahedron", 3
            case 15:
                return "point", 0
        raise ValueError(f"Unknown element type {etype}")

    def process_overall_info(self, int_precision, float_precision):
        for etype, e in self.elements.items():
            conn = torch.tensor(e["connectivity"], dtype=int_precision)
            conn -= conn.min()
            e["connectivity"] = conn

        for ptag, pe in self.PhysicalEntities.items():
            for k in list(pe.keys()):
                if k in ["dim", "name"]:
                    continue
                etype = k.split("_")[-1]
                base = min(self.elements[etype]["tags"])
                pe[k]["elemIDs"] = torch.unique(
                    torch.tensor([t - base for t in pe[k]["tags"]],
                                 dtype=int_precision)
                )
                pe[k]["nodeIDs"] = torch.unique(
                    torch.tensor(pe[k]["nodeIDs"], dtype=int_precision)
                )

        self.nodes = torch.tensor(self.nodes, dtype=float_precision)


def read_gmsh(path_to_msh,
              IntPrecision=torch.int64,
              FloatPrecision=torch.float64):
    gmsh = GMSH_object()
    gmsh(path_to_msh, IntPrecision, FloatPrecision)
    return gmsh


def get_nodes_and_elements_IDs(gmsh_mesh,
                               entity_dim,
                               entity_tag=None,
                               entity_name=None):
    if entity_tag is None:
        entity_tag = get_tag(gmsh_mesh, entity_name)
    return get_nodes_and_elements_IDs_using_tag(gmsh_mesh, entity_dim, entity_tag)


def get_tag(gmsh_mesh, entity_name):
    for tag in gmsh_mesh.PhysicalEntities:
        if gmsh_mesh.PhysicalEntities[tag]["name"] == entity_name:
            return tag
    raise ValueError(f"No entity found for name {entity_name}")


def get_nodes_and_elements_IDs_using_tag(gmsh_mesh,
                                         entity_dim,
                                         entity_tag):
    entity_tag = str(entity_tag)
    for key in gmsh_mesh.PhysicalEntities[entity_tag]:
        if key not in ["dim", "name"]:
            etype = key.split("_")[-1]
            _, edim = gmsh_mesh.process_element_type_tag(etype)
            if edim == entity_dim:
                nodeIDs = gmsh_mesh.PhysicalEntities[entity_tag][key]["nodeIDs"]
                elemIDs = gmsh_mesh.PhysicalEntities[entity_tag][key]["elemIDs"]
                return nodeIDs, elemIDs

    raise ValueError("No matching elements found")
