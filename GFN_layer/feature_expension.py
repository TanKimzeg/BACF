import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import degree
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.data import Data


class FeatureExpansion(MessagePassing):
    def __init__(self, ak:int) -> None:
        super(FeatureExpansion, self).__init__('add')
        self.ak = ak
        self.edge_norm_diag = 1e-8  # edge norm is used, and set A diag to it

    @staticmethod
    def compute_degree(edge_index:torch.Tensor, num_nodes:int) -> torch.Tensor:
        deg = degree(edge_index[0], num_nodes)
        deg = deg.view(-1, 1)
        return deg
    
    @staticmethod
    def norm(edge_index:torch.Tensor, num_nodes:int, edge_weight:torch.Tensor, diag_val=1e-8, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        # Add edge_weight for loop edges.
        loop_weight = torch.full((num_nodes, ),
                                 diag_val,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def compute_aks(self, num_nodes:int, x:torch.Tensor, 
                    edge_index:torch.Tensor, edge_weight:torch.Tensor=None) -> torch.Tensor:
        edge_index, norm = self.norm(edge_index, num_nodes, edge_weight, 
                                     diag_val=self.edge_norm_diag)
        xs = list()
        for _ in range(self.ak):
            x = self.propagate(edge_index, x=x, norm=norm)
            xs.append(x)

        return torch.cat(xs, dim=-1)

    def compute_centrality(data:Data) -> torch.Tensor:
        raise NotImplementedError("No need to use centrality in this layer.")

    def message(self, x_j, norm:torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j
    
    def transform(self, data:Data) -> torch.Tensor:
        if data.x is None:
            data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)

        deg = self.compute_degree(data.edge_index, data.num_nodes)
        aks = self.compute_aks(data.num_nodes, data.x, data.edge_index)
        # cent = self.compute_centrality(data)
        # data.x = torch.cat([data.x, deg, aks, cent], dim=-1)
        data.x = torch.cat([deg, data.x, aks], dim=-1)

        return data.x.clone()
