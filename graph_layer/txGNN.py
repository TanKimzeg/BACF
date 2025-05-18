import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv,MessagePassing
import torch_geometric.nn.functional as F
from .utils import build_tx_graph

#    自定义GNN层
class EdgeSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels):
        super(EdgeSAGEConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        
        self.update_activation = nn.ReLU()
        self.aggr_lin = nn.Linear(in_channels + out_channels, out_channels)
        self.message_lin = nn.Linear(in_channels + edge_channels, out_channels)
        self.message_activation = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        #   x: node features
        #   edge_index: graph connectivity
        #   edge_attr: edge features
        # x = self.lin(x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j, edge_attr):
        m_j = torch.cat((x_j, edge_attr), dim=-1)
        m_j = self.message_activation(self.message_lin(m_j))
        return m_j

    def update(self, aggr_out, x):
        aggr_out = self.update_activation(self.aggr_lin(
            torch.cat((aggr_out,x), dim=-1)
        ))
        return aggr_out
    
    def __repr__(self):
        return '{}({}, {}, edge_channels={})'.format(self.__class__.__name__, self.in_channels, self.out_channels,
                                                     self.edge_channels)

#    定义图神经网络
class TxGNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, edge_dim):
        super(TxGNN, self).__init__()
        self.conv1 = EdgeSAGEConv(input_dim, hidden_dim, edge_dim)
        self.conv2 = EdgeSAGEConv(hidden_dim, output_dim, edge_dim)

    def forward(self, data:Data) -> torch.Tensor:
        #   data: PyG Data object
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        # print(x.shape) # [batch*8, 4]
        return x
    
