import os
import torch
from .utils import get_tx_graphs
import argparse

def main(args: argparse.Namespace, label: str):
    addr_list = []
    output_dir = os.path.join(args.output, label)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # read the addrs from the ready_addr.txt file
    log_dir = os.path.join(args.logdir, label)
    read_addr_path = os.path.join(log_dir, "ready_addr.txt")
    with open(read_addr_path, "r", encoding="utf-8") as f:
        for line in f:
            addr = line.strip()
            addr_list.append(addr)

    ##################################################
    # addr_list = addr_list[:10]  # 只取前10个地址进行测试
    ##################################################

    print(f"Loaded {len(addr_list)} addresses from {read_addr_path}.")

    # get the tx graphs for each addr
    tx_graphs:dict[str, list[torch.Tensor]] = get_tx_graphs(addr_list)
    
    # dim_0: node_num ( expected 8 )
    # dim_1: feature_dim ( expected 37 )
    first_graph = next(iter(tx_graphs.values()))[0]
    dim_0, dim_1 = first_graph.size(0), first_graph.size(1)
    print(f"Node feature dimension: {dim_1}, Node count per graph: {dim_0}")
    
    # get file handler
    file_handler = dict()
    for i in range(dim_0):
        for j in range(dim_1):
            file_path = os.path.join(output_dir, f"{i}{j}.txt")
            file_handler[(i,j)] = open(file_path, "w", encoding="utf-8")
    
    for addr, graph_list in tx_graphs.items():
        print(f"Processing address: {addr}, number of graphs: {len(graph_list)}") 
        stacked_graph = torch.stack(graph_list, dim=0)  # [batch_size, node_num, feature_dim]
        assert stacked_graph.size(1) == dim_0
        assert stacked_graph.size(2) == dim_1
        for i in range(dim_0):
            for j in range(dim_1):
                feature_value = stacked_graph[:, i, j].tolist()
                file_handler[(i,j)].write(" ".join(map(str, feature_value)) + "\n")

    for handler in file_handler.values():
        handler.close()

    print(f"Feature values saved to {output_dir} directory.")    