import os
from .utils import get_data_loader,get_tx_graphs
from . import txGNN
import torch
from torch_geometric.data import Data
from .trainer import Trainer

torch.random.manual_seed(8420)
def train2impl(args,label:str):
    addr_list = []

    # read the addrs from the ready_addr.txt file
    read_addr_path = f"{os.path.join(args.logdir,label)}/ready_addr.txt"
    with open(read_addr_path, "r", encoding="utf-8") as f:
        for line in f:
            addr = line.strip()
            addr_list.append(addr)
    # addr_list = addr_list[:10]  # 只取前10个地址进行测试
    print(f"Loaded {len(addr_list)} addresses from {read_addr_path}.")

    # get the tx graphs for each addr
    tx_graphs:dict[list[Data]] = get_tx_graphs(addr_list)
    train_loader = get_data_loader(tx_graphs,method="random")
    eval_loader = get_data_loader(tx_graphs,method="custom")
    input_dim, edge_dim = list(tx_graphs.values())[0][0].x.size(1), list(tx_graphs.values())[0][0].edge_attr.size(1)
    assert input_dim == 4, f"input_dim is {input_dim}"
    assert edge_dim == 10, f"edge_dim is {edge_dim}"

    model = txGNN.TxGNN(input_dim, output_dim=4, 
                        hidden_dim=16, edge_dim=edge_dim)
    trainer = Trainer(arg=args,label=label,model=model, train_dataloader=train_loader, 
                      eval_dataloader=eval_loader)
    # train the model
    trainer.train_model_with_loader()
    trainer.save(save_path=args.modelsave)
    trainer.impl()

def load2impl(args,label:str,model_path:str):
    addr_list = []

    # read the addrs from the ready_addr.txt file
    read_addr_path = f"{os.path.join(args.logdir,label)}/ready_addr.txt"
    with open(read_addr_path, "r", encoding="utf-8") as f:
        for line in f:
            addr = line.strip()
            addr_list.append(addr)
    # addr_list = addr_list[:10]  # 只取前10个地址进行测试
    print(f"Loaded {len(addr_list)} addresses from {read_addr_path}.")

    # get the tx graphs for each addr
    tx_graphs:dict[list[Data]] = get_tx_graphs(addr_list)
    train_loader = get_data_loader(tx_graphs,method="random")
    eval_loader = get_data_loader(tx_graphs,method="custom")
    input_dim, edge_dim = list(tx_graphs.values())[0][0].x.size(1), list(tx_graphs.values())[0][0].edge_attr.size(1)
    assert input_dim == 4, f"input_dim is {input_dim}"
    assert edge_dim == 10, f"edge_dim is {edge_dim}"

    model = txGNN.TxGNN(input_dim=4, output_dim=4, 
                        hidden_dim=16, edge_dim=10)
    trainer = Trainer(arg=args,label=label, model=model, train_dataloader=train_loader, 
                      eval_dataloader=eval_loader)
    # load the model from path
    trainer.load(model_path=model_path)
    trainer.impl()
    