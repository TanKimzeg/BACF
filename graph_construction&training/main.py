from collections import defaultdict
from utils import get_data_loader,get_tx_graphs
import txGNN
import torch
from torch_geometric.data import Data
from trainer import Trainer

if __name__ == "__main__":
    torch.random.manual_seed(8420)
    addr_list = []

    # read the addrs from the ready_addr.txt file
    read_addr_path = f"F:/logs/ready_addr.txt"
    with open(read_addr_path, "r", encoding="utf-8") as f:
        for line in f:
            addr = line.strip()
            addr_list.append(addr)
    # addr_list = addr_list[:10]  # 只取前10个地址进行训练
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
    trainer = Trainer(model=model, train_dataloader=train_loader, 
                      eval_dataloader=eval_loader)
    # train the model
    trainer.train_model_with_loader()
    trainer.save(save_path="F:/model_save")

    # load the model from path
    # model_name = input("请输入模型名称:")
    # model_path = f"F:/model_save/{model_name}"
    # while not os.path.exists(model_path):
    #     print(f"模型文件不存在,请重新输入: {model_name}")
    #     model_name = input("请输入模型名称:")
    #     model_path = f"F:/model_save/{model_name}"
    # trainer.load(save_path=model_path)
    trainer.eval_model()
    