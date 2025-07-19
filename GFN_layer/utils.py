from . import config
from .embedding import addr_feature,context_feature, addrFeature_aggregate
from .feature_expension import FeatureExpansion
from collections import defaultdict
import json
import os
from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Sampler
from torch_geometric.data import Data,Dataset
from torch_geometric.loader import DataLoader

class Dict(Dataset):
    def __init__(self,data_source:dict[list[Data]]):
        '''
        data_source: a dict of list of Data'''
        self.data_source = data_source

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return sum(len(v) for v in self.data_source.values())
    

class DictBatchSampler(Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source
        self.keys = list(data_source.keys())

    def __iter__(self):
        for key in self.keys:
            yield self.data_source[key]
    
    def __len__(self):
        return len(self.keys)    
        
        
def get_data_loader(tx_graphs:dict[list[Data]], method:str) -> tuple:
    '''
    Get the data loader for the tx graph
    '''

    match method:
        case "random":
            # Create a DataLoader for the graphs
            datas = [data for datas in tx_graphs.values() for data in datas]
            tx_Sampler = RandomSampler(datas)
            loader = DataLoader(datas, sampler=tx_Sampler, batch_size=16)
            # loader = DataLoader(tx_graphs, batch_size=32, shuffle=True)

        case "sequential":
            tx_Sampler = SequentialSampler(datas)
            loader = DataLoader(datas, sampler=tx_Sampler, batch_size=64)

        case "custom":
            tx_Sampler = DictBatchSampler(tx_graphs)
            loader = DataLoader(Dict(tx_graphs), batch_sampler=tx_Sampler, collate_fn=lambda x: x)

    print(f"Created {method} DataLoader with {
        sum(len(v) for v in tx_graphs.values())} graphs.")
    return loader


def get_tx_data(addr:str) -> list[dict]:
    '''
    return the list of all txs of the addr
    '''
    json_path_1 = os.path.join(config.args.addrdir,config.label)+f"/k=1/"
    json_path_2 = os.path.join(config.args.addrdir,config.label)+f"/k=2/"
    try:
        with open(f"{json_path_2}{addr}.json", "r") as f:
            js = json.load(f)
        return js["data"][0]
    except FileNotFoundError:
        with open(f"{json_path_1}{addr}.json", "r") as f:
            js = json.load(f)
        return js["data"][0]


def addr_count(txs:list[dict]) -> list[dict]:
    '''
    tag: "in" or "out"
    input a txs list and a tag of in or out,
    return a dict of the count of each address
    '''
    addr_cnt = [defaultdict(int) ,defaultdict(int)]
    for tx in txs:
        for _input in tx["inputs"]:
            if "address" in _input:
                addr_cnt[0][_input["address"]] += 1
    for tx in txs:
        for output in tx["outputs"]:
            if "address" in output:
                addr_cnt[1][output["address"]] += 1
    return addr_cnt

def get_tx_graphs(addrs: list) -> dict[str, list[torch.Tensor]]:
    # Assume this is a list of addrs
    # addrs = ["addr1", "addr2", "addr3"]

    # Get tx data for each addr
    txdata_of_addrs = defaultdict(list)
    for addr in addrs:
        txs:list[dict] = get_tx_data(addr)["txs"]
        txdata_of_addrs[addr] = txs
    
    # Get addr count for each addr
    addr_cnt_of_addrs = defaultdict(list)
    for addr in addrs:
        addr_cnt_of_addrs[addr] = addr_count(txdata_of_addrs[addr])
    
    # Build tx graph for each tx
    tx_graphs:dict[list[torch.Tensor]] = defaultdict(list)
    for addr in tqdm(addrs, desc="Generating GFN features", 
                     unit="addr", dynamic_ncols=True):
        addr_graph:list[torch.Tensor] = list()
        for tx in txdata_of_addrs[addr]:
            graph = build_tx_graph(tx, addr_cnt_of_addrs[addr])
            addr_graph.append(graph)
        tx_graphs[addr] = addr_graph.copy()
    
    return tx_graphs

def build_tx_graph(k1tx:dict,addr_cnt:list[dict]) -> torch.Tensor:
    '''
    build a graph from a tx dict
    return a graph
    '''
    in_addrFeature = defaultdict(list)
    for _input in k1tx["inputs"]:
        if "address" in _input:
            _addr = _input["address"]
            txdata = get_tx_data(_addr)
            in_addrFeature[_addr] = addr_feature(txdata) + [float(_input["value"])]

    out_addrFeature = defaultdict(list)
    for output in k1tx["outputs"]:
        if "address" in output:
            _addr = output["address"]
            txdata = get_tx_data(_addr)
            out_addrFeature[_addr] = addr_feature(txdata) + [float(output["value"])]

    aggregated_in_addrFeature = addrFeature_aggregate(in_addrFeature, addr_cnt[0])
    aggregated_out_addrFeature = addrFeature_aggregate(out_addrFeature, addr_cnt[1])
    # 转换为tensor
    aggregated_in_addrFeature = torch.tensor(aggregated_in_addrFeature, 
                                             dtype=torch.float).view(len(aggregated_in_addrFeature),-1)
    aggregated_out_addrFeature = torch.tensor(aggregated_out_addrFeature, 
                                              dtype=torch.float).view(len(aggregated_out_addrFeature), -1)
    # contextFeature = context_feature(k1tx)
    x = torch.cat([aggregated_in_addrFeature, aggregated_out_addrFeature], dim=0)
    x = torch.nan_to_num(x, nan=0.0)  # 将 NaN 替换为 0

    # 生成边
    edge = [list([0]*4 + [1]*4 + [2]*4 + [3]*4), 
            list([4, 5, 6, 7]*4)]
    edge = torch.LongTensor(edge)  # .t().contiguous()


    assert x.size(0) == 8, f"输入节点\输出节点共{x.size(0)}"
    assert x.size(1) == 37, f"每个节点有{x.size(1)}个属性"
    assert edge.size(0) == 2, f"edge.size(0) = {edge.size(0)}"
    # assert edge.size(1) == 16, "edge 的列数不是16!"
    # assert contextFeature.size(0) == edge.size(1), "每条边都要一个属性!"
    graph = Data(
        x=x,
        edge_index=edge,
    )
    graph = FeatureExpansion(ak=3).transform(graph)
    return graph