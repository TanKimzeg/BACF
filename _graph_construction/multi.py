import torch
import torch.nn as nn
from feature import statistics
import numpy as np
import sys
from collections import defaultdict,Counter
from useful import read 

def connection(txs:dict) -> tuple:
    in_address_tx = defaultdict(list)
    in_values = defaultdict(list)
    out_address_tx = defaultdict(list)
    out_values = defaultdict(list)
    for tx in txs:
        for _input in tx["inputs"]:
            addr, value = _input['address'], _input['value']
            in_address_tx[addr].append(tx['txid'])
            in_values[addr].append(value)
        for output in tx["outputs"]:
            addr, value = output['address'], output['value']
            out_address_tx[addr].append(tx['txid'])
            out_values[addr].append(value)
    for key, val in in_address_tx.items():
        in_address_tx[key] = tuple(val)
    for key, val in out_address_tx.items():
        out_address_tx[key] = tuple(val)
    return in_address_tx, in_values, out_address_tx, out_values


def adj_matrix(txs:dict,address_tx:dict,addrs:list[str]) -> tuple:
    '''
    生成邻接矩阵
    作用是生成两个稀疏邻接矩阵:
    ad_tx 和 tx_ad,
    分别表示地址到交易和交易到地址的关系矩阵。
    此外，它还返回每个地址的度（即关联交易的数量）。
    '''
    index_ad = []
    index_tx = []
    degree = []
    for idx, addr in enumerate(addrs):
        degree.append(len(address_tx[addr])) 
        # 该地址涉及的交易数，即该地址的度数
        index_ad.extend([[idx]*len(address_tx[addr]) for _ in range(len(address_tx[addr]))])
        index_tx.extend([txs.index(tx) for 
                         tx in address_tx[addr]])
    
    index = torch.LongTensor([index_ad,index_tx])
    index_T = torch.LongTensor([index_tx,index_ad])
    ones = torch.ones(len(index_ad)).float()
    ad_tx = torch.sparse.FloatTensor(index, ones, torch.Size([len(addrs), len(txs)])).to_dense()
    tx_ad = torch.sparse.FloatTensor(index_T, ones, torch.Size([len(txs), len(addrs)])).to_dense()
    return ad_tx,tx_ad,np.array(degree)
            
def similarity_matrix(ad_tx:torch.Tensor,
                      tx_ad:torch.Tensor,
                      degree:np.array,
                      psi:float) -> torch.Tensor:
    '''
    计算地址和交易的相似度矩阵
    '''
    Similarity = ad_tx.mm(tx_ad).float() # 论文中 S = A*A^T
    inv_degree = torch.diag(torch.tensor(1./degree)).float() # D^-1
    normalized_Similarity = Similarity.mm(inv_degree)
    relu = nn.ReLU() # ReLU是一种激活函数
    return relu(normalized_Similarity - 
                torch.ones([len(degree),len(degree)]*psi))

def aggregate_parament(num:int, psi:float) -> tuple:
    '''
    计算聚合参数,需要实验调优
    '''
    if num > 1200:
        psi = 0.9
    else:
        psi = 0.7
    beta = 0
    if num>800 and num<=1200:
        beta = num/40
    if num>1200 and num <=2500:
        beta = num/12
    if num>2500 and num<=4500:
        beta = num/3
    if num>4500 and num<=7500:
        beta = num/2.4
    if num>7500 and num<10000:
        beta = num/2
    if num>=10000:
        beta = num/1.8
    return psi,beta
    
def transverse(address_tx:dict) -> tuple:
    txset = list(address_tx.values())
    once_txs = [item for item,count in Counter(txset)
                .items() if count == 1]
    more_txs = set(txset) - set(once_txs)
    more_txs_addrs = defaultdict(list)
    once_txs_addrs = defaultdict(list)
    for addr, tx in address_tx.items():
        if tx in more_txs:
            more_txs_addrs[tx].append(addr)
        else:
            once_txs_addrs[tx].append(addr)
    return more_txs_addrs,once_txs_addrs

def deduplication(in_more_txs_addrs:dict,in_once_txs_addrs:dict,
                  out_more_txs_addrs:dict,out_once_txs_addrs:dict,
                  raw_values:list) -> tuple:
    '''
    功能:对多地址的交易进行去重,同时为这些地址生成新的标识
    单交易地址:直接保留原始值
    ''' 
    in_values = dict()
    in_address_tx = dict()
    out_values = dict()
    out_address_tx = dict()
    i = 0
    for tx,addrs in in_more_txs_addrs.items():
        i += 1
        in_address_tx['Deduplication_in_'+str(i)] = set(tx)
        in_values['Deduplication_in_'+str(i)] = []
        for addr in addrs:
            in_values['Deduplication_in_'+str(i)].append(raw_values[addr])
    for tx,addr in in_once_txs_addrs.items():
        in_address_tx[addr[0]] = set(tx)
        in_values[addr[0]] = raw_values[addr[0]]
    for tx,addrs in out_more_txs_addrs.items():
        i += 1
        out_address_tx['Deduplication_out_'+str(i)] = set(tx)
        out_values['Deduplication_out_'+str(i)] = []
        for addr in addrs:
            out_values['Deduplication_out_'+str(i)].append(raw_values[addr])
    for tx,addr in out_once_txs_addrs.items():
        out_address_tx[addr[0]] = set(tx)
        out_values[addr[0]] = raw_values[addr[0]]
    return in_address_tx,in_values,out_address_tx,out_values


def multi_aggregate_1(in_address_tx:dict,in_values:list,
                      out_address_tx:dict,out_values:list) -> tuple:
    '''
    功能:对地址和交易进行聚合
    虽然地址出现不止一次,但是仍然可能只与一个交易相关联
    这种情况被称为"once"类地址
    '''
    in_more_txs_addrs,in_once_txs_addrs = transverse(in_address_tx)
    out_once_txs_addrs,out_more_txs_addrs = transverse(out_address_tx)
    in_address_tx,in_values,out_address_tx,out_values = deduplication(in_more_txs_addrs,
                                        in_once_txs_addrs,out_more_txs_addrs,out_once_txs_addrs,in_values)
    return in_address_tx,in_values,out_address_tx,out_values

def multi_aggregate_2(similarity:torch.Tensor, beta:float) -> tuple:
    ''''
    功能:通过相似性矩阵对地址进行聚合,筛选出符合聚合条件的地址索引
    如果某个地址的相似性值大于beta,则认为该地址是一个聚合地址
    '''
    num = len(similarity)
    aggregate_index = []
    aggregated_num = dict()
    for i in range(num):
        nonzero = torch.nonzero(similarity[i]) # remain address
        if nonzero > beta:
            aggregated_num[i] = nonzero
            aggregate_index.append(i)

    return aggregate_index, aggregated_num

def multi(txs:dict,psi:float) -> tuple:
    
    edge = []
    embedding = defaultdict(list)
    in_address_tx, in_values, out_address_tx, out_values = connection(txs)
    in_address_tx, in_values, out_address_tx, out_values = multi_aggregate_1(in_address_tx,in_values)
    in_addrs = list(in_address_tx.keys())
    out_addrs = list(out_address_tx.keys())
    in_ad_tx, in_tx_ad, in_degree = adj_matrix(txs,in_address_tx,in_addrs)
    out_ad_tx, out_tx_ad, out_degree = adj_matrix(txs,out_address_tx,out_addrs)
    in_num,out_num = len(in_ad_tx), len(out_ad_tx)
    in_psi, in_beta = aggregate_parament(in_num,psi)
    out_psi, out_beta = aggregate_parament(out_num,psi)
    in_similarity = similarity_matrix(in_ad_tx,in_tx_ad,in_degree,in_psi)
    out_similarity = similarity_matrix(out_ad_tx,out_tx_ad,out_degree,out_psi)
    in_aggregate_index, in_aggregate_num = multi_aggregate_2(in_similarity, in_beta)
    out_aggregate_index, out_aggregate_num = multi_aggregate_2(out_similarity, out_beta)
    for addr in in_aggregate_index:
        _, index = in_similarity[addr].sort(descending=True)
        addr_id = index.tolist()[:len(torch.nonzero(in_similarity[addr]))]
        aggregeted_values = []
        for i in addr_id:
            aggregeted_values.append(in_values[in_addrs[i]])
        embedding['multi_in_'+str(addr)] = statistics(aggregeted_values) + [in_aggregate_num[addr]]
        for tx in in_address_tx[in_addrs[addr]]:
            edge.append((tx, 'multi_in_'+str(addr)))
    for addr in out_aggregate_index:
        _, index = out_similarity[addr].sort(descending=True)
        addr_id = index.tolist()[:len(torch.nonzero(out_similarity[addr]))]
        aggregeted_values = []
        for i in addr_id:
            aggregeted_values.append(out_values[out_addrs[i]])
        embedding['multi_out_'+str(addr)] = statistics(aggregeted_values) + [out_aggregate_num[addr]]
        for tx in out_address_tx[out_addrs[addr]]:
            edge.append((tx, 'multi_out_'+str(addr)))
    return embedding, edge
