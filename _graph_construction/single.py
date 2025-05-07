from useful import read,appear_cnt
from feature import statistics
from collections import defaultdict

def single(txs:dict) -> tuple:
    '''
    embedding: 统计每个交易中single类的地址的输入输出值的统计特征
    edge: 交易和single类地址的边
    '''
    in_appear_cnt, out_appear_cnt = appear_cnt(txs)
    embedding = defaultdict(list)
    edge = []
    for tx in txs:
        in_aggregate_num, out_aggregate_num, in_values, out_values = single_aggregate(
            tx, in_appear_cnt, out_appear_cnt) 
        if len(in_values):
            embedding['single_in_'+tx['txid']] = statistics(in_values) + [in_aggregate_num]
            edge.append((tx['txid'], 'single_in_'+tx['txid'])) 
        if len(out_values):
            embedding['single_out_'+tx['txid']] = statistics(out_values) + [out_aggregate_num]
            edge.append((tx['txid'], 'single_out_'+tx['txid']))
    return embedding, edge

def single_aggregate(tx:dict,in_appear_cnt:dict,
                     out_appear_cnt:dict) -> tuple:
    '''
    对传入的单一交易中single类的地址进行聚合
    '''
    in_values = []
    out_values = []
    in_aggregate_num = 0
    out_aggregate_num = 0
    for input_ in tx["inputs"]:
        if in_appear_cnt[input_["address"]] == 1:
            in_aggregate_num += 1
            in_values.append(input_["value"])
            tx['inputs'].remove(input_)
    for output in tx["outputs"]:
        if out_appear_cnt[output["address"]] == 1:
            out_aggregate_num += 1
            out_values.append(output["value"])
            tx['outputs'].remove(output)
    return in_aggregate_num, out_aggregate_num, in_values, out_values

