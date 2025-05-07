from collections import defaultdict
from torch_geometric.data import Data

def context_feature(tx: dict) -> list:
    '''
    return a list of context embedding
    the context embedding contains 
    input addr count,
    output addr count, 
    timestamp, 
    input total value,
    output total value,
    input avg value,
    output avg value,
    input max value,
    output max value,
    input min value,
    output min value,
    '''
    input_addr_count = len(tx["inputs"])
    output_addr_count = len(tx["outputs"])
    timestamp = tx["time"]
    total_value = sum([float(input["value"])
                for input in tx["inputs"]])
    input_avg_value = total_value / input_addr_count if input_addr_count > 0 else 0
    output_avg_value = total_value / output_addr_count if output_addr_count > 0 else 0
    input_max_value = max([float(input["value"]) 
                for input in tx["inputs"]], default=0)
    output_max_value = max([float(output["value"])
                for output in tx["outputs"]], default=0)
    input_min_value = min([float(input["value"])
                for input in tx["inputs"]], default=0)
    output_min_value = min([float(output["value"])
                for output in tx["outputs"]], default=0)
    return [
        input_addr_count, output_addr_count, 
        timestamp, total_value,
        input_avg_value, output_avg_value, 
        input_max_value, output_max_value, 
        input_min_value, output_min_value
    ]

def addr_feature(txdata_of_addr:dict) -> list:
    '''
    return a list of addr embedding
    the addr embedding contains 
    total tx count, 历史交易笔数
    avg cost in tx, 历史交易中平均额度
    active period,  活跃周期(时间差/交易次数)
    '''
    total_tx_count = 0
    amount = 0
    active_period = 0
    total_tx_count += txdata_of_addr["txCount"]
    amount += (float(txdata_of_addr["receive"]) - 
               float(txdata_of_addr["spend"]))/txdata_of_addr["txCount"]
    active_period += (txdata_of_addr["txs"][-1]["time"] - 
                        txdata_of_addr["txs"][0]["time"])/txdata_of_addr["txCount"]/3600 # 转换为小时

    avg_txCount = total_tx_count / len(txdata_of_addr) if len(txdata_of_addr) > 0 else 0
    avg_amount = amount / len(txdata_of_addr) if len(txdata_of_addr) > 0 else 0
    avg_active_period = active_period / len(txdata_of_addr) if len(txdata_of_addr) > 0 else 0
    return [avg_txCount, avg_amount, avg_active_period]

def addrFeature_aggregate(addrFeature:dict[list],addr_cnt:dict) -> list:
    '''
    addrFeature: a dict of addr and its feature list
    addr_cnt: a dict of addr and its count
    return a list of aggregated addr feature
    aggregate the addr feature by its count,
    [
        range from 1 to 3
        range from 4 to 7
        range from 8 to 15
        range from 16 to infinity
    ]
    '''
    appear_count13, appear_count47, appear_count815, appear_count16 = [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]
    for addr, feature in addrFeature.items():
        if addr_cnt[addr] < 4:
            appear_count13 = [appear_count13[i] + feature[i]
                              for i in range(len(appear_count13))]
        elif addr_cnt[addr] < 8:
            appear_count47 = [appear_count47[i] + feature[i]
                              for i in range(len(appear_count47))]
        elif addr_cnt[addr] < 16:
            appear_count815 = [appear_count815[i] + feature[i]
                              for i in range(len(appear_count815))]
        else:
            appear_count16 = [appear_count16[i] + feature[i]
                              for i in range(len(appear_count16))]
    appear_count13 = list(map(lambda x: x/len(appear_count13), 
                              appear_count13)) if len(appear_count13) > 0 else [0,0,0,0]
    appear_count47 = list(map(lambda x: x/len(appear_count47), 
                              appear_count47)) if len(appear_count47) > 0 else [0,0,0,0]
    appear_count815 = list(map(lambda x: x/len(appear_count815), 
                               appear_count815)) if len(appear_count815) > 0 else [0,0,0,0]
    appear_count16 = list(map(lambda x: x/len(appear_count16), 
                               appear_count16)) if len(appear_count16) > 0 else [0,0,0,0]
    return [
        appear_count13,
        appear_count47,
        appear_count815,
        appear_count16 
    ]
