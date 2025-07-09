import numpy as np
import statsmodels.robust as rb
from scipy import stats

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
    total_tx_count = txdata_of_addr["txCount"]
    amount = (float(txdata_of_addr["receive"]) - 
               float(txdata_of_addr["spend"]))/txdata_of_addr["txCount"]
    active_period = (txdata_of_addr["txs"][0]["time"] - 
                        txdata_of_addr["txs"][-1]["time"])/min(50,txdata_of_addr["txCount"])/3600 # 转换为小时

    return [total_tx_count, amount, active_period]

def addrFeature_aggregate(addrFeature:dict[str, list],addr_cnt:dict) -> list:
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
    appear_count13, appear_count47, appear_count815, appear_count16 = list(), list(), list(), list()
    # feature_dim = len(list(addrFeature.values())[0])
    for addr, feature in addrFeature.items():
        if addr_cnt[addr] < 4:
            appear_count13.append(feature)
        elif addr_cnt[addr] < 8:
            appear_count47.append(feature)
        elif addr_cnt[addr] < 16:
            appear_count815.append(feature)
        else:
            appear_count16.append(feature)
    appear_count13 = statistics(appear_count13) if appear_count13 else [0]*37
    appear_count47 = statistics(appear_count47) if appear_count47 else [0]*37
    appear_count815 = statistics(appear_count815) if appear_count815 else [0]*37
    appear_count16 = statistics(appear_count16) if appear_count16 else [0]*37
    return [
        appear_count13,
        appear_count47,
        appear_count815,
        appear_count16 
    ]

def statistics(features: list[list]) -> list[list]:
    '''
    return a list of statistics of features
    the statistics contains 
    mean, std, min, max, median
    '''
    features = np.array(features)
    features = np.nan_to_num(features, nan=0.0)  
    _max = np.max(features, axis=0)
    _min = np.min(features, axis=0)
    _range = _max - _min
    mean = np.mean(features, axis=0).tolist()
    std = np.std(features, axis=0).tolist()
    min_val = np.min(features, axis=0).tolist()
    max_val = np.max(features, axis=0).tolist()
    median = np.median(features, axis=0).tolist()
    MAD = rb.scale.mad(features, axis=0).tolist()
    skewness = stats.skew(features, axis=0).tolist()
    kurtosis = stats.kurtosis(features, axis=0).tolist()
    return [len(features),  *_range.tolist(), 
            *mean, *std, *min_val, *max_val, *median, 
            *MAD, *skewness, *kurtosis] # 1+4*9 = 37
