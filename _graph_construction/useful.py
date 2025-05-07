import json
import pandas as pd
from collections import defaultdict
import math

def read_txs(path:str) -> dict:
    f = open(path, 'r')
    data = json.load(f)
    f.close()
    return data["data"][0]["txs"]

def appera_cnt(txs:dict) -> tuple:
    '''
    统计每个地址在所有交易中出现的次数
    '''
    in_appear_cnt = defaultdict(int)
    out_appear_cnt = defaultdict(int)
    for tx in txs:
        for input_ in tx["inputs"]:
            in_appear_cnt[input_["address"]] += 1
        for output in tx["outputs"]:
            out_appear_cnt[output["address"]] += 1
    return in_appear_cnt, out_appear_cnt