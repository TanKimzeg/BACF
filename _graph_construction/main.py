import pickle
import datetime
from useful import read_txs
from single import single
from multi import multi 
from collections import defaultdict
from tqdm import tqdm

def building(txs:dict,tx_path:str) -> tuple:

    tx_embedding = defaultdict(list)
    single_embedding, single_edge = single(txs)
    multi_embedding, multi_edge = multi(txs,psi=0.7)
    edge = single_edge + multi_edge
    embedding = dict()
    embedding.update(tx_embedding)
    embedding.update(single_embedding)
    embedding.update(multi_embedding)
    return embedding, edge


def main():
    f = open('txs.pickle', 'rb')
    pickle.dump(,f)
    f.close()

if __name__ == "__main__":
    main()
