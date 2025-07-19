# check if all addresses have been prepared
# and if not, log the missing addresses
import os
import json
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class Args:
    addrdir = "F:/json_data"
    logdir = "F:/log"


def check_addresses(arg: Args,label:str):
    labeled_addr_path = f"{arg.addrdir}/{label}/k=1/"
    neighbor_addr_path = f"{arg.addrdir}/{label}/k=2/"
    if not os.path.exists(labeled_addr_path):
        print(f"Labeled address path does not exist: {labeled_addr_path}")
        return
    log_dir = os.path.join(arg.logdir,label)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    neighbor_addr_list = set(os.path.splitext(file)[0] for file in os.listdir(neighbor_addr_path))

    # read the labeled address file to get the list of addresses
    ready_addr = set() # set of labeled addresses that are ready
    missing_neighbor = list()
    for json_file in tqdm(os.listdir(labeled_addr_path), desc=f"Checking addresses for {label}", 
                          unit='addr', dynamic_ncols=True, leave=True):
        if json_file.endswith(".json"):
            ready = True
            sub_missing = set()
            with open(os.path.join(labeled_addr_path, json_file), "r", encoding="utf-8") as f:
                json_data = json.load(f)
                txs_data = json_data["data"][0]["txs"]
                for tx in txs_data:
                    for _input in tx["inputs"]:
                        if "address" in _input:
                            if _input["address"] not in neighbor_addr_list:
                                sub_missing.add(_input["address"])
                                ready = False
                    for _output in tx["outputs"]:
                        if "address" in _output:
                            if _output["address"] not in neighbor_addr_list:
                                sub_missing.add(_output["address"])
                                ready = False
            if ready:
                ready_addr.add(json_file.split(".")[0])
            else:
                missing_neighbor.append((len(sub_missing), list(sub_missing)))

    if missing_neighbor:
        missing_neighbor.sort(key=lambda x: x[0])
        print(f"Min missing neighbors: {missing_neighbor[0][0]}")
        print(f"Max missing neighbors: {missing_neighbor[-1][0]}")
    # log the missing addresses
    # create the log directory if it does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    # 记录缺失地址
    with open(os.path.join(log_dir, "missing_neighbor.txt"), "w", encoding="utf-8") as f:
        # while missing_neighbor:
            # _, addrs = heapq.heappop(missing_neighbor)
        for _, addrs in missing_neighbor:
            for address in addrs:
                if address != "coinbase":
                    f.write(address + "\n")
    print(f"Unready addr count: {len(missing_neighbor)}, saving at {log_dir}/missing_neighbor.txt")

    # 记录准备好的地址
    with open(os.path.join(log_dir, "ready_addr.txt"), "w", encoding="utf-8") as f:
        for address in ready_addr:
            f.write(address + "\n")
    print(f"Ready addr count: {len(ready_addr)}, saving at {log_dir}/ready_addr.txt")

if __name__ == "__main__":
    arg = Args()
    labels = [
        'Blackmail',    
        'Cyber-Security',    
        'DarknetMarket', 
        'Exchange',
        'P2PFIS',
        'P2PFS',
        'Gambling',
        'CriminalBlacklist',
        'MoneyLaundering',
        'PonziScheme',
        'MiningPool',
        'Tumbler',  
        'Individual'
    ]
    for label in labels:
        check_addresses(arg, label)
        print(f"Address check completed for label: {label}")