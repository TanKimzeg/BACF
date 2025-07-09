# check if all addresses have been prepared
# and if not, log the missing addresses
import os
import sys
import json
import traceback

def check_addresses(arg,label:str):
    labeled_addr_path = f"{arg.addrdir}/{label}/k=1/"
    neighbor_addr_path = f"{arg.addrdir}/{label}/k=2/"
    log_dir = os.path.join(arg.logdir,label)
    ready_addr_path = os.path.join(log_dir, "ready_addr.txt")
    if os.path.exists(ready_addr_path):
        print(f"Ready addresses already exist at {ready_addr_path}.")
        return

    # 
    neighbor_addr_list = set(os.path.splitext(file)[0] for file in os.listdir(neighbor_addr_path))

    # read the labeled address file to get the list of addresses
    ready_addr = set() # set of labeled addresses that are ready
    missing_neighbor = set() # set of addresses that are missing neighbors, need to pycurl them
    for json_file in os.listdir(labeled_addr_path):
        if json_file.endswith(".json"):
            ready = True
            with open(os.path.join(labeled_addr_path, json_file), "r", encoding="utf-8") as f:
                json_data = json.load(f)
                txs_data = json_data["data"][0]["txs"]
                for tx in txs_data:
                    try:
                        for _input in tx["inputs"]:
                            if _input["address"] not in neighbor_addr_list:
                                missing_neighbor.add(_input["address"])
                                ready = False
                        for _output in tx["outputs"]:
                            if _output["address"] not in neighbor_addr_list:
                                missing_neighbor.add(_output["address"])
                                ready = False
                    except KeyError as e:
                        print(f"KeyError: {e} in file {json_file}")
                        traceback.print_exc()
            if ready:
                ready_addr.add(json_file.split(".")[0])
                print(f"Address {json_file.split(".")[0]} is ready.")

    # log the missing addresses
    # create the log directory if it does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    # 记录缺失地址
    with open(os.path.join(log_dir, "missing_neighbor.txt"), "w", encoding="utf-8") as f:
        for address in missing_neighbor:
            f.write(address + "\n")

    # 记录准备好的地址
    with open(os.path.join(log_dir, "ready_addr.txt"), "w", encoding="utf-8") as f:
        for address in ready_addr:
            f.write(address + "\n")