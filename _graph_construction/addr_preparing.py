import os
import json

#    tree of tx_path:
#    tx_path
#    ├── 0x
#    │   ├── 0x
#    │   │   ├── 0x.json
#    │   │   ├── 0x.json
#    │   │   └── 0x.json
#    │   ├── 0x.json
#    │   └── 0x.json
#    ├── 0x.json
#    └── 0x.json

def integrity_check(addr:str,tx_path:str) -> bool:
    json_path = tx_path + f"{addr}.json"
    if not os.path.exists(json_path):
        return False
    with open(json_path, 'r') as f:
        js = json.load(f)
        if js['data'][0]['txCount'] < 50:#  tx count in graph
            return False

    return True
    
def addrs_checking(addrs:list[str],tx_path:str) -> list[str]:
    '''
    Check the addr list and remove invalid addresses.
    '''
    print('Checking addresses...')
    valid_addrs = []
    for i,addr in enumerate(addrs):
        if not integrity_check(addr,tx_path):
            print('Unsatisfied address:',addr)
            addrs.pop(i)
    print('Checking finished.')
    return addrs

def addr_preparing(tx_path:str) -> list[str]:
    filename = os.listdir(tx_path)
    addrs = [addr[:-5] for addr in filename if addr.endswith('.json')]
    addrs = addrs_checking(addrs,tx_path)
    print('Addresses prepared.')
    return addrs
