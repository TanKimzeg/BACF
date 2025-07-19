import asyncio
import aiohttp
import aiofiles
import time
from tqdm import tqdm
import os
import json
from retrying import retry
from dataclasses import dataclass
from enum import Enum


@dataclass
class Consts:
    head = {
        "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0'
    }
    raw_addr_url = "https://services.tokenview.io/vipapi/address/btc/"
    api_key = os.environ.get('APIKEY', '')
    tail = f"/1/50?apikey={api_key}"

    file_root_dir = f"F:/json_data/"
    log_dir = f"F:/log/"
    dataset_path = f"E:/modelproject/BTCanalysis/BABD-13.csv"

    class labels(Enum):
        Blackmail = 0
        CyberSecurity = 1
        DarknetMarket = 2
        Exchange = 3
        P2PFIS = 4
        P2PFS = 5
        Gambling = 6
        CriminalBlacklist = 7
        MoneyLaundering = 8
        PonziScheme = 9
        MiningPool = 10
        Tumbler = 11
        Individual = 12

@retry(retry_on_exception=lambda x: isinstance(x, (aiohttp.ClientError, asyncio.TimeoutError)), 
       stop_max_attempt_number=10, wait_fixed=2000)
async def get_response(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=Consts.head,timeout=aiohttp.ClientTimeout(total=20)) as response:
                response.encoding = "utf-8"
                status:int = response.status
                if (status == 200) and (response is not None):
                    return await response.json()
                elif status == 429:
                    tqdm.write("Rate limit.")
                    time.sleep(10)
                    await get_response(url)
                else:
                    tqdm.write(f"Error status: {status}")
                    await get_response(url)
    except asyncio.TimeoutError:
        tqdm.write("Request timed out.")
        await get_response(url)
    except aiohttp.ClientError as e:
        tqdm.write(f"Client error: {e}")
        await get_response(url)
    except Exception as e:
        tqdm.write(str(e))
        await get_response(url)


async def get_json_by_addr(addr,label,k:int=1):
    file_dir = os.path.join(Consts.file_root_dir, label, f"k={k}")
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    filename = os.path.join(file_dir, f"{addr}.json")
    if os.path.exists(filename):
        return
    try:
        tqdm.write("正在请求地址：{}".format(addr))
        url = Consts.raw_addr_url + addr + Consts.tail
        json_data = await get_response(url)
        if json_data["enMsg"] == "SUCCESS":
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(json_data, f)
        else:
            raise ConnectionError
    except Exception as e:
        tqdm.write(str(e))
        await get_json_by_addr(addr, label, k)


async def process_addr_list(addr_label: zip, batch_size: int = 100):
    addr_label = list(addr_label)
    try:
        tasks = []
        with tqdm(total=len(addr_label), desc=f"Processing addr list", 
                  dynamic_ncols=True, leave=True, unit='addr') as pbar:
            for i in range(0, len(addr_label), batch_size):
                batch = addr_label[i:i + batch_size]
                tasks = [get_json_by_addr(addr, Consts.labels(label).name) for addr, label in batch if addr != 'coinbase']
                await asyncio.gather(*tasks)
                pbar.update(len(batch))

        await asyncio.gather(*tasks)
    except Exception as e:
        tqdm.write(f"Error processing addresses: {e}")

async def process_addr_file(file_path: str,label: str, batch_size: int=100):
    try:
        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as file:
            addresses = await file.read()
            addresses = addresses.splitlines()

        with tqdm(total=len(addresses), desc=f"Processing {label} addresses",
                    dynamic_ncols=True, leave=True, unit='addr') as pbar:
            for i in range(0, len(addresses), batch_size):
                batch = addresses[i:i + batch_size]
                tasks = [get_json_by_addr(addr, label, k=2) for addr in batch if addr != 'coinbase']
                await asyncio.gather(*tasks)
                pbar.update(len(batch))

    except Exception as e:
        tqdm.write(f"Error processing addresses: {e}")

def main():
    from dataset_labels import get_dataset_labels
    addr_label = get_dataset_labels(Consts.dataset_path)
    asyncio.run(process_addr_list(addr_label))

    for label in Consts.labels:
        missing_addr_file = os.path.join(Consts.log_dir, f"{label.name}", "missing_neighbor.txt")
        asyncio.run(process_addr_file(missing_addr_file, label.name))
    
    print("Processing complete.")

if __name__ == "__main__":
    main()