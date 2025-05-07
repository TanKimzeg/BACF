import json
import os
import asyncio
import aiofiles
import pandas as pd
import shutil


async def copy(k, root_addr, dst):
    if k == 1:
        src = f"F:/json_data/k=1/{root_addr}.json"
        dst_path = f"{dst}/{root_addr}/{root_addr}.json"
        if os.path.exists(src) and not os.path.exists(dst_path):
            await shutil.copyfile(src, dst_path)
    else:
        former_path = f"{dst}/{root_addr}/k={k-1}/"
        if not os.path.exists(f"{dst}/{root_addr}/k={k}/"):
            os.makedirs(f"{dst}/{root_addr}/k={k}/")

        file_list = os.listdir(former_path)
        addr_list = []

        for json_file_name in file_list:
            async with aiofiles.open(os.path.join(former_path, json_file_name), "r") as file:
                json_file = await file.read()
                json_data = json.loads(json_file)
                for tx in json_data["data"][0]["txs"]:
                    for inputs in tx["inputs"]:
                        if "address" in inputs:
                            addr_list.append(inputs["address"])
                    for outputs in tx["outputs"]:
                        if "address" in outputs:
                            addr_list.append(outputs["address"])

        tasks = []
        for addr in addr_list:
            src = f"F:/json_data/k={k}/{addr}.json"
            dst_path = f"{dst}/{root_addr}/k={k}/{addr}.json"
            if os.path.exists(src) and not os.path.exists(dst_path):
                tasks.append(shutil.copyfile(src, dst_path))

        await asyncio.gather(*tasks)

async def process_company(company, root_addr_file_path, dst):
    addr_file_list = os.listdir(root_addr_file_path)
    for file_name in addr_file_list:
        with open(os.path.join(root_addr_file_path, file_name), "r") as f:
            df = pd.read_csv(f, skiprows=1)
            tasks = []
            for addr in df["address"]:
                if not os.path.exists(os.path.join(dst, addr)):
                    print(f"Dealing address: {addr}")
                    tasks.append(copy(1, addr, dst))
                    # tasks.append(copy(2, addr, dst))
                    # tasks.append(copy(3, addr, dst))
                    # tasks.append(copy(4, addr, dst))
            await asyncio.gather(*tasks)

async def main():
    company_list = ['CloudBet.com', 'CoinGaming.io']
    tasks = []
    for company in company_list:
        root_addr_file_path = f"F:/raw_addr_csv/{company}/"
        dst = f"F:/rebuild_dir/{company}/"
        tasks.append(process_company(company, root_addr_file_path, dst))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
