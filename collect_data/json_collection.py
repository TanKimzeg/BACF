import csv
import random
import asyncio
import aiohttp
import aiofiles
import time
import pandas as pd
import os
import json


head = {
    "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0'
}
raw_addr_url = "https://services.tokenview.io/vipapi/address/btc/"
tail = "/1/50?apikey=BTCxTvlBqH16IKBOBv3N"

async def get_response(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=head,timeout=aiohttp.ClientTimeout(connect=10)) as response:
                response.encoding = "utf-8"
                status = response.status
                if (status == 200) and (response is not None):
                    return await response.json()
                elif status == 429:
                    time.sleep(random.randint(1, 5))
                    print("Rate limit.")
                    await get_response(url)
                else:
                    await get_response(url)
    except Exception as e:
        print(e)
        await get_response(url)


async def get_json_by_addr(addr,label):
    try:
        print("正在处理地址：{}".format(addr))
        url = raw_addr_url + addr + tail
        json_data = await get_response(url)
        if json_data["msg"] == "成功":
            filename = os.path.join(f"F:/json_data/{label}/k=2/", "%s.json" % addr)
            # make sure k is right!
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(json_data, f)
            return json_data
        else:
            raise ConnectionError
    except Exception as e:
        # with open("D:/log.txt","a")as exc_file:
        #     exc_file.write(time.ctime())
        #     exc_file.write(addr)
        #     exc_file.write(type(e).__name__)
        print(e)
        await get_json_by_addr(addr)

async def process_addresses(file_path,label):
    try:
        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as file:
            addresses = await file.read()
            addresses = addresses.splitlines()

        tasks = []
        for addr in addresses:
            tasks.append(get_json_by_addr(addr,label))

        await asyncio.gather(*tasks)
    except Exception as e:
        print(f"Error processing addresses: {e}")

def main():
    labels = [
        'gambling',
        # 'mining'
    ]
    for label in labels:
        missing_addr_file = f"F:/log/{label}/missing_addr.txt"  # Replace with the actual path to your file
        asyncio.run(process_addresses(missing_addr_file,label))

if __name__ == "__main__":
    main()