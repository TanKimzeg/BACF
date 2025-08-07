import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib

def get_txcount(data_root_path, labels: list[str]) -> dict[str, list[int]]:
    txcount = {label: [] for label in labels}
    for label in labels:
        label_dir = os.path.join(data_root_path, label, "k=1")
        if not os.path.exists(label_dir):
            continue
        for file in os.listdir(label_dir):
            if not file.endswith(".json"):
                continue
            with open(os.path.join(label_dir, file), 'r') as f:
                data = json.load(f)["data"][0]
                if "txCount" in data:
                    txcount[label].append(data["txCount"])
    return txcount

def plot_violin_distribution(txcount: dict[str, list[int]], output_path: str|None=None):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    
    # 将数据转换为适合 Violin Plot 的格式
    data = []
    labels = []
    for label, counts in txcount.items():
        if counts:
            data.extend(counts)
            labels.extend([label] * len(counts))
    
    # 创建 Violin Plot
    sns.violinplot(x=labels, y=data, scale="area", inner="quartile", palette="muted")
    
    # 添加标题和标签
    # plt.title('Transaction Count Distribution (Violin Plot)', fontsize=16)
    # plt.xlabel('Labels', fontsize=12)
    plt.ylabel('Transaction Count', fontsize=20)
    plt.ylim(-1_000, 12_000)
    plt.xticks(rotation=45)
    # 去掉网格线
    plt.grid(False)
    # 边框黑色
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    ax.yaxis.set_tick_params(which='major', direction='in', length=6)
    ax.yaxis.set_ticks_position('left')

    # 保存或显示图像
    plt.tight_layout()
    plt.draw()  # 强制渲染
    if output_path:
        plt.savefig(output_path, format='svg')
    else:
        plt.show()

if __name__ == "__main__":
    data_root_path = rf"F:/json_data/"
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
    tx_count = get_txcount(data_root_path, labels)
    joblib.dump(tx_count, os.path.join(rf".\draw_fig", "txcount_distribution.pkl"))

    tx_count = joblib.load(os.path.join(rf".\draw_fig", "txcount_distribution.pkl"))
    plot_violin_distribution(tx_count, output_path=rf".\draw_fig\txcount_distribution.svg")