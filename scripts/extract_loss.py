import os
import pandas as pd

loss = []
with open(os.path.abspath("./data/lstm.log"), 'r', encoding='utf-16') as f:
    lines = f.readlines()
    for line in lines:
        if "Test Loss:" in line:
            loss.append(float(line.split(", Accuracy:")[0].split(":")[-1].strip()))

# 保存为CSV文件
df = pd.DataFrame({'Loss': loss})
df.to_csv('./data/lstm_loss.csv', index=True, encoding='utf-8-sig')
print("Loss values saved to ./data/lstm_loss.csv")