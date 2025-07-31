import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split

def load_dataset(data_path: str) -> torch.Tensor:
    '''
    torch.Size() = [addr_num, timesteps, feature_num]
    '''
    feature_list = []

    for fn in os.listdir(data_path):
        if fn.endswith('.txt') and not (fn.endswith('_filter.txt') or fn.endswith('_sample.txt')):
            file_path = os.path.join(data_path, fn)
            with open(file_path, 'r') as data_file:
                seq = list()
                for line in data_file:
                    seq.append(list(map(float, line.strip().split(' '))))
                # 计算max_seq_len
                max_seq_len = max([len(l) for l in seq])
            # 添加前导0
            for i in range(len(seq)):
                seq[i] = [0.0]*(max_seq_len - len(seq[i])) + seq[i]
            # 转换为numpy数组
            seq = np.array(seq, dtype=np.float32)
            feature_list.append(seq)

    # 堆叠所有地址的特征，形成三维张量
    combined_features = np.stack(feature_list, axis=0)  # 按地址堆叠

    # 转换为 PyTorch Tensor
    return torch.tensor(combined_features, dtype=torch.long).permute(1, 2, 0)

def data_split(labels: list[str], data_root_path: str, test_size: float=0.2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    :return: X_train, y_train, X_test, y_test, X, y
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X: list[torch.Tensor] = []
    y: list[torch.Tensor] = []
    for idx, label in enumerate(labels):
        data_path = os.path.join(data_root_path, label)
        _X = load_dataset(data_path)
        X.append(_X)
        y.append(torch.tensor([idx] * _X.shape[0], dtype=torch.long))

    X: torch.Tensor = torch.cat(X, dim=0)
    y: torch.Tensor = torch.cat(y, dim=0)

    assert len(X) == len(y), f"X.shape = {X.shape}, y.shape = {y.shape}"

    # 制作dataloader
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    X_train_tensor = X_train.clone().detach().float().to(device)
    del X_train
    y_train_tensor = y_train.clone().detach().long().to(device)
    del y_train
    X_test_tensor = X_test.clone().detach().float().to(device)
    del X_test
    y_test_tensor = y_test.clone().detach().long().to(device)
    del y_test

    return (X_train_tensor, y_train_tensor, 
            X_test_tensor, y_test_tensor, X, y)

def get_loaders(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X, y,batch_size: int=32) -> list[DataLoader]:
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    eval_dataset = TensorDataset(X, y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    return [train_loader, test_loader, eval_loader]

if __name__ == "__main__":
    # 示例用法
    data_path = os.path.abspath('./data/')  # 替换为实际数据路径
    dataset = load_dataset(data_path)
    print(dataset.shape)  # 输出数据集的形状
    print(dataset[0, 0, :])  # 打印第一个地址的第一个特征的所有时序数据
    print(dataset[0, 1, :])  # 打印第一个地址的第二个特征的所有时序数据