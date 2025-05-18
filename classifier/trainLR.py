import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. 定义逻辑回归模型
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        logits = self.linear(x)
        probas = torch.sigmoid(logits)
        return probas

# 2. 数据准备
# 假设输入数据是一个二维张量，每行是一个序列
# X: [num_samples, input_dim], y: [num_samples] (分类标签)
def train_by_file(model:LogisticRegressionModel,
                  train_loader:DataLoader,
                  test_loader:DataLoader):
    # 4. 定义损失函数和优化器
    # criterion = nn.MSELoss()  # 适用于回归任务
    criterion = nn.CrossEntropyLoss()  # 适用于分类任务
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 5. 模型训练
    num_epochs = 100
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 6. 模型评估
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for batch_X, batch_y in test_loader:
            y_pred = model(batch_X)
            y_pred_classes = torch.argmax(y_pred, dim=1)  # 获取预测类别
            all_preds.append(y_pred_classes)
            all_labels.append(batch_y)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # 计算准确率
        accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy())
        print(f'Accuracy: {accuracy:.4f}')


def main(args,label:str,labels:list[str]):
    '''
    对于labels下的所有*_filter.txt文件进行训练
    其中,label:正类,其他labels为负类
    一共需要训练32个模型
    对于每个模型的训练,读取所有labels下该filter.txt文件,随机采样
    保存训练参数
    '''
    input_dim = 64  # 输入特征维度，由args传入与FMLP的output_dim保持一致
    output_dim = 2  # 输出类别数（对于二分类，输出维度为2）
    for node in range(8):
        for feature in range(4):
            try:
                X = []
                y = []
                # 模型初始化
                model = LogisticRegressionModel(input_dim, output_dim)
                # 获取文件夹下所有{node}{feature}_filter.txt文件
                for l in labels:
                    lines = open(f"{os.path.join(args.output,l)}/{node}{feature}_filter.txt").readlines()
                    lines = [list(map(float,line.strip().split())) 
                            for line in lines if line.strip()]
                    _X = torch.tensor(lines,dtype=torch.float32)
                    X.append(_X)
                    if l == label:
                        y.append(torch.ones(_X.shape[0],dtype=torch.long))
                    else:
                        y.append(torch.zeros(_X.shape[0],dtype=torch.long))
                # 转换为张量
                X = torch.cat(X, dim=0)
                y = torch.cat(y, dim=0)

                # 制作dataloader
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
                print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
                train_dataset = TensorDataset(X_train, y_train)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                test_dataset = TensorDataset(X_test, y_test)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                train_by_file(model, train_loader=train_loader, test_loader=test_loader)
                print(f"Training completed for node {node}, feature {feature}.")
                # 保存模型参数到modelsave文件夹下
                torch.save(model.state_dict(), os.path.join(args.modelsave,f"{node}{feature}_model.pth"))
                print(f"Model saved to {args.modelsave}/{node}{feature}_model.pth")
            except FileNotFoundError:
                continue
    print("All models trained and saved.")

if __name__ == "__main__":
    pass
    main()