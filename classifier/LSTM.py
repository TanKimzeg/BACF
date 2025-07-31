import os
import torch
from tqdm import tqdm
from .dataset import get_loaders
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 定义LSTM分类模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=0.2,  # 层间dropout
            bidirectional=False  # 单向LSTM
        )
        
        # 全连接层
        if self.lstm.bidirectional:
            self.fc1 = nn.Linear(hidden_size * 2, 64)
        else:
            self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(0.3)
        
        # 激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32).to(device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 只取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 全连接层
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# 2. 数据准备和训练函数
def train(model: LSTMClassifier, train_loader, epoch: int, optimizer: optim.Optimizer):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 模型训练
    model.train()
    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        batch_X = batch_X.to(device).float()
        batch_y = batch_y.to(device)

        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            tqdm.write(f'Epoch [{epoch + 1}], Batch [{batch_idx + 1}/{len(train_loader)}], '
                    f'Loss: {loss.item():.4f}')

    # 模型评估
def test(model: LSTMClassifier, test_loader) -> float:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        all_preds = []
        all_labels = []
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device).float()
            batch_y = batch_y.to(device)

            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)

            test_loss += loss.item()
            y_pred_classes = torch.argmax(y_pred, dim=1)
            all_preds.append(y_pred_classes)
            all_labels.append(batch_y)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

    # 计算准确率
    accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy())
    avg_loss = test_loss / len(test_loader)
    tqdm.write(f'Test Loss: {test_loss/len(test_loader)}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy


def main(args, labels: list[str],dim: tuple[int, int]):
    hidden_dim = 128  # LSTM隐藏层维度
    num_layers = 2  # LSTM层数
    num_classes = len(labels)  # 类别数

    best_acc = 0.0  # 用于保存最佳准确率
    train_loader, test_loader, eval_loader = get_loaders(
         labels=labels, data_root_path=args.data_dir, batch_size=args.batch_size)

    model = LSTMClassifier(
        input_size=dim[0]*dim[1], hidden_size=hidden_dim, 
        num_layers=num_layers, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #  学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min', factor=0.1, patience=5, verbose=True
    )

    for epoch in tqdm(range(args.epochs), desc="LSTM Training Progress", 
                      unit='epoch', dynamic_ncols=True):
        train(model, train_loader=train_loader, epoch=epoch, optimizer=optimizer)
        test_loss, test_acc = test(model, test_loader=test_loader)

        scheduler.step(test_loss)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.modelsave, "lstm_model.pth"))
            tqdm.write(f"Model saved to {args.modelsave}/lstm_model.pth with accuracy: {test_acc:.4f}")

    print("All models trained and saved.")

    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(args.modelsave, "lstm_model.pth")))
    evaluate(model, eval_loader, labels)

def evaluate(model: LSTMClassifier, eval_loader, labels: list[str]):
    # 生成分类报告
    print('Generating classification report...')
    from sklearn.metrics import classification_report
    model.eval()    
    all_preds = []  
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in eval_loader:
            batch_X = batch_X.to(device).float()
            y_pred = model(batch_X)
            y_pred_classes = torch.argmax(y_pred, dim=1)
            all_preds.append(y_pred_classes.cpu())
            all_labels.append(batch_y.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    print(classification_report(all_labels.numpy(), all_preds.numpy(), target_names=labels, digits=4))

if __name__ == "__main__":
    pass