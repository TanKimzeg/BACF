from txGNN import TxGNN
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import time,os

# To create a DataLoader for multiple graphs:
# graphs = []
# for tx in txs:
#     graphs.append(build_tx_graph(tx))
# loader = DataLoader(graphs, batch_size=32, shuffle=True)
class Trainer:
    def __init__(self,model:TxGNN,train_dataloader:DataLoader,
                 eval_dataloader:DataLoader):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        # self.test_dataloader = test_dataloader

    def train_model_with_loader(self,epochs:int=100, lr:float=0.01) -> TxGNN:
        '''
        Train the GNN model with multiple graphs using DataLoader.
        :param model: The TxGNN model.
        :param loader: DataLoader containing multiple graphs.
        :param epochs: Number of training epochs.
        :param lr: Learning rate.
        :return: The trained model.
        '''
        print("Training the model...")
        # 使用 Adam 优化器和 MSE 损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()  # 或 CrossEntropyLoss，根据任务选择

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in self.train_dataloader:  # 每次从 DataLoader 中取出一个批次
                optimizer.zero_grad()

                # 前向传播
                out:torch.Tensor = self.model(batch)  # batch 是一个包含多个图的批次
                # 计算损失
                assert out.shape == batch.y.shape, "输出和目标的维度不匹配!"
                loss:torch.Tensor = criterion(out, batch.y)  # batch.y 是批次中所有图的目标值

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # 打印每个 epoch 的平均损失
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(self.train_dataloader)}")
        return self.model

    def eval_model(self):
        print("Outputting the embedding sequence...")
        # use the model to extract each tx's embedding, 
        embedding_sequence:list[torch.Tensor] = list()
        for batch in self.eval_dataloader:
            embedding:torch.Tensor = self.model(batch).t()
            assert embedding.size(0) == 4
            assert embedding.size(1)%8 == 0
            batch_size = embedding.size(1)//8
            embedding_sequence.extend([embedding[:,i*batch_size:(i+1)*batch_size] 
                                       for i in range(8)])

        # save the embedding sequence to a file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = f"F:/model_save/embedding_{timestamp}.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            for node in embedding_sequence:
                for row in node:
                    f.write(" ".join(map(str, row.tolist())) + "\n")
        print(f"Embedding sequence saved to {save_path}")


    def save(self, save_path: str):
        # Save the trained model
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file_path = f"{save_path}/{timestamp}.pth"
        torch.save(self.model.state_dict(), save_file_path)
        print(f"Trained model saved as {save_file_path}")

    def load(self, model_path: str):
        original_state_dict = self.model.state_dict()
        new_state_dict = torch.load(model_path)
        for key in new_state_dict:
            if key in original_state_dict:
                original_state_dict[key] = new_state_dict[key]
            else:
                raise KeyError(f"Key {key} not found in original state dict")
        self.model.load_state_dict(original_state_dict)