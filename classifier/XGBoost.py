import os
from torch import Tensor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train(X_train: Tensor, y_train: Tensor, X_test: Tensor, 
          y_test: Tensor, num_class:int, save_path: str=None) -> None:
    # 在XGB中, 输入数据需要是二维的
    # 如果数据是多维的, 需要展平为二维张量
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    assert X_train.ndim == 2, f"X_train should be 2D, got {X_train.ndim}D"
    assert X_test.ndim == 2, f"X_test should be 2D, got {X_test.ndim}D"
    assert y_train.size(0) == X_train.size(0)
    assert y_test.size(0) == X_test.size(0)
    
    X_train = X_train.cpu().numpy() if isinstance(X_train, Tensor) else X_train
    y_train = y_train.cpu().numpy() if isinstance(y_train, Tensor) else y_train
    X_test = X_test.cpu().numpy() if isinstance(X_test, Tensor) else X_test
    y_test = y_test.cpu().numpy() if isinstance(y_test, Tensor) else y_test
    
    # 初始化XGBoost分类器
    xgb_model = XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', num_class=num_class)
    
    # 训练模型
    xgb_model.fit(X_train, y_train)

    # 保存模型
    if save_path:
        joblib.dump(xgb_model, save_path)

    # 预测
    y_pred = xgb_model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Test Accuracy: {accuracy:.4f}")
    # 打印分类报告
    print(classification_report(y_test, y_pred))

def main(args, labels: list[str]):
    from .dataset import data_split
    X_train, y_train, X_test, y_test, X, y = data_split(
        labels=labels, data_root_path=args.data_dir, test_size=0.2)

    train(X_train, y_train, X, y, num_class=len(labels),
          save_path=os.path.join(args.modelsave, 'xgboost_model.pkl'))

if __name__ == "__main__":  
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    # 生成一个示例数据集
    X, y = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42)
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 训练模型
    train(X_train, y_train, X_test, y_test,num_class=2)
