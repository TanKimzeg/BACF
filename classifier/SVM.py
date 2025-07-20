import os
import joblib
from sklearn.svm import SVC
from torch import Tensor
from sklearn.metrics import accuracy_score, classification_report

def train(X_train: Tensor, y_train: Tensor, X_test: Tensor, 
          y_test: Tensor, save_path: str=None) -> None:
    X_train = X_train.cpu().numpy() if isinstance(X_train, Tensor) else X_train
    y_train = y_train.cpu().numpy() if isinstance(y_train, Tensor) else y_train
    X_test = X_test.cpu().numpy() if isinstance(X_test, Tensor) else X_test
    y_test = y_test.cpu().numpy() if isinstance(y_test, Tensor) else y_test

    # 在SVM中, 输入数据需要是二维的
    # 如果数据是多维的, 需要展平为二维张量
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    assert X_train.ndim == 2, f"X_train should be 2D, got {X_train.ndim}D"
    assert X_test.ndim == 2, f"X_test should be 2D, got {X_test.ndim}D"
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]

    # 初始化SVM分类器
    svm_model = SVC(kernel='linear', probability=True)

    # 训练模型
    svm_model.fit(X_train, y_train)

    # 保存模型
    if save_path:
        joblib.dump(svm_model, save_path)

    # 预测
    y_pred = svm_model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Test Accuracy: {accuracy:.4f}")

    # 打印分类报告
    print(classification_report(y_test, y_pred))


def main(args, labels: list[str]):
    from .dataset import data_split
    X_train, y_train, X_test, y_test, X, y = data_split(
        labels=labels, data_root_path=args.data_dir, test_size=0.2)

    train(X_train, y_train, X, y,
          save_path=os.path.join(args.modelsave, 'svm_model.pkl'))

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # 生成一个示例数据集
    X, y = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练模型
    train(X_train, y_train, X_test, y_test)