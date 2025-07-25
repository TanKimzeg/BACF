import os
from torch import Tensor
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.model_selection import GridSearchCV

def train(X_train: Tensor, y_train: Tensor, X_test: Tensor, 
          y_test: Tensor, save_path: str=None) -> None:
    X_train = X_train.cpu().numpy() if isinstance(X_train, Tensor) else X_train
    y_train = y_train.cpu().numpy() if isinstance(y_train, Tensor) else y_train
    X_test = X_test.cpu().numpy() if isinstance(X_test, Tensor) else X_test
    y_test = y_test.cpu().numpy() if isinstance(y_test, Tensor) else y_test
    
    # 在LightGBM中, 输入数据需要是二维的
    # 如果数据是多维的, 需要展平为二维张量
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    assert X_train.ndim == 2, f"X_train should be 2D, got {X_train.ndim}D"
    assert X_test.ndim == 2, f"X_test should be 2D, got {X_test.ndim}D"
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]
    
    # 初始化LightGBM分类器
    estimator = LGBMClassifier(num_leaves=31)
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [20, 40, 60],
    }
    gbm = GridSearchCV(estimator, param_grid)
    gbm.fit(X_train, y_train)
    print(f"Best parameters found: {gbm.best_params_}")
    if save_path:
        joblib.dump(gbm, save_path)
    print("Generating classification report...")
    y_pred = gbm.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"LightGBM Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")


def main(args, labels: list[str]):
    from .dataset import data_split
    X_train, y_train, X_test, y_test, X, y = data_split(
        labels=labels, data_root_path=args.data_dir, test_size=0.2)

    train(X_train, y_train, X, y, num_class=len(labels), 
          save_path=os.path.join(args.modelsave, 'lightgbm_model.pkl'))
    # gbm = LGBMClassifier(num_leaves=31, learning_rate=args.lr, n_estimators=20)
    # gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='logloss')
    # 保存模型
    # 预测
    # y_pred = gbm.predict(X, num_iteration=gbm.best_iteration_)
    # 计算准确率
    # accuracy = accuracy_score(y, y_pred)
    # print(f"LightGBM Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # 生成一个示例数据集
    X, y = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练模型
    train(X_train, y_train, X_test, y_test)