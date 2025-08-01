import os
import joblib
from torch import Tensor
from sklearn.ensemble import RandomForestClassifier
from aeon.classification.distance_based import ProximityForest as AeonRF
from sklearn.metrics import accuracy_score, classification_report


def train(X_train: Tensor, y_train: Tensor, X_test: Tensor,
          y_test: Tensor, save_path: str=None) -> None:
    X_train = X_train.cpu().numpy() if isinstance(X_train, Tensor) else X_train
    y_train = y_train.cpu().numpy() if isinstance(y_train, Tensor) else y_train
    X_test = X_test.cpu().numpy() if isinstance(X_test, Tensor) else X_test
    y_test = y_test.cpu().numpy() if isinstance(y_test, Tensor) else y_test

    # 在KNN中, 输入数据需要是二维的
    # 如果数据是多维的, 需要展平为二维张量
    # X_train = X_train.reshape(X_train.shape[0], -1)
    # X_test = X_test.reshape(X_test.shape[0], -1)
    # assert X_train.ndim == 2, f"X_train should be 2D, got {X_train.ndim}D"
    # assert X_test.ndim == 2, f"X_test should be 2D, got {X_test.ndim}D"
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]

    # rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier = AeonRF(n_jobs=-1)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)

    # 保存模型
    if save_path:
        joblib.dump(rf_classifier, save_path)
    print(classification_report(y_test, y_pred))
    print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.2f}")

def main(args, labels: list[str]):
    from .dataset import data_split
    X_train, y_train, X_test, y_test, X, y = data_split(
        labels=labels, data_root_path=args.data_dir, test_size=0.2)

    train(X_train, y_train, X, y,
          save_path=os.path.join(args.save_dir, 'rf_model.pkl'))

if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    X = np.random.rand(50,2,5)  # Adjusted to 50 samples
    y = np.random.randint(0,3,size=50)  # Adjusted to match the number of samples
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train(X_train, y_train, X_test, y_test)