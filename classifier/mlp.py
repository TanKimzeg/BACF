import os
import joblib
from torch import Tensor
from aeon.classification.deep_learning import MLPClassifier as mlp
from sklearn.metrics import classification_report, accuracy_score

def train(X_train: Tensor, y_train: Tensor, X_test: Tensor, 
          y_test: Tensor, save_path: str=None) -> None:
    X_train = X_train.cpu().numpy() if isinstance(X_train, Tensor) else X_train
    y_train = y_train.cpu().numpy() if isinstance(y_train, Tensor) else y_train
    X_test = X_test.cpu().numpy() if isinstance(X_test, Tensor) else X_test
    y_test = y_test.cpu().numpy() if isinstance(y_test, Tensor) else y_test
    assert X_train.ndim == 3
    assert X_train.shape[0] == y_train.shape[0]

    model = mlp()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if save_path:
        joblib.dump(model, save_path)

    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")


def main(args, labels: list[str]):
    from .dataset import data_split
    X_train, y_train, X_test, y_test, X, y = data_split(
        labels=labels, data_root_path=args.data_dir, test_size=0.2)

    train(X_train, y_train, X, y,
          save_path=os.path.join(args.save_dir, 'mlp_model.pkl'))


if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    X = np.random.rand(50,2,5)  # Adjusted to 50 samples
    y = np.random.randint(0,3,size=50)  # Adjusted to match the number of samples

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train(X_train, y_train, X_test, y_test)