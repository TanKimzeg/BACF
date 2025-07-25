import os
import joblib
from torch import Tensor
from aeon.classification.hybrid import HIVECOTEV1, HIVECOTEV2
from sklearn.metrics import classification_report, accuracy_score

def train(X_train: Tensor, y_train: Tensor, X_test: Tensor,
          y_test: Tensor, save_path: str=None) -> None:
    X_train = X_train.cpu().numpy() if isinstance(X_train, Tensor) else X_train
    y_train = y_train.cpu().numpy() if isinstance(y_train, Tensor) else y_train
    X_test = X_test.cpu().numpy() if isinstance(X_test, Tensor) else X_test
    y_test = y_test.cpu().numpy() if isinstance(y_test, Tensor) else y_test

    model = HIVECOTEV2()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)

    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

def main(args, labels: list[str]):
    from .dataset import data_split
    X_train, y_train, X_test, y_test, X, y = data_split(
        labels=labels, data_root_path=args.data_dir, test_size=0.2)

    train(X_train, y_train, X_test, y_test,
          save_path=os.path.join(args.save_dir, 'hivecote_model.pkl'))

if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    X = np.random.rand(100, 3, 6)  # Adjusted to 100 samples
    y = np.random.randint(0, 3, size=100)  # Adjusted to match the number of samples

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train(X_train, y_train, X_test, y_test)   
