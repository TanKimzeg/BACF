import pandas as pd


def get_dataset_labels(dataset_path: str) -> zip:
    df = pd.read_csv(dataset_path)
    accounts = df['account'].tolist()
    labels = df['label'].tolist()
    return zip(accounts, labels)

