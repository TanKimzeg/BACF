from .LR import main as LR
from .LSTM import main as LSTM
from .KNN import main as KNN
from .LightGBM import main as LightGBM
from .XGBoost import main as XGBoost
from .DT import main as DF
from .SVM import main as SVM


__all__ = [
    'LR',
    'LSTM',
    'KNN',
    'XGBoost',
    'DT',
    'LightGBM',
    'SVM'
]