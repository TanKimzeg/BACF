import os
from . import LSTM, DT, KNN, LightGBM, XGBoost, LR, SVM, RF, hive_cote, mlp
from .dataset import get_loaders, data_split
from sklearnex import patch_sklearn, unpatch_sklearn

def combine_models(labels: list[str], classifier_args, dim: tuple[int, int]):
    patch_sklearn()
    X_train, y_train, X_test, y_test, X, y = data_split(
        labels=labels, data_root_path=classifier_args.data_dir, test_size=0.2)
    DT(X_train, y_train, X_test, y_test,
       save_path=os.path.join(classifier_args.modelsave, 'dt_model.pkl'))
    KNN(X_train, y_train, X_test, y_test,
        save_path=os.path.join(classifier_args.modelsave, 'knn_model.pkl'))
    LightGBM(X_train, y_train, X_test, y_test,
             save_path=os.path.join(classifier_args.modelsave, 'lightgbm_model.pkl'))
    LR(X_train, y_train, X_test, y_test,
       save_path=os.path.join(classifier_args.modelsave, 'lr_model.pkl'))
    train_loader, test_loader, eval_loader = get_loaders(X_train, y_train, X_test, y_test, X, y,
          batch_size=classifier_args.batch_size)
    LSTM(train_loader, test_loader, eval_loader,
         labels=labels,args=classifier_args, dim=dim)
    RF(X_train, y_train, X_test, y_test,
       save_path=os.path.join(classifier_args.modelsave, 'rf_model.pkl'))
    SVM(X_train, y_train, X_test, y_test,
        save_path=os.path.join(classifier_args.modelsave, 'svm_model.pkl'))
    XGBoost(X_train, y_train, X_test, y_test, num_class=len(labels),
            save_path=os.path.join(classifier_args.modelsave, 'xgboost_model.pkl'))
    hive_cote(X_train, y_train, X_test, y_test,
              save_path=os.path.join(classifier_args.modelsave, 'hive_cote_model.pkl'))
    mlp(X_train, y_train, X_test, y_test,
            save_path=os.path.join(classifier_args.modelsave, 'mlp_model.pkl'))
    unpatch_sklearn()
