import random
from itertools import chain

import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVC

random.seed(42)

for n in chain(range(1, 10), range(10, 100, 10), range(100, 2048, 100)):
    train_df = pd.read_pickle("train.pkl")
    test_df = pd.read_pickle("test.pkl")
    validation_df = pd.read_pickle("validation.pkl")
    onehot = sklearn.preprocessing.OneHotEncoder()
    pca = PCA(n_components=n)

    X_train = train_df["feature"].values
    X_train = np.stack(X_train)
    X_train = pca.fit_transform(X_train)
    Y_train = train_df["meta"].values
    X_val = validation_df["feature"].values
    X_val = np.stack(X_val)
    X_val = pca.transform(X_val)
    Y_val = validation_df["meta"].values
    X_test = test_df["feature"].values
    X_test = np.stack(X_test)
    X_test = pca.transform(X_test)
    Y_test = test_df["meta"].values

    clf = SVC(probability=True)
    clf.fit(X_train, Y_train)
    result_val = clf.predict_proba(X_val)
    Y_val_onehot = onehot.fit_transform(Y_val.reshape(-1, 1)).toarray()
    f1_val = sklearn.metrics.f1_score(Y_val, result_val.argmax(axis=1), average="macro")
    auc_val = sklearn.metrics.roc_auc_score(Y_val_onehot, result_val)
    print(f"n: {n} f1_val: {f1_val:.4f}, auc_val: {auc_val:.4f}")
    result_test = clf.predict_proba(X_test)
    Y_test_onehot = onehot.fit_transform(Y_test.reshape(-1, 1)).toarray()
    f1_test = sklearn.metrics.f1_score(
        Y_test, result_test.argmax(axis=1), average="macro"
    )
    auc_test = sklearn.metrics.roc_auc_score(Y_test_onehot, result_test)
    print(f"n: {n} f1_test: {f1_test:.4f}, auc_test: {auc_test:.4f}")
