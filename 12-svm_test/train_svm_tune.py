import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
from sklearn.svm import SVC

train_df = pd.read_pickle("train.pkl")
test_df = pd.read_pickle("test.pkl")
validation_df = pd.read_pickle("validation.pkl")
onehot = sklearn.preprocessing.OneHotEncoder()


X_train = train_df["feature"].values
X_train = np.stack(X_train)
Y_train = train_df["meta"].values
X_val = validation_df["feature"].values
X_val = np.stack(X_val)
Y_val = validation_df["meta"].values
X_test = test_df["feature"].values
X_test = np.stack(X_test)
Y_test = test_df["meta"].values

clf = SVC(probability=True)
clf.fit(X_train, Y_train)
result_val = clf.predict_proba(X_val)
Y_val_onehot = onehot.fit_transform(Y_val.reshape(-1, 1)).toarray()
f1_val = sklearn.metrics.f1_score(Y_val, result_val.argmax(axis=1), average="macro")
auc_val = sklearn.metrics.roc_auc_score(Y_val_onehot, result_val)
print(f"f1_val: {f1_val}, auc_val: {auc_val}")
result_test = clf.predict_proba(X_test)
Y_test_onehot = onehot.fit_transform(Y_test.reshape(-1, 1)).toarray()
f1_test = sklearn.metrics.f1_score(Y_test, result_test.argmax(axis=1), average="macro")
auc_test = sklearn.metrics.roc_auc_score(Y_test_onehot, result_test)
print(f"f1_test: {f1_test}, auc_test: {auc_test}")
