import random

import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
from sklearn.svm import SVC

random.seed(42)

train_df = pd.read_pickle("train.pkl")
test_df = pd.read_pickle("test.pkl")
validation_df = pd.read_pickle("validation.pkl")
onehot = sklearn.preprocessing.OneHotEncoder()

# collapse data
mapping_dict = {
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 3,
}
train_df["meta"] = train_df["meta"].map(mapping_dict)
test_df["meta"] = test_df["meta"].map(mapping_dict)
validation_df["meta"] = validation_df["meta"].map(mapping_dict)

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
auc_val = sklearn.metrics.roc_auc_score(
    Y_val_onehot, result_val, average="macro", multi_class="ovr"
)
print(f"f1_val: {f1_val:.4f}, auc_val: {auc_val:.4f}")
classification_report = sklearn.metrics.classification_report(
    Y_val, result_val.argmax(axis=1)
)
print(classification_report)
confusion_matrix = sklearn.metrics.confusion_matrix(Y_val, result_val.argmax(axis=1))
print(confusion_matrix)

result_test = clf.predict_proba(X_test)
Y_test_onehot = onehot.fit_transform(Y_test.reshape(-1, 1)).toarray()
f1_test = sklearn.metrics.f1_score(Y_test, result_test.argmax(axis=1), average="macro")
auc_test = sklearn.metrics.roc_auc_score(
    Y_test_onehot, result_test, average="macro", multi_class="ovr"
)
classification_report = sklearn.metrics.classification_report(
    Y_test, result_test.argmax(axis=1)
)
confusion_matrix = sklearn.metrics.confusion_matrix(Y_test, result_test.argmax(axis=1))

with open("result.txt", "w") as f:
    f.write(f"f1_test: {f1_test:.4f}, auc_test: {auc_test:.4f}")
    f.write(classification_report)
    f.write(repr(confusion_matrix))
# print(f"f1_test: {f1_test:.4f}, auc_test: {auc_test:.4f}")


# print(classification_report)
# print(confusion_matrix)
