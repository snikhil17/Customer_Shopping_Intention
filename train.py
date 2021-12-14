# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import PowerTransformer
import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import pickle

# Loading Data
df = pd.read_csv("training_data_skf_no_smote.csv")


useful_cols = [col for col in df.columns if col not in ["id", "Revenue", "kfold"]]
categorical = [col for col in useful_cols if df[col].dtype in ["object", "bool"]]
numerical = [col for col in useful_cols if col not in categorical]


df_train, df_test = train_test_split(df, test_size=0.2, random_state=7)

le = preprocessing.LabelEncoder()

df_train.Weekend = le.fit_transform(df_train.Weekend)
df_test.Weekend = le.transform(df_test.Weekend)


dicts = df_train.to_dict(orient="records")
dv = DictVectorizer(sparse=False)
df_train = pd.DataFrame(
    dv.fit_transform(dicts), columns=list(dv.get_feature_names_out())
)
df_test = pd.DataFrame(dv.transform(dicts), columns=list(dv.get_feature_names_out()))

useful_cols = [col for col in df_train.columns if col not in ["id", "Revenue", "kfold"]]
categorical = [col for col in useful_cols if df_train[col].dtype in ["object", "bool"]]
numerical = [col for col in useful_cols if col not in categorical]

pt = PowerTransformer()
pt_num_tr = pd.DataFrame(pt.fit_transform(df_train[useful_cols]), columns=useful_cols)
pt_num_ts = pd.DataFrame(pt.transform(df_test[useful_cols]), columns=useful_cols)
df_train = pd.concat([df_train.drop(useful_cols, axis=1), pt_num_tr], axis=1)
df_test = pd.concat([df_test.drop(useful_cols, axis=1), pt_num_ts], axis=1)

useful_cols = [col for col in df_train.columns if col not in ["id", "Revenue", "kfold"]]
categorical = [col for col in useful_cols if df_train[col].dtype in ["object", "bool"]]
numerical = [col for col in useful_cols if col not in categorical]
scaler = preprocessing.RobustScaler()

df_train.Revenue = df_train.Revenue.astype("int")
df_train.kfold = df_train.kfold.astype("int")
df_test.Revenue = df_test.Revenue.astype("int")
df_test.kfold = df_test.kfold.astype("int")


# Models
params_bg = {"n_estimators": 533, "max_samples": 32}
model_bg = BaggingClassifier(**params_bg, random_state=7)

params_lgb = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "num_leaves": 47,
    "max_depth": 17,
    "learning_rate": 0.331,
    "n_estimators": 469,
    "reg_alpha": 30.654408167803027,
    "reg_lambda": 8.742258358130245,
    "subsample": 0.45,
    "subsample_freq": 9,
    "colsample_bytree": 0.49,
    "min_child_samples": 26,
    "min_child_weight": 32,
}
model_lgb = LGBMClassifier(**params_lgb, random_state=6)


params_mlp = {"alpha": 0.09631013728513668, "hidden_layer_sizes": 7, "max_iter": 30}
model_mlp = MLPClassifier(**params_mlp, random_state=17, tol=1e-4)


params_dt = {
    "max_leaf_nodes": 6,
    "max_depth": 254,
    "criterion": "entropy",
    "class_weight": "balanced",
}
model_dt = DecisionTreeClassifier(**params_dt, random_state=42)

"""### **Final Model: Voting Classifier**"""

scores_train = []
scores_valid = []
for fold in range(5):
    xtrain = df_train[df_train.kfold != fold].reset_index(drop=True)
    xvalid = df_train[df_train.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()

    ytrain = xtrain.Revenue
    yvalid = xvalid.Revenue

    xtrain = xtrain[useful_cols]
    xvalid = xvalid[useful_cols]
    xtest = xtest[useful_cols]

    xtrain[numerical] = scaler.fit_transform(xtrain[numerical])
    xvalid[numerical] = scaler.transform(xvalid[numerical])
    xtest[numerical] = scaler.transform(xtest[numerical])

    model_vclf = VotingClassifier(
        estimators=[
            ("BaggingClassifier", model_bg),
            ("LightGBM", model_lgb),
            ("MLPClassifier", model_mlp),
            ("DecisionTree", model_dt),
        ],
        voting="hard",
    )
    model_vclf.fit(xtrain, ytrain)

    preds_valid = model_vclf.predict(xvalid)
    preds_train = model_vclf.predict(xtrain)
    f1_score_valid = metrics.f1_score(yvalid, preds_valid)
    f1_score_train = metrics.f1_score(ytrain, preds_train)
    print(f"Training Acc for fold: {fold}: {model_vclf.score(xtrain,ytrain)}")
    print(f"Validation Acc for fold: {fold}: {model_vclf.score(xvalid,yvalid)}")
    print(f"Fold {fold} f1-score-train: ", f1_score_train)
    print(f"Fold {fold} f1-score-Valid: ", f1_score_valid)
    scores_train.append(f1_score_train)
    scores_valid.append(f1_score_valid)
print(np.mean(scores_train), np.std(scores_train))
print(np.mean(scores_valid), np.std(scores_valid))

"""## **We can use the final model i.e. ``voting classifier`` for deployment as it has decent mean_f1-score and satisfactory standard deviation.**"""

"""Save the Bagging CLF model"""
output_file = f"model_bg.bin"

with open(output_file, "wb") as f_out:
    pickle.dump(model_bg, f_out)

print(f"the model is saved to {output_file}")


"""Save the LGB model"""
output_file = f"model_lgb.bin"

with open(output_file, "wb") as f_out:
    pickle.dump(model_lgb, f_out)


print(f"the model is saved to {output_file}")


"""Save the DT model"""
output_file = f"model_dt.bin"

with open(output_file, "wb") as f_out:
    pickle.dump(model_dt, f_out)
print(f"the model is saved to {output_file}")


"""Save the MLP model"""
output_file = f"model_mlp.bin"

with open(output_file, "wb") as f_out:
    pickle.dump(model_mlp, f_out)
print(f"the model is saved to {output_file}")


"""Save the Voting CLF model"""
output_file = f"model_vclf.bin"
with open(output_file, "wb") as f_out:
    pickle.dump(model_vclf, f_out)
print(f"the model is saved to {output_file}")


def preprocess_train(df_train, y_train):

    # Label-Encoding boolean variable:
    le = preprocessing.LabelEncoder()
    df_train.Weekend = le.fit_transform(df_train.Weekend)

    # OHE
    dicts = df_train.to_dict(orient="records")
    dv = DictVectorizer(sparse=False)
    df_train = pd.DataFrame(
        dv.fit_transform(dicts), columns=list(dv.get_feature_names_out())
    )
    columns = list(dv.get_feature_names_out())

    # PT
    pt = PowerTransformer()
    pt_num_tr = pd.DataFrame(pt.fit_transform(df_train[columns]), columns=columns)
    df_train = pd.concat([df_train.drop(columns, axis=1), pt_num_tr], axis=1)

    # Scaling
    scaler = preprocessing.RobustScaler()
    df_train = scaler.fit_transform(df_train)

    # Models
    params_bg = {"n_estimators": 533, "max_samples": 32}
    model_bg = BaggingClassifier(**params_bg, random_state=7)

    params_lgb = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "num_leaves": 47,
        "max_depth": 17,
        "learning_rate": 0.331,
        "n_estimators": 469,
        "reg_alpha": 30.654408167803027,
        "reg_lambda": 8.742258358130245,
        "subsample": 0.45,
        "subsample_freq": 9,
        "colsample_bytree": 0.49,
        "min_child_samples": 26,
        "min_child_weight": 32,
    }
    model_lgb = LGBMClassifier(**params_lgb, random_state=6)

    params_mlp = {"alpha": 0.09631013728513668, "hidden_layer_sizes": 7, "max_iter": 30}
    model_mlp = MLPClassifier(**params_mlp, random_state=17, tol=1e-4)

    params_dt = {
        "max_leaf_nodes": 6,
        "max_depth": 254,
        "criterion": "entropy",
        "class_weight": "balanced",
    }
    model_dt = DecisionTreeClassifier(**params_dt, random_state=42)

    # Final Model
    model = VotingClassifier(
        estimators=[
            ("BaggingClassifier", model_bg),
            ("LightGBM", model_lgb),
            ("MLPClassifier", model_mlp),
            ("DecisionTree", model_dt),
        ],
        voting="hard",
    )
    model.fit(df_train, y_train)

    return le, dv, pt, scaler, model


def predict(df, pt, scaler, dv, model):
    df.Weekend = le.transform(df.Weekend)
    dicts = df.to_dict(orient="records")
    X = dv.transform(dicts)
    X = pt.transform(X)
    X = scaler.transform(X)
    y_pred = model.predict(X)

    return y_pred


print("Training on created functions:")
# """Trying out Model-Prediction on first 20 rows"""
xtrain = df.drop("Revenue", axis=1).copy()
ytrain = df.Revenue.copy()
le, dv, pt, scaler, model = preprocess_train(xtrain, ytrain)

df_q = df.head(20)
xtrain = df_q.drop(["Revenue", "kfold"], axis=1)
ytrain = df_q.Revenue
preds_train = predict(xtrain, pt, scaler, dv, model)
print([(i, j) for i, j in zip(ytrain, preds_train)][:20])

output_file = f"model_final.bin"
with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)

print(f"the model is saved to {output_file}")
