import pickle

from flask import Flask
from flask import request
from flask import jsonify
# from lightgbm import LGBMClassifier
# from sklearn import preprocessing
# from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import metrics
# from sklearn.feature_extraction import DictVectorizer
# import warnings

# warnings.filterwarnings("ignore")
# from sklearn.preprocessing import PowerTransformer

# from lightgbm import LGBMClassifier
# from sklearn.ensemble import VotingClassifier
# import pickle

app = Flask("customer_pred")
model_file = "model_final.bin"
# model_bg = "model_bg.bin"
# model_mlp = "model_mlp.bin"
# model_lgb = "model_lgb.bin"
# model_dt = "model_dt.bin"
data_preprocess = "pre_processing.bin"

with open(model_file, "rb") as f_in:
    model_final = pickle.load(f_in)

# with open(model_bg, "rb") as f_1:
#     model_bgc = pickle.load(f_1)

# with open(model_mlp, "rb") as f_2:
#     model_mlp = pickle.load(f_2)

# with open(model_lgb, "rb") as f_3:
#     model_lgb = pickle.load(f_3)

# with open(model_dt, "rb") as f_4:
#     model_dt = pickle.load(f_4)

with open(data_preprocess, "rb") as f_p:
    le, dv, pt, scaler = pickle.load(f_p)



@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()
    # X = le.transform(customer)
    X = dv.transform([customer])
    X = pt.transform(X)
    X = scaler.transform(X)
    y_pred = model_final.predict(X)
    customer_purchase = y_pred == 1

    result = {
        "customer_purchase?": bool(customer_purchase),
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
