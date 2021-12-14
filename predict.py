import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = "model_final.bin"

with open(model_file, "rb") as f_in:
    le, dv, pt, scaler, model = pickle.load(f_in)

app = Flask("customer_intention_prediction")


@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    X = pt.transform([customer])
    X = scaler.transform([customer])
    y_pred = model.predict(X)
    # customer_purchase = y_pred >= 0.5

    result = {
        # "customer_purchase_probability": float(y_pred),
        "customer_purchase?": int(y_pred),
    }

    return jsonify(result)
    # df.Weekend = le.transform(df.Weekend)
    # dicts = df.to_dict(orient="records")
    # X = dv.transform(dicts)
    # X = pt.transform(X)
    # X = scaler.transform(X)

    # y_pred = model.predict_proba(X)[:, 1]
    # match_win = y_pred >= 0.5

    # result = {
    #     'win_probability': float(y_pred),
    #     'match_win': bool(match_win)
    # }

    # return jsonify(result)

    # return y_pred


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
