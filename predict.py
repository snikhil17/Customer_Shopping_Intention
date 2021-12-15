import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = "model_final.bin"

with open(model_file, "rb") as f_in:
    le, dv, pt, scaler, model = pickle.load(f_in)

app = Flask("customer_pred")


@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()
    # X = le.transform(customer)
    X = dv.transform([customer])
    X = pt.transform(X)
    X = scaler.transform(X)
    y_pred = model.predict(X)
    customer_purchase = y_pred == 1

    result = {
        "customer_purchase?": bool(customer_purchase),
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
