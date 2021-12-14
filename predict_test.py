import requests


url = "http://localhost:9696/predict"

customer = {
    "Administrative": 3,
    "Administrative_Duration": 157.4,
    "BounceRates": 0.036363636,
    "Browser": 2,
    "ExitRates": 0.081818182,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "Month": "Jul",
    "OperatingSystems": 3,
    "PageValues": 0.0,
    "ProductRelated": 9,
    "ProductRelated_Duration": 128.5,
    "Region": 1,
    "SpecialDay": 0.0,
    "TrafficType": 3,
    "VisitorType": "Returning_Visitor",
    "Weekend": True,
    "kfold": 4,
}


response = requests.post(url, json=customer).json()
print(response)
