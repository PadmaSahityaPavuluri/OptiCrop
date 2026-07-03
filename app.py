from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [
        float(request.form["N"]),
        float(request.form["P"]),
        float(request.form["K"]),
        float(request.form["temperature"]),
        float(request.form["humidity"]),
        float(request.form["ph"]),
        float(request.form["rainfall"])
    ]

    prediction = model.predict([values])[0]

    return render_template("index.html", prediction_text=f"Recommended Crop: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)