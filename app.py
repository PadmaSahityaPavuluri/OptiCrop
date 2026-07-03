from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/findyourcrop')
def findyourcrop():
    return render_template("findyourcrop.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Create input array
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Scale input
        features = scaler.transform(features)

        # Predict crop
        prediction = model.predict(features)

        # Debug output (shown in VS Code terminal)
        print("Prediction:", prediction)
        print("Prediction type:", type(prediction[0]))

        # If prediction is already a crop name
        crop = str(prediction[0])

        return render_template(
            "findyourcrop.html",
            prediction_text=f"Recommended Crop: {crop}"
        )

    except Exception as e:
        return render_template(
            "findyourcrop.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)