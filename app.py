import joblib
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

bundle = joblib.load("model.joblib")
vectorizer = bundle["vectorizer"]
model = bundle["model"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please enter some news text."}), 400

    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    score = float(model.decision_function(features)[0])
    return jsonify({"prediction": prediction, "score": round(score, 3)})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
