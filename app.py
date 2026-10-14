import joblib
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

bundle = joblib.load("model.joblib")
vectorizer = bundle["vectorizer"]
model = bundle["model"]


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/detect")
def detect():
    return render_template("detect.html")


@app.route("/features")
def features():
    return render_template("features.html")


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/faq")
def faq():
    return render_template("faq.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please enter some news text."}), 400

    features_vec = vectorizer.transform([text])
    prediction = model.predict(features_vec)[0]
    score = float(model.decision_function(features_vec)[0])
    return jsonify({"prediction": prediction, "score": round(score, 3)})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)