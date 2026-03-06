from flask import Flask, request, jsonify
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from score import score

# ignore sklearn version warning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)

# load trained model
model = joblib.load("svm_model.pkl")


@app.route("/score", methods=["POST"])
def score_endpoint():

    data = request.get_json()
    text = data["text"]

    prediction, propensity = score(text, model, 0.5)

    return jsonify({
        "prediction": prediction,
        "propensity": propensity
    })


if __name__ == "__main__":
    app.run(port=5000)