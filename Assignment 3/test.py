import joblib
from score import score
from app import app
import pytest
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

model = joblib.load("svm_model.pkl")


def test_score():

    text = "Hello how are you"

    pred, prob = score(text, model, 0.5)

    assert pred is not None
    assert prob is not None

    assert isinstance(pred, int)
    assert isinstance(prob, float)

    assert pred in [0, 1]
    assert 0 <= prob <= 1


def test_threshold_zero():

    pred, prob = score("hello", model, 0)

    assert pred == 1


def test_threshold_one():

    pred, prob = score("hello", model, 1)

    assert pred == 0


def test_spam():

    text = "Congratulations! You won a free lottery!"

    pred, prob = score(text, model, 0.5)

    assert pred in [0, 1]
    assert 0 <= prob <= 1


def test_not_spam():

    text = "Let's meet tomorrow at the office."

    pred, prob = score(text, model, 0.5)

    assert pred in [0, 1]
    assert 0 <= prob <= 1


@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


def test_flask_endpoint(client):

    response = client.post(
        "/score",
        json={"text": "Win free money now"}
    )

    data = response.get_json()

    assert response.status_code == 200
    assert "prediction" in data
    assert "propensity" in data

    assert data["prediction"] in [0, 1]
    assert 0 <= data["propensity"] <= 1