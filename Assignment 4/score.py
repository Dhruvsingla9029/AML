import numpy as np

def score(text: str, model, threshold: float):

    # model expects list input
    decision = model.decision_function([text])[0]
    prob = 1 / (1 + np.exp(-decision))

    if prob >= threshold:
        prediction = 1
    else:
        prediction = 0

    return prediction, prob