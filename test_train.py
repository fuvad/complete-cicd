"""This is test_train"""
import joblib
from train import model

def test_model_saved():
    """This is testing model"""
    loaded_model = joblib.load("model.pkl")
    assert loaded_model is not None

def test_model_predicts():
    """This is testing the prediction"""
    preds = model.predict([[5.1, 3.5, 1.4, 0.2]])
    assert len(preds) == 1
