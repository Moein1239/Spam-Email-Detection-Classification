import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def save_model(model, vectorizer, model_name="spam_model.pkl"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, model_name))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))


def load_model(model_name="spam_model.pkl"):
    model_path = os.path.join(MODEL_DIR, model_name)
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer
