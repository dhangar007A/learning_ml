import streamlit as st
import joblib
import string
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"


st.set_page_config(page_title="Emotion Detection")

model = joblib.load(MODEL_DIR / "emotion_model.pkl")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
emotion_mapping = joblib.load(MODEL_DIR / "emotion_mapping.pkl")

reverse_mapping = {value: key for key, value in emotion_mapping.items()}


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = "".join(char for char in text if not char.isdigit())
    text = "".join(char for char in text if char.isascii())
    return text


st.title("Emotion Detection App")
st.write("Type a sentence and the model will predict the emotion behind it.")

user_input = st.text_area("Enter your text here")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        emotion = reverse_mapping[prediction]

        st.success(f"Predicted Emotion: {emotion}")

        if hasattr(model, "predict_proba"):
            prediction_index = list(model.classes_).index(prediction)
            confidence = model.predict_proba(vectorized_text)[0][prediction_index]
            st.info(f"Confidence: {confidence:.2%}")
