from pathlib import Path
import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Next Word Prediction", layout="centered")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_model.h5"
TOKENIZER_PATH = PROJECT_ROOT / "artifacts" / "tokenizer.pkl"
MAX_LEN_PATH = PROJECT_ROOT / "artifacts" / "max_len.pkl"


@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)

    with TOKENIZER_PATH.open("rb") as file:
        tokenizer = pickle.load(file)

    with MAX_LEN_PATH.open("rb") as file:
        max_len = pickle.load(file)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_resources()


def predict_next_word(text: str) -> str:
    cleaned_text = text.strip().lower()
    if not cleaned_text:
        return ""

    sequence = tokenizer.texts_to_sequences([cleaned_text])[0]
    if not sequence:
        return ""

    padded_sequence = pad_sequences([sequence], maxlen=max_len - 1, padding="pre")
    predictions = model.predict(padded_sequence, verbose=0)
    predicted_index = int(np.argmax(predictions))

    return tokenizer.index_word.get(predicted_index, "")


st.title("Next Word Prediction (LSTM)")
st.write("Enter a sentence and the model will predict the **next word**.")

user_input = st.text_input(
    "Enter text:",
    placeholder="Type a sentence here...",
)

if st.button("Predict Next Word"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        next_word = predict_next_word(user_input)
        if next_word:
            st.success(f"**Predicted Next Word:** {next_word}")
        else:
            st.warning("Could not predict a next word for the given input.")

st.markdown("---")
st.caption("LSTM-based Next Word Prediction using Streamlit")
