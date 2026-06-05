import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# MUST be the first Streamlit command
st.set_page_config(page_title="Next Word Prediction", layout="centered")

# ------------------------------
# Load saved files
# ------------------------------
@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_resources()

# ------------------------------
# Prediction function
# ------------------------------
def predict_next_word(text: str) -> str:
    text = text.strip().lower()
    if not text:
        return ""

    sequence = tokenizer.texts_to_sequences([text])[0]

    if len(sequence) == 0:
        return ""

    sequence = pad_sequences([sequence], maxlen=max_len - 1, padding="pre")

    preds = model.predict(sequence, verbose=0)
    predicted_index = int(np.argmax(preds))

    # Most reliable way to map index -> word
    predicted_word = tokenizer.index_word.get(predicted_index, "")

    return predicted_word


# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🧠 Next Word Prediction (LSTM)")
st.write("Enter a sentence and the model will predict the **next word**.")

user_input = st.text_input(
    "✍️ Enter text:",
    placeholder="Type a sentence here..."
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

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("LSTM-based Next Word Prediction using Streamlit")