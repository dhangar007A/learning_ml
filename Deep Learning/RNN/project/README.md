# Next Word Prediction with LSTM

This project is a small Streamlit application that predicts the next word for a user-entered sentence. It uses a trained LSTM model built with TensorFlow/Keras.

## What this project contains

- `app/streamlit_app.py`: the Streamlit app you run locally
- `models/lstm_model.h5`: the trained LSTM model
- `artifacts/tokenizer.pkl`: tokenizer used during training
- `artifacts/max_len.pkl`: saved maximum sequence length
- `data/quote_dataset.csv`: dataset used for training and experimentation
- `notebooks/training_workflow.ipynb`: notebook for model training and testing

## Updated folder structure

```text
project/
|-- README.md
|-- app/
|   `-- streamlit_app.py
|-- artifacts/
|   |-- max_len.pkl
|   `-- tokenizer.pkl
|-- data/
|   `-- quote_dataset.csv
|-- models/
|   `-- lstm_model.h5
`-- notebooks/
    `-- training_workflow.ipynb
```

## Why the structure was cleaned up

The old layout kept code, notebook files, model files, and dataset files in one place. That makes the project harder to understand later. The updated layout separates:

- application code
- trained model files
- preprocessing artifacts
- training data
- notebook-based experiments

## Renamed files

- `app.py` -> `app/streamlit_app.py`
- `01.ipynb` -> `notebooks/training_workflow.ipynb`
- `qoute_dataset.csv` -> `data/quote_dataset.csv`

These names are more descriptive and fix the spelling issue in the dataset filename.

## How to run the app

From the `project` folder, run:

```powershell
streamlit run app/streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal.

## How the app works

1. The app loads the trained model, tokenizer, and saved sequence length.
2. The input text is converted into tokens.
3. The sequence is padded to the required input length.
4. The model predicts the most likely next word.
5. The predicted word is shown in the UI.

## Main dependencies

- `streamlit`
- `tensorflow`
- `numpy`

If needed, install them with:

```powershell
pip install streamlit tensorflow numpy
```

## Notes

- The app now loads files using paths relative to the project folder, so it is more reliable after the folder cleanup.
- The notebook has been updated to read the renamed dataset and save the model to the new `models/` folder.
- If you retrain the model and also want to refresh `tokenizer.pkl` or `max_len.pkl`, save them again into the `artifacts/` folder.
