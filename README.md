# Machine Learning Practice Projects

This repository contains hands-on machine learning practice work in Python. It includes exploratory data analysis, preprocessing, feature engineering, supervised learning models, NLP, unsupervised learning, ensemble learning, hyperparameter tuning, and Streamlit apps that use saved model artifacts for prediction.

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- ensamble_learning/
|   `-- implementation.ipynb
|-- model_tuning/
|   `-- 01.ipynb
|-- nlp/
|   |-- 01.ipynb
|   |-- app.py
|   |-- bag_of_words.ipynb
|   |-- data/
|   |   `-- train.txt
|   |-- models/
|   |   |-- emotion_mapping.pkl
|   |   |-- emotion_model.pkl
|   |   `-- tfidf_vectorizer.pkl
|   `-- tf_idf.ipynb
|-- practice_03/
|   `-- logistic_regression/
|       `-- 01.ipynb
|-- project/
|   |-- app.py
|   |-- columns.pkl
|   |-- heart.csv
|   |-- knn_model.pkl
|   |-- project_01.ipynb
|   `-- scaler.pkl
|-- project_01/
|   |-- practice_01/
|   |   |-- 01.ipynb
|   |   `-- insurance.csv
|   `-- practice_02/
|       |-- 02.ipynb
|       `-- heart.csv
`-- unsupervised_learning/
    |-- PCA.ipynb
    `-- k_means_implementation.ipynb
```

## Projects and Notebooks

### 1. Medical Insurance Charges Prediction

**Notebook:** `project_01/practice_01/01.ipynb`  
**Dataset:** `project_01/practice_01/insurance.csv`

Regression practice for predicting medical insurance charges.

Work covered:

- EDA with summary statistics, histograms, count plots, box plots, and correlation heatmaps.
- Missing-value and duplicate-row checks.
- Categorical encoding for `sex`, `smoker`, and `region`.
- BMI category feature engineering.
- Feature scaling with `StandardScaler`.
- Feature selection using Pearson correlation and chi-square testing.
- Linear Regression model training and evaluation.

Saved notebook result:

| Model | Metric | Score |
| --- | --- | ---: |
| Linear Regression | R2 | `0.8039` |
| Linear Regression | Adjusted R2 | `0.7994` |

### 2. Heart Disease EDA and Cleaning

**Notebook:** `project_01/practice_02/02.ipynb`  
**Dataset:** `project_01/practice_02/heart.csv`

Exploratory analysis and preprocessing practice for a heart disease classification dataset.

Work covered:

- Dataset inspection, descriptive statistics, and duplicate checks.
- Target distribution visualization for `HeartDisease`.
- Numerical feature distribution plots for `Age`, `RestingBP`, `Cholesterol`, and `MaxHR`.
- Replacement of invalid zero values in `Cholesterol` and `RestingBP`.
- Initial categorical feature exploration.

Current status: EDA and cleaning are implemented; no model is trained in this notebook yet.

### 3. Titanic Survival Classification

**Notebook:** `practice_03/logistic_regression/01.ipynb`  
**Dataset:** Titanic dataset loaded from seaborn

Classification practice using the Titanic dataset.

Work covered:

- Dataset loading with seaborn.
- Dropping less useful or duplicate columns such as `deck`, `embark_town`, `alive`, `class`, `who`, and `adult_male`.
- Missing-value handling for `age` and `embarked`.
- Label encoding for categorical columns.
- Train/test splitting and feature scaling.
- Evaluation with accuracy score, confusion matrix, and classification report.
- Cross-validation practice with SVC.

Saved notebook results:

| Model | Accuracy |
| --- | ---: |
| Logistic Regression | `0.8033` |
| K-Nearest Neighbors | `0.7921` |
| Gaussian Naive Bayes | `0.7752` |
| Decision Tree Classifier | `0.8033` |
| Support Vector Classifier | `0.8146` |

Best saved result: Support Vector Classifier with accuracy `0.8146`.

### 4. Heart Disease Prediction App

**Notebook:** `project/project_01.ipynb`  
**App:** `project/app.py`  
**Dataset:** `project/heart.csv`

End-to-end heart disease classification practice with saved artifacts and a Streamlit interface.

Work covered:

- EDA and preprocessing for the heart disease dataset.
- Invalid medical value handling.
- One-hot encoding for categorical variables.
- Train/test splitting and feature scaling.
- Model comparison with accuracy and F1 score.
- Saving the selected KNN model, scaler, and expected feature columns with Joblib.
- Streamlit app for collecting user inputs and predicting heart disease likelihood.

Models compared:

- Logistic Regression
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Decision Tree Classifier
- Support Vector Classifier

Saved notebook results:

| Model | Accuracy |
| --- | ---: |
| Logistic Regression | `0.8696` |
| K-Nearest Neighbors | `0.8641` |
| Gaussian Naive Bayes | `0.8478` |
| Decision Tree Classifier | `0.8098` |
| Support Vector Classifier | `0.8478` |

Saved artifacts:

- `project/knn_model.pkl`
- `project/scaler.pkl`
- `project/columns.pkl`

### 5. NLP Emotion Detection App

**Notebook:** `nlp/01.ipynb`  
**App:** `nlp/app.py`  
**Dataset:** `nlp/data/train.txt`

Text classification project for predicting emotions from user-entered sentences.

Work covered:

- Loading semicolon-separated text and emotion labels.
- Text preprocessing with lowercasing, punctuation removal, number removal, emoji/non-ASCII removal, tokenization, and stopword removal.
- Label mapping for emotion names.
- Bag-of-Words and TF-IDF feature extraction.
- Naive Bayes and Logistic Regression model comparison.
- Saving the best model, TF-IDF vectorizer, and emotion mapping with Joblib.
- Streamlit app for entering text and predicting the emotion.

Saved notebook results:

| Model | Features | Accuracy |
| --- | --- | ---: |
| Multinomial Naive Bayes | Bag-of-Words | `0.7678` |
| Multinomial Naive Bayes | TF-IDF | `0.6609` |
| Logistic Regression | TF-IDF | `0.8616` |

Saved artifacts:

- `nlp/models/emotion_model.pkl`
- `nlp/models/tfidf_vectorizer.pkl`
- `nlp/models/emotion_mapping.pkl`

### 6. Model Tuning with Iris

**Notebook:** `model_tuning/01.ipynb`  
**Dataset:** Iris dataset loaded from seaborn

Hyperparameter tuning practice using KNN and SVC.

Work covered:

- Iris dataset loading and train/test splitting.
- KNN model training.
- SVC model training with custom `C`, `kernel`, and `gamma` values.
- Grid search with `GridSearchCV`.
- Random search with `RandomizedSearchCV`.
- Comparison of parameter combinations using cross-validation scores.

Saved notebook highlights:

- KNN test score: `1.0`
- SVC test score: `1.0`
- Best displayed GridSearchCV mean test score: `0.98`
- Best displayed RandomizedSearchCV mean test score: `0.98`

### 7. Ensemble Learning with Iris

**Notebook:** `ensamble_learning/implementation.ipynb`  
**Dataset:** Iris dataset loaded from seaborn

Ensemble learning practice using stacking, bagging, and boosting methods.

Work covered:

- Iris dataset loading and label encoding.
- Train/test splitting.
- Stacking classifier with Decision Tree, SVC, and Logistic Regression base learners.
- Random Forest bagging model.
- AdaBoost model.
- Gradient Boosting model.
- XGBoost model.
- Evaluation with accuracy score and classification report.

Saved notebook results:

| Model | Accuracy |
| --- | ---: |
| Stacking Classifier | `1.0` |
| Random Forest Classifier | `1.0` |
| AdaBoost Classifier | `0.9333` |
| Gradient Boosting Classifier | `1.0` |
| XGBoost Classifier | `1.0` |

### 8. Unsupervised Learning

**Notebooks:**  
`unsupervised_learning/k_means_implementation.ipynb`  
`unsupervised_learning/PCA.ipynb`

Clustering and dimensionality reduction practice using generated datasets.

Work covered:

- K-Means clustering on synthetic blob data.
- Elbow method using inertia values to choose the number of clusters.
- Cluster visualization with seaborn scatter plots.
- Two-moons dataset generation with `make_moons`.
- Comparison of K-Means and DBSCAN clustering behavior on non-linear data.
- DBSCAN clustering with `eps` and `min_samples`.
- PCA dimensionality reduction from 5 features to 2 principal components.
- PCA visualization using `PC1` and `PC2`.

## Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- scikit-learn
- Streamlit
- Joblib
- XGBoost
- Jupyter Notebook

## How to Run

Install the common dependencies:

```bash
pip install -r requirements.txt
```

Open the notebooks with Jupyter:

```bash
jupyter notebook
```

Run the Streamlit heart disease app:

```bash
cd project
streamlit run app.py
```

The Streamlit app expects these files to be present in the `project` folder:

- `knn_model.pkl`
- `scaler.pkl`
- `columns.pkl`

Run the Streamlit emotion detection app:

```bash
streamlit run nlp/app.py
```

The emotion detection app expects these files to be present:

- `nlp/models/emotion_model.pkl`
- `nlp/models/tfidf_vectorizer.pkl`
- `nlp/models/emotion_mapping.pkl`

## Learning Topics Covered

- Exploratory data analysis
- Data cleaning and preprocessing
- Feature engineering
- Label encoding and one-hot encoding
- Feature scaling
- Train/test splitting
- Regression modeling
- Classification modeling
- Model evaluation with accuracy, F1 score, R2 score, confusion matrix, and classification reports
- Cross-validation
- Hyperparameter tuning with grid search and randomized search
- Ensemble learning with stacking, bagging, and boosting
- Unsupervised learning with K-Means, DBSCAN, and PCA
- Model serialization with Joblib
- Basic Streamlit app development

## Possible Next Improvements

- Organize notebook names with descriptive titles.
- Move reusable preprocessing steps into Python scripts.
- Move reusable NLP preprocessing into a shared Python module used by both the notebook and app.
- Add cross-validation results for the heart disease app model selection.
- Add a short note to each notebook explaining the dataset source and final conclusion.
