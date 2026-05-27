# Machine Learning Practice Projects

This repository contains hands-on machine learning practice work in Python. It includes exploratory data analysis, preprocessing, feature engineering, supervised learning models, ensemble learning, hyperparameter tuning, and a small Streamlit app that uses saved model artifacts for heart disease prediction.

## Repository Structure

```text
.
|-- README.md
|-- ensamble_learning/
|   `-- implementation.ipynb
|-- model_tuning/
|   `-- 01.ipynb
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
`-- project_01/
    |-- practice_01/
    |   |-- 01.ipynb
    |   `-- insurance.csv
    `-- practice_02/
        |-- 02.ipynb
        `-- heart.csv
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

### 5. Model Tuning with Iris

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

### 6. Ensemble Learning with Iris

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
pip install numpy pandas matplotlib seaborn scipy scikit-learn streamlit joblib xgboost notebook
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
- Model serialization with Joblib
- Basic Streamlit app development

## Possible Next Improvements

- Add a `requirements.txt` file.
- Organize notebook names with descriptive titles.
- Move reusable preprocessing steps into Python scripts.
- Add validation and clearer probability output to the Streamlit app.
- Add cross-validation results for the heart disease app model selection.
- Add a short note to each notebook explaining the dataset source and final conclusion.
