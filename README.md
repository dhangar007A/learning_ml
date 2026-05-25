# Machine Learning Practice Projects

This repository contains my machine learning practice notebooks. The work so far focuses on exploratory data analysis, data cleaning, preprocessing, feature engineering, and training beginner-friendly supervised learning models with Python and scikit-learn.

## Repository Structure

```text
.
|-- project_01/
|   |-- practice_01/
|   |   |-- 01.ipynb
|   |   `-- insurance.csv
|   `-- practice_02/
|       |-- 02.ipynb
|       `-- heart.csv
`-- practice_03/
    `-- logistic_regression/
        `-- 01.ipynb
```

## Projects Completed So Far

### 1. Medical Insurance Charges Prediction

**Notebook:** `project_01/practice_01/01.ipynb`  
**Dataset:** `project_01/practice_01/insurance.csv`

This notebook predicts medical insurance charges using regression.

Work implemented:

- Loaded and explored the insurance dataset.
- Performed EDA using summary statistics, histograms, count plots, box plots, and correlation heatmaps.
- Checked missing values and duplicate rows.
- Encoded categorical features:
  - `sex` converted to `is_female`
  - `smoker` converted to `is_smoker`
  - `region` converted using one-hot encoding
- Created BMI category features using feature engineering.
- Scaled numerical columns with `StandardScaler`.
- Used Pearson correlation and chi-square tests for feature selection.
- Trained a **Linear Regression** model to predict `charges`.

Model implemented:

- `LinearRegression`

Current saved results:

- R2 score: `0.8039`
- Adjusted R2 score: `0.7994`

### 2. Heart Disease Dataset Analysis

**Notebook:** `project_01/practice_02/02.ipynb`  
**Dataset:** `project_01/practice_02/heart.csv`

This notebook currently focuses on EDA and basic preprocessing for a heart disease classification dataset.

Work implemented:

- Loaded and explored the heart disease dataset.
- Checked dataset shape, descriptive statistics, and duplicate rows.
- Visualized the target column `HeartDisease`.
- Plotted distributions for numerical features:
  - `Age`
  - `RestingBP`
  - `Cholesterol`
  - `MaxHR`
- Replaced invalid zero values in `Cholesterol` and `RestingBP` with non-zero mean values.
- Visualized categorical columns such as `Sex`.

Current status:

- EDA and data cleaning are implemented.
- No machine learning model has been trained in this notebook yet.

### 3. Titanic Survival Classification

**Notebook:** `practice_03/logistic_regression/01.ipynb`  
**Dataset:** Titanic dataset loaded from seaborn

This notebook predicts passenger survival on the Titanic using multiple classification models.

Work implemented:

- Loaded the Titanic dataset using `seaborn`.
- Dropped less useful or duplicate columns:
  - `deck`
  - `embark_town`
  - `alive`
  - `class`
  - `who`
  - `adult_male`
- Filled missing `age` values with the mean age.
- Removed rows with missing `embarked` values.
- Encoded categorical columns using `LabelEncoder`.
- Converted the cleaned dataset to integer values.
- Split data into training and testing sets with `train_test_split`.
- Scaled features with `StandardScaler` for models that benefit from scaling.
- Evaluated models using accuracy, confusion matrix, and classification report.

Models implemented:

- `LogisticRegression`
- `KNeighborsClassifier`
- `GaussianNB`
- `DecisionTreeClassifier`
- `SVC` with RBF kernel

Current saved accuracy scores:

| Model | Accuracy |
| --- | ---: |
| Logistic Regression | `0.8033` |
| K-Nearest Neighbors | `0.7921` |
| Gaussian Naive Bayes | `0.7752` |
| Decision Tree Classifier | `0.8033` |
| Support Vector Classifier | `0.8146` |

Best saved model so far:

- **Support Vector Classifier**, with accuracy `0.8146`

## Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- scikit-learn

## How to Run

1. Clone the repository.
2. Install the required Python libraries.
3. Open the notebooks in Jupyter Notebook, JupyterLab, or VS Code.
4. Run the cells from top to bottom.

Example installation command:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn notebook
```

## Current Learning Progress

So far, this repository includes:

- Regression modeling with Linear Regression.
- Classification modeling with Logistic Regression, KNN, Naive Bayes, Decision Tree, and SVM.
- EDA and preprocessing on insurance, heart disease, and Titanic datasets.
- Feature encoding, feature scaling, feature engineering, and basic feature selection.

Future improvements can include training classification models on the heart disease dataset, adding cross-validation, comparing more metrics, and saving final models.
