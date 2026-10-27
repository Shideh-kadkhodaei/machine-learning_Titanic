# Titanic Survival Prediction

## 📌 Project Overview

This project uses machine learning classification algorithms to predict whether a passenger survived the Titanic disaster based on passenger information such as age, gender, passenger class, family relationships, fare, and port of embarkation.

The project covers exploratory data analysis (EDA), missing-value handling, feature engineering, categorical encoding, model comparison, cross-validation, and hyperparameter tuning.

Six classification approaches are evaluated, and GridSearchCV is used to optimize a Random Forest model for the final predictions.

## 🎯 Objectives

- Explore the Titanic passenger dataset and identify survival patterns.
- Analyze survival rates across passenger demographics and ticket classes.
- Handle missing values and engineer meaningful features.
- Encode categorical variables and scale features where appropriate.
- Compare multiple classification algorithms.
- Evaluate model performance using accuracy and confusion matrices.
- Apply cross-validation and hyperparameter tuning.
- Generate survival predictions for the test dataset.

## 📂 Dataset

The project uses three CSV files:

| File | Description |
|---|---|
| `train.csv` | Training data containing passenger features and survival labels |
| `test.csv` | Passenger data used for final predictions |
| `sampleSubmission.csv` | Sample submission template |

The training dataset contains **891 passengers and 12 columns**, while the test dataset contains **418 passengers and 11 columns**.

### Key Features

| Feature | Description |
|---|---|
| `PassengerId` | Unique passenger identifier |
| `Survived` | Target variable: 0 = did not survive, 1 = survived |
| `Pclass` | Passenger ticket class |
| `Name` | Passenger name |
| `Sex` | Passenger gender |
| `Age` | Passenger age |
| `SibSp` | Number of siblings and spouses aboard |
| `Parch` | Number of parents and children aboard |
| `Ticket` | Ticket identifier |
| `Fare` | Passenger fare |
| `Cabin` | Cabin information |
| `Embarked` | Port of embarkation |

## 🛠️ Technologies Used

- **Python** — programming language
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical operations
- **Matplotlib** — data visualization
- **Seaborn** — statistical visualization
- **Scikit-learn** — preprocessing, classification, model evaluation, and hyperparameter tuning
- **XGBoost** — gradient boosting classification
- **Jupyter Notebook** — interactive development environment

## 🔍 Project Workflow

### 1. Exploratory Data Analysis (EDA)

The dataset is explored to understand its structure, distributions, and relationships.

The analysis includes:

- Dataset information and descriptive statistics.
- Missing-value analysis.
- Survival distribution.
- Survival rates by gender, passenger class, and embarkation port.
- Correlation heatmap for numerical features.
- Pairplots to explore relationships between passenger attributes and survival.

The training dataset shows an overall survival rate of approximately **38.38%**.

The exploratory analysis also reveals differences in survival rates across gender and passenger class.

### 2. Data Preprocessing

Several preprocessing steps are applied to prepare the data for classification.

**Feature engineering**

- Extract passenger titles from the `Name` column.
- Group uncommon titles into a `Rare` category.
- Standardize title variations such as `Mlle`, `Ms`, and `Mme`.
- Create a binary `HasCabin` feature indicating whether cabin information is available.

**Missing-value handling**

- Fill missing `Embarked` values using the most frequent training value.
- Fill missing `Fare` values in the test dataset using the training median.
- Impute missing ages using medians grouped by gender and passenger class.

**Feature selection and encoding**

- Remove `Name`, `Ticket`, and `Cabin` after processing.
- Exclude `PassengerId` from model features.
- Separate the target variable, `Survived`, from the training features.
- Apply one-hot encoding to `Sex`, `Embarked`, and `Title`.
- Align training and test feature columns to maintain a consistent input structure.

### 3. Data Visualization

The project includes visualizations to investigate:

- Survival rates by gender.
- Survival rates by passenger class.
- Age distributions by survival status.
- Relationships between age, fare, passenger class, and survival.
- Fare distributions across passenger classes.
- Age distributions segmented by gender and survival.

These visualizations support the interpretation of passenger survival patterns.

### 4. Classification Models

Six classification algorithms are trained and evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Support Vector Machine (SVM)
6. XGBoost Classifier

The data is divided into training and validation sets using an 80/20 split, with stratification to preserve the target class proportions.

StandardScaler is used for scale-sensitive models, including Logistic Regression, KNN, and SVM. Tree-based models are trained on the encoded features without scaling.

## 📊 Model Performance

The following accuracy scores were obtained on the validation set:

| Model | Validation Accuracy |
|---|---:|
| Logistic Regression | 83.80% |
| K-Nearest Neighbors | 82.12% |
| Decision Tree | 81.01% |
| Random Forest | 80.45% |
| Support Vector Machine (SVM) | 82.68% |
| XGBoost | 80.45% |

Logistic Regression achieves the highest validation accuracy among these six initial model evaluations, at approximately **83.80%**.

Confusion matrices are also calculated to examine correct and incorrect survival predictions for each model.

### 5. Cross-Validation and Hyperparameter Tuning

Random Forest is further evaluated using 5-fold cross-validation.

The initial Random Forest model achieves a mean cross-validation accuracy of approximately **82.27%**.

GridSearchCV is then used to search the following hyperparameters:

| Parameter | Values Tested |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | 4, 6, 8 |

The best parameters found are:

- `n_estimators = 200`
- `max_depth = 6`

The best cross-validation accuracy is approximately **82.27%**.

### 6. Final Predictions

The optimized Random Forest model is fitted on the full encoded training dataset and used to predict survival for the 418 passengers in the test dataset.

The predictions are combined with their corresponding passenger IDs and exported to:

`submission.csv`

The output contains two columns:

| Column | Description |
|---|---|
| `PassengerId` | Passenger identifier from the test dataset |
| `Survived` | Predicted survival class (0 or 1) |

## 🚀 Getting Started

### Prerequisites

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

### Run the Project

1. Clone or download the repository.
2. Place `train.csv`, `test.csv`, and `sampleSubmission.csv` in the project directory.
3. Open the Jupyter Notebook.
4. Run the notebook cells in order.
5. Find the generated predictions in `submission.csv`.

Start Jupyter Notebook with:

```bash
jupyter notebook
```

## 📁 Project Structure

```text
Titanic-Survival-Prediction/
│
├── train.csv
├── test.csv
├── sampleSubmission.csv
├── Titanic Survival Prediction.ipynb
├── submission.csv
└── README.md
```

## 🔮 Future Improvements

- Compare models using precision, recall, F1-score, and ROC-AUC.
- Improve feature engineering with family size and passenger titles.
- Use pipelines to make preprocessing and model evaluation more consistent.
- Explore additional hyperparameter combinations.
- Evaluate models using repeated or stratified cross-validation.
- Compare the optimized Random Forest with Logistic Regression and other promising classifiers.
