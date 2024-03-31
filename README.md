# Classification of Grape Varieties in Wine

## Table of Contents

- [Introduction](#introduction)
- [Data Description](#data-description)
- [Descriptive Analysis](#descriptive-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Predictions](#predictions)
- [Conclusions](#conclusions)
- [Resources](#resources)
- [Folder Structure](#folder-structure)

## Introduction <a name="introduction"></a>

This project aims to develop a model to classify grape varieties in wine (A, B, or C) using machine learning techniques. The model is trained using a dataset from [Kaggle](https://www.kaggle.com/datasets/rajyellow46/wine-quality).

## Data Description <a name="data-description"></a>

The dataset contains 534 examples of wine, each with 13 features:

- Alcohol
- Malic acid
- Ash
- Alcalinity of ash
- Magnesium
- Total phenols
- Flavanoids
- Nonflavanoid phenols
- Proanthocyanins
- Color intensity
- Hue
- OD280/OD315 of diluted wines
- Proline

The target variable is the wine class (A, B, or C).

## Descriptive Analysis <a name="descriptive-analysis"></a>

Descriptive analysis of the data was performed to understand its distribution and correlations. Distribution plots, histograms, and correlation matrices were generated.

## Data Preprocessing <a name="data-preprocessing"></a>

Variables were normalized using standard scaling. Principal Component Analysis (PCA) was applied to reduce dimensionality and eliminate multicollinearity.

## Model Training <a name="model-training"></a>

Three machine learning models were trained:

- XGBoost
- SVM
- Random Forest

## Model Evaluation <a name="model-evaluation"></a>

The models were evaluated using the following metrics:

- F1-score
- AUC-ROC multiclass
- Precision
- Recall

## Hyperparameter Optimization <a name="hyperparameter-optimization"></a>

Model hyperparameters were optimized using the Optuna library.

## Predictions <a name="predictions"></a>

Predictions of grape variety were generated for new wine examples.
```
new_data = [
    [13.72, 1.43, 2.5, 16.7, 108, 3.4, 3.67, 0.19, 2.04, 6.8, 0.89, 2.87, 1285],
    [12.37, 0.94, 1.36, 10.6, 88, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520]
]
```

## Conclusions <a name="conclusions"></a>

The three models demonstrate a high F1-score, suggesting potential overfitting due to the limited dataset size. This overfitting can be addressed by increasing the amount of data to enable better generalization of the models to new samples. Additionally, regularization techniques and the exploration of simpler models can help mitigate the risk of overfitting by preventing the models from learning noise and focusing on capturing underlying patterns more effectively.

The model with the best performance was XGBoost, achieving an F1-score of 0.98 and an AUC-ROC multiclass of 0.99. 

## Resources <a name="resources"></a>

- [Kaggle Dataset](https://www.kaggle.com/datasets/rajyellow46/wine-quality)
- [Optuna](https://optuna.org/)
- [MLflow](https://mlflow.org/)

## Folder Structure <a name="folder-structure"></a>
The project directory structure is organized as follows:
```
├── README.md
├── data
│ └── wine.csv
├── notebooks
│ ├── wine_classification.ipynb
├── models
│ ├── xgboost
│ ├── svm
│ └── random_forest
└── requirements.txt
```
## Code

The code for this project is in Jupyter notebooks:

- `wine_classification.ipynb`: Descriptive analysis of the data. Model training, evaluation, and optimization.
