
# Credit Risk Analysis

![Finance](https://img.shields.io/badge/Domain-Finance-blue)
![ML](https://img.shields.io/badge/ML-Binary%20Classification-purple)
![Python](https://img.shields.io/badge/Language-Python-green)
![Models](https://img.shields.io/badge/Algorithms-LogisticRegression_RandomForest_SVM_XGBoost_CatBoost-red)

## Overview

This project focuses on predicting credit risk using a real-world dataset consisting of 45,000 loan applications. The classification task aims to determine whether an applicant is likely to default on a loan (binary target: `loan_status`). The dataset is highly imbalanced (35,000 True vs. 10,000 False), so different sampling strategies were applied to improve predictive performance.

<div align="center">
  <img src="/documentation/F1-Score%20Comparison.png" alt="F1-Score Comparison" width="600"/>
</div>

## Purpose

The goal is to build reliable machine learning models for binary classification, compare their performance on various data balancing techniques, and determine which model performs best under different conditions.

## Dataset

The dataset contains 13 input variables and one target variable:

- **Numerical**: age, income, emp_exp, loan_amnt, loan_int_rate, loan_percent_income, cb_person_cred_hist_length, credit_score
- **Categorical**: sex, education, home_ownership, loan_intent, previous_loan_defaults_on_file
- **Target**: loan_status (0 = No Default, 1 = Default)

Three dataset versions were used:
1. Original (imbalanced)
2. Undersampled using `RandomUnderSampler`
3. Oversampled using `ADASYN`

## Algorithms Used

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- CatBoost

## Repository Structure

```
credit-risk-analysis/
├── data/
│   ├── [clean]full_loan_data.csv
│   ├── [raw]full_loan_data.csv
│   ├── test_set.csv
│   ├── train_set_imbalanced.csv
│   ├── train_set_oversampled.csv
│   ├── train_set_undersampled.csv
│   ├── sklearn_based_models_prediction_output.csv
│   └── xgboost_and_catboost_prediction_output.csv
├── Evaluation.ipynb
├── Exploratory Data Analysis.ipynb
├── Modeling - Sklearn Based Models.ipynb
├── Modeling - XGBoost and CatBoost.ipynb
├── LICENSE
└── README.md
```

## Workflow

1. **Data Preprocessing**: Handling missing values, encoding categorical features, and normalization using `RobustScaler`.
2. **Data Sampling**: Applied `RandomUnderSampler` and `ADASYN` to balance the dataset.
3. **Modeling**: Trained 5 classification models on all 3 dataset versions.
4. **Optimization**: Hyperparameter tuning using `Optuna`.
5. **Evaluation**: Compared models using Accuracy, Precision, Recall, F1 Score, and ROC AUC.

## Key Results

- **Best Model**: XGBoost on undersampled data  
  - F1 Score: **0.8492**
  - ROC AUC: **0.8970**
- CatBoost also showed strong performance on oversampled data.
- Sampling improved recall and model robustness significantly.

## Technical Details

- Created 3 datasets: imbalanced, undersampled, oversampled  
- 15 models trained using five ML algorithms  
- Tuned with Optuna, evaluated by F1 and AUC

## Usage

To explore or reproduce the analysis:

```bash
git clone https://github.com/yourusername/credit-risk-analysis.git
cd credit-risk-analysis
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Then run the notebooks:

```bash
jupyter notebook
```

## Results

All model results are saved in `/Evaluation.ipynb`. Visualizations, confusion matrices, and model comparations are included in the notebooks.

<div align="center">
  <img src="/documentation/Confusion%20Matrix%20Comparison.png" alt="F1-Score Comparison" width="600"/>
</div>

## Future Work

- Add SHAP or LIME for model explainability  
- Deploy the best model as an API or web app  
- Explore ensemble stacking or blending techniques

## License

This project is licensed under the terms included in the [LICENSE](LICENSE) file.

## Acknowledgments

- Open source libraries: scikit-learn, Optuna, XGBoost, CatBoost, imbalanced-learn
- Credit risk analytics community for inspiration and resources
