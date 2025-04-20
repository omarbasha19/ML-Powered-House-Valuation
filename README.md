# ML-Powered House Valuation

## Overview
This project presents a machine learning approach to estimating house prices using structured data. Multiple regression models were trained, evaluated, and compared to identify the most accurate solution for real estate valuation.

The pipeline includes feature engineering, model tuning, and error analysis to enhance prediction quality. The dataset used is based on housing features such as location attributes, number of rooms, property size, and more.

## Project Highlights
- Built and compared multiple regression models:
  - Decision Tree
  - Random Forest
  - XGBoost
  - Tuned XGBoost (GridSearchCV)
  - LightGBM
- Applied log transformation, outlier handling, and skewness correction
- Performed Z-score normalization for better convergence
- Introduced feature: `Price per Room` to boost model performance
- Evaluated models using RMSE, R², MAE, and residual plots
- Visualized errors and feature importances using SHAP and heatmaps

## Dataset
- Based on the Boston Housing dataset
- Contains features like:
  - CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT
  - Target: MEDV (Median house price in $1000s)

Dataset Source: [Kaggle – Boston Housing](https://www.kaggle.com/c/boston-housing)

## Workflow
1. Load and explore the dataset
2. Preprocess: encode, transform, handle skewness/outliers
3. Train/test split and Z-score scaling
4. Model training, evaluation, and tuning
5. Residual and feature importance visualization
6. Deploy interactive prediction function for manual input

## Technologies
- Python
- Jupyter Notebook
- pandas, numpy
- scikit-learn
- xgboost, lightgbm
- seaborn, matplotlib, SHAP

## Results Summary

| Model               | R² Score | RMSE   | MAE    |
|---------------------|----------|--------|--------|
| Decision Tree       | 0.84     | 0.18   | 0.13   |
| Random Forest       | 0.91     | 0.12   | 0.09   |
| XGBoost             | 0.92     | 0.10   | 0.08   |
| Tuned XGBoost       | 0.94     | 0.09   | 0.07   |
| LightGBM            | 0.95     | 0.08   | 0.06   |
| XGBoost + Feature   | 0.94     | 0.08   | 0.06   |

## Future Work
- Deploy the model via API or Streamlit app
- Extend dataset with location and economic indicators
- Experiment with neural networks and ensemble stacking

## Conclusion
By applying advanced machine learning techniques and thoughtful feature engineering, this project demonstrates a powerful pipeline for predicting housing prices with high accuracy and interpretability.
