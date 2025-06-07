# Solar-Panel-Performance-Predictor

A machine learning solution for predicting solar panel efficiency and enabling predictive maintenance in photovoltaic (PV) systems. This project uses ensemble methods to forecast performance degradation and potential failures based on historical and real-time sensor data.

📋 Project Overview
As solar energy systems become increasingly popular in sustainable energy infrastructures, maintaining high performance and reducing downtime are essential. Traditional maintenance methods for photovoltaic (PV) systems are often reactive, leading to energy loss and increased costs. This project develops a Machine Learning model that predicts performance degradation and potential failures in solar panels using sensor data.

🎯 Objectives
-Predict solar panel efficiency using historical and real-time sensor data
-Enable predictive maintenance to reduce system downtime
-Optimize energy output through early failure detection
-Provide actionable insights for maintenance scheduling

📊 Dataset
The dataset consists of three files:

train.csv: 20,000 samples × 17 features (training data with target variable)
test.csv: 12,000 samples × 16 features (test data for predictions)
sample_submission.csv: 5 × 2 (submission format reference)

Key Features:

Categorical Features: string_id, error_code, installation_type
Numerical Features: Various sensor readings and operational parameters
Target Variable: efficiency (solar panel efficiency percentage)

🔧 Technical Implementation
Data Preprocessing

-Label Encoding: Categorical variables encoded using LabelEncoder
-Data Cleaning: Non-numeric values converted to numeric format
-Missing Value Imputation: Mean imputation for numerical features
-Feature Validation: Ensured consistent encoding across train/test sets

Machine Learning Models
1. Random Forest Regressor
pythonRandomForestRegressor(n_estimators=200, random_state=42)

Performance: 89.07% score
Advantages: Handles mixed data types well, provides feature importance

2. XGBoost Regressor
pythonXGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)

Performance: 89.13% score
Advantages: Excellent gradient boosting performance, handles complex patterns

3. Ensemble Method

Approach: Simple averaging of Random Forest and XGBoost predictions
Rationale: Combines strengths of both models to reduce overfitting

Evaluation Metric
Custom scoring function: 100 * (1 - sqrt(RMSE))

Higher scores indicate better performance
Takes into account both accuracy and precision of predictions

🚀 Getting Started
Prerequisites
bashpip install pandas numpy scikit-learn xgboost
Installation

Clone the repository
Install required dependencies
Ensure your dataset files are in the same directory as the notebook

Usage

-Load Data: Place train.csv, test.csv, and sample_submission.csv in the working directory
-Run Preprocessing: Execute data cleaning and encoding steps
-Train Models: Train both Random Forest and XGBoost models
-Generate Predictions: Create ensemble predictions for test set
-Submit Results: Output saved as final_submission.csv

📈 Model Performance
Model          Validation Score
Random Forest      89.07%
XGBoost            89.13%

🛠️ Key Features

-Robust Preprocessing: Handles mixed data types and missing values
-Ensemble Approach: Combines multiple algorithms for better performance
-Production Ready: Clean, well-documented code suitable for deployment
-Scalable: Can be extended with additional models or features

🔮 Future Enhancements

-Feature Engineering: Create time-based features, degradation rates
-Advanced Ensemble: Implement weighted averaging or stacking
-Deep Learning: Explore LSTM networks for temporal patterns
-Cross-Validation: Implement time-series cross-validation
-Feature Selection: Use feature importance for dimensionality reduction
