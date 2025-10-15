# alloy-property-predictor
Alloy Property Prediction
Project Overview

This project implements a Machine Learning system to predict metal alloy properties. It is designed to aid engineering students and support local manufacturing by enabling material selection based on composition and processing parameters.

The system is fully interactive, accessible via a Streamlit web interface, and works efficiently on limited computational resources.

Features

Predicts alloy properties using Random Forest, Gradient Boosting, SVM, and Neural Networks.

Handles missing values, normalization, and categorical encoding automatically.

Evaluates model performance with MAE, RMSE, and R² score, along with visualizations.

Provides user-friendly interface for uploading datasets and predicting properties.

Fully reproducible, with preprocessing and models packaged in a clear structure.
2. Upload your CSV dataset

The CSV should contain alloy compositions and properties.

Select the column you want to predict (target column).

3. Train and Evaluate Models

Click “Train Models” in the interface.

View evaluation metrics (MAE, RMSE, R²) for each model.

Visualizations show actual vs predicted values.

4. Optional: Predict Single Alloy

(Future addition) Input alloy composition manually for immediate property prediction.

Machine Learning Models

Random Forest Regressor

Gradient Boosting Regressor

Support Vector Regressor (SVM)

Neural Network (MLP Regressor)

Evaluation Metrics

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

Visual comparison: Actual vs Predicted scatter plots.

Dataset

Use public datasets only, with proper licensing.

Place CSV files in the data/ folder.

The system is designed to handle both numerical and categorical features.

Notes

Designed for low computational resources.

Fully reproducible: all preprocessing steps are included.

Can be extended with additional models or GUI enhancements.
