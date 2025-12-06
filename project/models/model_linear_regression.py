"""
Linear Regression Model for Lead Futures Prediction

This script loads the ready_to_train.csv dataset and trains a linear regression model
to predict the target variable.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import joblib


def load_data(data_path: str) -> tuple:
    """
    Load and prepare the training data.
    
    Parameters:
        data_path: Path to the ready_to_train.csv file
    
    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature column names
    """
    # Load the data
    df = pd.read_csv(data_path)
    
    print(f"Data loaded: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Separate features and target
    # Target column is 'Target'
    target_col = 'Target'
    
    # Feature columns are all columns except 'Date' and 'Target'
    feature_cols = [col for col in df.columns if col not in ['Date', target_col]]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print(f"Target distribution:")
    print(f"  Class 0: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.2f}%)")
    print(f"  Class 1: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.2f}%)")
    
    # Check for missing values
    if df[feature_cols].isna().sum().sum() > 0:
        print("\nWarning: Missing values detected in features")
        print(df[feature_cols].isna().sum())
    
    return X, y, feature_cols


def train_linear_regression(X, y, feature_names=None, test_size=0.2, random_state=42):
    """
    Train a linear regression model.
    
    Parameters:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names (optional)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        model: Trained LinearRegression model
        X_train, X_test, y_train, y_test: Train/test splits
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Initialize and train the model
    print("\nTraining linear regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n" + "="*60)
    print("Model Performance Metrics")
    print("="*60)
    print(f"\nTraining Set:")
    print(f"  MSE:  {train_mse:.6f}")
    print(f"  MAE:  {train_mae:.6f}")
    print(f"  R²:   {train_r2:.6f}")
    
    print(f"\nTest Set:")
    print(f"  MSE:  {test_mse:.6f}")
    print(f"  MAE:  {test_mae:.6f}")
    print(f"  R²:   {test_r2:.6f}")
    
    # Display model coefficients
    print("\n" + "="*60)
    print("Model Coefficients")
    print("="*60)
    print(f"Intercept: {model.intercept_:.6f}")
    print("\nFeature Coefficients:")
    if feature_names:
        # Sort by absolute coefficient value for better interpretability
        coef_data = list(zip(feature_names, model.coef_))
        coef_data.sort(key=lambda x: abs(x[1]), reverse=True)
        for feature_name, coef in coef_data:
            print(f"  {feature_name:25s}: {coef:10.6f}")
    else:
        for i, coef in enumerate(model.coef_):
            print(f"  Feature {i}: {coef:.6f}")
    
    return model, X_train, X_test, y_train, y_test


def save_model(model, filepath: str):
    """
    Save the trained model to disk.
    
    Parameters:
        model: Trained model
        filepath: Path where to save the model
    """
    joblib.dump(model, filepath)
    print(f"\nModel saved to: {filepath}")


def main():
    """Main function to run the training pipeline."""
    # Set paths
    data_path = Path(__file__).parent.parent / "data" / "ready_to_train.csv"
    model_path = Path(__file__).parent / "linear_regression_model.pkl"
    
    print("="*60)
    print("Linear Regression Model Training")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    X, y, feature_names = load_data(str(data_path))
    
    # Train model
    model, X_train, X_test, y_train, y_test = train_linear_regression(X, y, feature_names=feature_names)
    
    # Save model
    save_model(model, str(model_path))
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

