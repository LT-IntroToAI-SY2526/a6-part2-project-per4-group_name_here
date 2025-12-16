"""
Multivariable Linear Regression Project
Assignment 6 Part 3

Group Members:
- Aidan Leuenberger
-
-
-

Dataset: California Housing (sklearn built-in)
Predicting: Median House Value (in $100,000s)
Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
"""

from typing import Any
from numpy._typing._array_like import NDArray
from numpy import float64
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def load_and_explore_data():
    """
    Load the California Housing dataset and print basic information
    """
    print("=" * 70)
    print("LOADING AND EXPLORING DATA")
    print("=" * 70)

    # Load California Housing dataset from sklearn
    california = fetch_california_housing()

    # Create DataFrame with feature names
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df["MedHouseVal"] = california.target  # Target: median house value in $100,000s

    print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nFeatures: {california.feature_names}")
    print("Target: MedHouseVal (Median House Value in $100,000s)")

    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Summary Statistics ---")
    print(df.describe())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("No missing values!")

    return df


def visualize_data(data):
    """
    Create scatter plots showing relationship between each feature and target
    """
    print("\n" + "=" * 70)
    print("VISUALIZING RELATIONSHIPS")
    print("=" * 70)

    feature_columns = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    target_column = "MedHouseVal"

    # Create a 2x4 grid of scatter plots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, feature in enumerate(feature_columns):
        axes[i].scatter(data[feature], data[target_column], alpha=0.3, s=5)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Median House Value")
        axes[i].set_title(f"{feature} vs House Value")

    plt.tight_layout()
    plt.savefig("feature_relationships.png", dpi=150)
    plt.close()

    print("Saved scatter plots to 'feature_relationships.png'")

def prepare_and_split_data(data):
    """
    Separate features and target, then split into train/test sets
    """
    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)

    feature_columns = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    target_column = "MedHouseVal"

    # Separate features (X) and target (y)
    X = data[feature_columns]
    y = data[target_column]

    # Split into 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTotal samples: {len(data)}")
    print(f"Training samples: {len(X_train)} (80%)")
    print(f"Testing samples: {len(X_test)} (20%)")

    return X_train, X_test, y_train, y_test, feature_columns


def train_model(X_train, y_train, feature_names):
    """
    Train the linear regression model
    """
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Print the equation
    print("\n--- Model Equation ---")
    equation = f"MedHouseVal = {model.intercept_:.4f}"
    for name, coef in zip(feature_names, model.coef_):
        equation += f" + ({coef:.4f} * {name})"
    print(equation)

    print("\n--- Feature Importance ---")
    coef_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": model.coef_,
            "Abs_Coefficient": np.abs(model.coef_),
        }
    ).sort_values("Abs_Coefficient", ascending=False)

    for _, row in coef_df.iterrows():
        direction = "+" if row["Coefficient"] > 0 else "-"
        print(f"  {row['Feature']:12} : {row['Coefficient']:>10.4f} ({direction})")

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Model Performance ---")
    print(f"R² Score: {r2:.4f}")
    print(f"  (Model explains {r2 * 100:.1f}% of the variance in house prices)")
    print(f"\nRMSE: {rmse:.4f}")
    print(f"  (Average prediction error: ${rmse * 100000:,.0f})")

    # Comparison table
    print("\n--- Actual vs Predicted (first 10 test samples) ---")
    comparison = pd.DataFrame(
        {
            "Actual": y_test.head(10).values,
            "Predicted": y_pred[:10],
            "Error": y_test.head(10).values - y_pred[:10],
        }
    )
    comparison["Actual_$"] = comparison["Actual"] * 100000
    comparison["Predicted_$"] = comparison["Predicted"] * 100000
    print(comparison[["Actual", "Predicted", "Error"]].to_string())

    # Create actual vs predicted plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(y_test, y_pred, alpha=0.3, s=5)
    axes[0].plot([0, 5], [0, 5], "r--", label="Perfect Prediction")
    axes[0].set_xlabel("Actual Median House Value ($100k)")
    axes[0].set_ylabel("Predicted Median House Value ($100k)")
    axes[0].set_title(f"Actual vs Predicted (R² = {r2:.4f})")
    axes[0].legend()

    # Residual plot
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=5)
    axes[1].axhline(y=0, color="r", linestyle="--")
    axes[1].set_xlabel("Predicted Value")
    axes[1].set_ylabel("Residual (Actual - Predicted)")
    axes[1].set_title("Residual Plot")

    plt.tight_layout()
    plt.savefig("model_evaluation.png", dpi=150)
    plt.close()

    print("\nSaved evaluation plots to 'model_evaluation.png'")

    return y_pred


def make_prediction(model, feature_names):
    """
    Make a prediction for a sample house
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)

    # Example: A house in a wealthy area with moderate size
    sample_values = {
        "MedInc": 5.0,  # Median income ~$50k
        "HouseAge": 25.0,  # 25 years old
        "AveRooms": 6.0,  # 6 rooms average
        "AveBedrms": 1.0,  # 1 bedroom average
        "Population": 1500.0,  # 1500 people in block
        "AveOccup": 3.0,  # 3 people per household
        "Latitude": 34.05,  # Los Angeles area
        "Longitude": -118.25,  # Los Angeles area
    }

    sample = pd.DataFrame([sample_values])

    print("\n--- Sample House Features ---")
    for feature, value in sample_values.items():
        print(f"  {feature}: {value}")

    prediction = model.predict(sample)[0]

    print("\n--- Prediction ---")
    print(f"Predicted Median House Value: {prediction:.4f} ($100k)")
    print(f"In dollars: ${prediction * 100000:,.0f}")


if __name__ == "__main__":
    # Step 1: Load and explore
    data = load_and_explore_data()

    # Step 2: Visualize
    visualize_data(data)

    # Step 3: Prepare and split
    X_train, X_test, y_train, y_test, feature_names = prepare_and_split_data(data)

    # Step 4: Train
    model = train_model(X_train, y_train, feature_names)

    # Step 5: Evaluate
    predictions = evaluate_model(model, X_test, y_test)

    # Step 6: Make a prediction
    make_prediction(model, feature_names)

    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Analyze your results")
    print("2. Try improving your model (add/remove features)")
    print("3. Create your presentation")
    print("4. Practice presenting with your group!")
