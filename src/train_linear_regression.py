from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt

from data_prep import prepare_data


# Get the exact same prepared train/test data every time
X_train, X_test, y_train, y_test, time_train, time_test = prepare_data()


# Baseline model:
# always predict the mean entry count from the training set
baseline_value = y_train.mean()

baseline_predictions = [baseline_value] * len(y_test)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = root_mean_squared_error(
    y_test,
    baseline_predictions
)

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)


print("\nBaseline performance:")
print(f"MAE:  {baseline_mae:.2f}")
print(f"RMSE: {baseline_rmse:.2f}")
print(f"R²:   {baseline_r2:.3f}")


# Categorical features need to be converted into numbers
categorical_features = [
    "Station",
    "Fare Zone"
]

# This feature is already numerical
numeric_features = [
    "MinutesSinceMidnight"
]


# Apply one-hot encoding to categorical columns,
# but leave the numerical time feature unchanged
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# Pipeline:
# raw features -> preprocessing -> Linear Regression
linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# Train the model
linear_model.fit(
    X_train,
    y_train
)


# Make predictions on unseen test data
linear_predictions = linear_model.predict(
    X_test
)


# Evaluate predictions
linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = root_mean_squared_error(
    y_test,
    linear_predictions
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


print("\nLinear Regression performance:")
print(f"MAE:  {linear_mae:.2f}")
print(f"RMSE: {linear_rmse:.2f}")
print(f"R²:   {linear_r2:.3f}")


# Create a results table for diagnostics
results = X_test.copy()

results["Time"] = time_test
results["Actual"] = y_test
results["Predicted"] = linear_predictions

results["Error"] = (
    results["Predicted"]
    - results["Actual"]
)

results["AbsoluteError"] = (
    results["Error"].abs()
)


# Find the 20 largest mistakes
worst_predictions = (
    results
    .sort_values(
        "AbsoluteError",
        ascending=False
    )
    .head(20)
)


print("\nWorst 20 predictions:")
print(worst_predictions.to_string())


# Plot actual versus predicted demand
plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    linear_predictions,
    alpha=0.3
)

plt.xlabel("Actual entries")
plt.ylabel("Predicted entries")
plt.title(
    "Linear Regression: Actual vs Predicted Entries"
)

plt.tight_layout()
plt.show()