from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

from data_prep import prepare_data


# Load the same train/test split used by the other models
X_train, X_test, y_train, y_test, time_train, time_test = prepare_data()


# Station and Fare Zone are categorical
categorical_features = [
    "Station",
    "Fare Zone"
]

# Time is already numerical
numeric_features = [
    "MinutesSinceMidnight"
]


# Convert categorical features using one-hot encoding
# and leave the numerical feature unchanged
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


# Build a pipeline:
# preprocessing -> Decision Tree
tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            DecisionTreeRegressor(
                random_state=42
            )
        )
    ]
)


# Train the Decision Tree on the training data
tree_model.fit(
    X_train,
    y_train
)

tree_train_predictions = tree_model.predict(
    X_train
)

# Predict the unseen test observations
tree_predictions = tree_model.predict(
    X_test
)


# Evaluate test-set performance
tree_mae = mean_absolute_error(
    y_test,
    tree_predictions
)

tree_rmse = root_mean_squared_error(
    y_test,
    tree_predictions
)

tree_r2 = r2_score(
    y_test,
    tree_predictions
)

train_mae = mean_absolute_error(
    y_train,
    tree_train_predictions
)

train_rmse = root_mean_squared_error(
    y_train,
    tree_train_predictions
)

train_r2 = r2_score(
    y_train,
    tree_train_predictions
)


print("\nDecision Tree performance:")
print(f"MAE:  {tree_mae:.2f}")
print(f"RMSE: {tree_rmse:.2f}")
print(f"R²:   {tree_r2:.3f}")

print("\nDecision Tree training performance:")
print(f"MAE:  {train_mae:.2f}")
print(f"RMSE: {train_rmse:.2f}")
print(f"R²:   {train_r2:.3f}")