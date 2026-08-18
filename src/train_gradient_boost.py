from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

from data_prep import prepare_data


(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    time_train,
    time_val,
    time_test
) = prepare_data()


categorical_features = [
    "Station",
    "Fare Zone"
]

numeric_features = [
    "MinutesSinceMidnight"
]


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


gradient_boosting = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        )
    ]
)


# Train only on 2022
gradient_boosting.fit(
    X_train,
    y_train
)


# Predict training and validation data
train_predictions = gradient_boosting.predict(
    X_train
)

val_predictions = gradient_boosting.predict(
    X_val
)


# Training metrics
train_mae = mean_absolute_error(
    y_train,
    train_predictions
)

train_rmse = root_mean_squared_error(
    y_train,
    train_predictions
)

train_r2 = r2_score(
    y_train,
    train_predictions
)


# Validation metrics
val_mae = mean_absolute_error(
    y_val,
    val_predictions
)

val_rmse = root_mean_squared_error(
    y_val,
    val_predictions
)

val_r2 = r2_score(
    y_val,
    val_predictions
)


print("\nGradient Boosting training performance on 2022:")
print(f"MAE:  {train_mae:.2f}")
print(f"RMSE: {train_rmse:.2f}")
print(f"R²:   {train_r2:.3f}")

print("\nGradient Boosting validation performance on 2023:")
print(f"MAE:  {val_mae:.2f}")
print(f"RMSE: {val_rmse:.2f}")
print(f"R²:   {val_r2:.3f}")