from sklearn.ensemble import RandomForestRegressor
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


random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# Train only on 2022
random_forest.fit(
    X_train,
    y_train
)


# Predictions on training and validation data
train_predictions = random_forest.predict(
    X_train
)

val_predictions = random_forest.predict(
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


print("\nRandom Forest training performance on 2022:")
print(f"MAE:  {train_mae:.2f}")
print(f"RMSE: {train_rmse:.2f}")
print(f"R²:   {train_r2:.3f}")

print("\nRandom Forest validation performance on 2023:")
print(f"MAE:  {val_mae:.2f}")
print(f"RMSE: {val_rmse:.2f}")
print(f"R²:   {val_r2:.3f}")