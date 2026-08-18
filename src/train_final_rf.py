import pandas as pd

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


# Combine 2022 and 2023 into one final training dataset
X_final_train = pd.concat(
    [X_train, X_val],
    ignore_index=True
)

y_final_train = pd.concat(
    [y_train, y_val],
    ignore_index=True
)


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


final_random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                max_features=0.5,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# Train on 2022 + 2023
final_random_forest.fit(
    X_final_train,
    y_final_train
)


# Predict 2024
test_predictions = final_random_forest.predict(
    X_test
)


test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = root_mean_squared_error(
    y_test,
    test_predictions
)

test_r2 = r2_score(
    y_test,
    test_predictions
)


print("\nFinal Random Forest performance on 2024:")
print(f"MAE:  {test_mae:.2f}")
print(f"RMSE: {test_rmse:.2f}")
print(f"R²:   {test_r2:.3f}")