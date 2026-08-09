from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
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


depths = [
    3,
    5,
    10,
    15,
    20,
    None
]


for depth in depths:

    tree_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                DecisionTreeRegressor(
                    max_depth=depth,
                    random_state=42
                )
            )
        ]
    )

    # Train ONLY using 2022
    tree_model.fit(
        X_train,
        y_train
    )

    # How well does it fit 2022?
    train_predictions = tree_model.predict(
        X_train
    )

    # How well does it generalise to the next year, 2023?
    val_predictions = tree_model.predict(
        X_val
    )

    train_r2 = r2_score(
        y_train,
        train_predictions
    )

    val_r2 = r2_score(
        y_val,
        val_predictions
    )

    train_mae = mean_absolute_error(
        y_train,
        train_predictions
    )

    val_mae = mean_absolute_error(
        y_val,
        val_predictions
    )

    print(f"\nMax depth: {depth}")

    print(
        f"2022 Training   -> MAE: {train_mae:.2f}, "
        f"R²: {train_r2:.3f}"
    )

    print(
        f"2023 Validation -> MAE: {val_mae:.2f}, "
        f"R²: {val_r2:.3f}"
    )