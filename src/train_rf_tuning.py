from sklearn.ensemble import RandomForestRegressor
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


configs = [
    {
        "name": "100 trees, all features",
        "n_estimators": 100,
        "max_features": 1.0
    },
    {
        "name": "300 trees, all features",
        "n_estimators": 300,
        "max_features": 1.0
    },
    {
        "name": "100 trees, half features",
        "n_estimators": 100,
        "max_features": 0.5
    },
    {
        "name": "300 trees, half features",
        "n_estimators": 300,
        "max_features": 0.5
    }
]


for config in configs:

    random_forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=config["n_estimators"],
                    max_features=config["max_features"],
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )


    random_forest.fit(
        X_train,
        y_train
    )


    train_predictions = random_forest.predict(
        X_train
    )

    val_predictions = random_forest.predict(
        X_val
    )


    train_mae = mean_absolute_error(
        y_train,
        train_predictions
    )

    train_r2 = r2_score(
        y_train,
        train_predictions
    )


    val_mae = mean_absolute_error(
        y_val,
        val_predictions
    )

    val_r2 = r2_score(
        y_val,
        val_predictions
    )


    print(f"\n{config['name']}")

    print(
        f"2022 Training   -> MAE: {train_mae:.2f}, "
        f"R²: {train_r2:.3f}"
    )

    print(
        f"2023 Validation -> MAE: {val_mae:.2f}, "
        f"R²: {val_r2:.3f}"
    )