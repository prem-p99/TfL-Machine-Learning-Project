import pandas as pd
from pathlib import Path

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

from data_prep import prepare_data


# --------------------------------------------------
# 1. LOAD THE PREPARED DATA
# --------------------------------------------------

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


# --------------------------------------------------
# 2. COMBINE 2022 + 2023 FOR FINAL MODEL TRAINING
# --------------------------------------------------

X_final_train = pd.concat(
    [X_train, X_val],
    ignore_index=True
)

y_final_train = pd.concat(
    [y_train, y_val],
    ignore_index=True
)


# --------------------------------------------------
# 3. DEFINE PREPROCESSING
# --------------------------------------------------

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
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# --------------------------------------------------
# 4. DEFINE THE FINAL MODELS
# --------------------------------------------------

decision_tree = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            DecisionTreeRegressor(
                max_depth=None,
                random_state=42
            )
        )
    ]
)


random_forest = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
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


gradient_boosting = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            GradientBoostingRegressor(
                n_estimators=600,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
        )
    ]
)


# --------------------------------------------------
# 5. TRAIN THE MODELS ON 2022 + 2023
# --------------------------------------------------

print("\nTraining Decision Tree...")

decision_tree.fit(
    X_final_train,
    y_final_train
)


print("Training Random Forest...")

random_forest.fit(
    X_final_train,
    y_final_train
)


print("Training Gradient Boosting...")

gradient_boosting.fit(
    X_final_train,
    y_final_train
)


# --------------------------------------------------
# 6. MAKE PREDICTIONS ON 2024
# --------------------------------------------------

print("\nGenerating 2024 predictions...")


decision_tree_predictions = decision_tree.predict(
    X_test
)


random_forest_predictions = random_forest.predict(
    X_test
)


gradient_boosting_predictions = gradient_boosting.predict(
    X_test
)


# Previous-year baseline:
# use the corresponding 2023 value as the 2024 prediction
previous_year_predictions = y_val.to_numpy()


# --------------------------------------------------
# 7. CREATE THE PREDICTION-LEVEL DATASET
# --------------------------------------------------

predictions = X_test.copy()


predictions["Time"] = (
    time_test.to_numpy()
)


predictions["Actual"] = (
    y_test.to_numpy()
)


predictions["PreviousYearPrediction"] = (
    previous_year_predictions
)


predictions["DecisionTreePrediction"] = (
    decision_tree_predictions
)


predictions["RandomForestPrediction"] = (
    random_forest_predictions
)


predictions["GradientBoostingPrediction"] = (
    gradient_boosting_predictions
)


# --------------------------------------------------
# 8. CALCULATE ABSOLUTE ERRORS FOR EACH MODEL
# --------------------------------------------------

predictions["PreviousYearAbsoluteError"] = (
    predictions["Actual"]
    - predictions["PreviousYearPrediction"]
).abs()


predictions["DecisionTreeAbsoluteError"] = (
    predictions["Actual"]
    - predictions["DecisionTreePrediction"]
).abs()


predictions["RandomForestAbsoluteError"] = (
    predictions["Actual"]
    - predictions["RandomForestPrediction"]
).abs()


predictions["GradientBoostingAbsoluteError"] = (
    predictions["Actual"]
    - predictions["GradientBoostingPrediction"]
).abs()


# --------------------------------------------------
# 9. FUNCTION FOR CALCULATING OVERALL MODEL METRICS
# --------------------------------------------------

def calculate_metrics(
    model_name,
    actual,
    predicted
):
    return {
        "Model": model_name,
        "MAE": mean_absolute_error(
            actual,
            predicted
        ),
        "RMSE": root_mean_squared_error(
            actual,
            predicted
        ),
        "R2": r2_score(
            actual,
            predicted
        )
    }


# --------------------------------------------------
# 10. CALCULATE METRICS FOR ALL MODELS
# --------------------------------------------------

metrics = [
    calculate_metrics(
        "Previous Year Baseline",
        y_test,
        previous_year_predictions
    ),
    calculate_metrics(
        "Decision Tree",
        y_test,
        decision_tree_predictions
    ),
    calculate_metrics(
        "Random Forest",
        y_test,
        random_forest_predictions
    ),
    calculate_metrics(
        "Gradient Boosting",
        y_test,
        gradient_boosting_predictions
    )
]


metrics_df = pd.DataFrame(
    metrics
)


# --------------------------------------------------
# 11. CREATE OUTPUT FOLDER
# --------------------------------------------------

output_folder = Path(
    "data/model_outputs"
)


output_folder.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 12. EXPORT RESULTS FOR POWER BI
# --------------------------------------------------

predictions.to_csv(
    output_folder / "model_predictions.csv",
    index=False
)


metrics_df.to_csv(
    output_folder / "model_metrics.csv",
    index=False
)


# --------------------------------------------------
# 13. DISPLAY RESULTS
# --------------------------------------------------

print("\nModel comparison:")
print(
    metrics_df.to_string(
        index=False
    )
)


print("\nPrediction dataset shape:")
print(
    predictions.shape
)


print(
    "\nSaved files:"
)

print(
    output_folder / "model_predictions.csv"
)

print(
    output_folder / "model_metrics.csv"
)