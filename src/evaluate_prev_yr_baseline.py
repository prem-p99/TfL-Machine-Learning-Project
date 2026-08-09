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


# 2023 acts as the prediction for 2024
previous_year_predictions = y_val


mae = mean_absolute_error(
    y_test,
    previous_year_predictions
)

rmse = root_mean_squared_error(
    y_test,
    previous_year_predictions
)

r2 = r2_score(
    y_test,
    previous_year_predictions
)


print("\nPrevious-year baseline performance on 2024:")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.3f}")