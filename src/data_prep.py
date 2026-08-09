import pandas as pd
from pathlib import Path


def load_year(file, year):
    # Load station entry data
    df = pd.read_excel(
        file,
        sheet_name="Station_Entries",
        header=2
    )

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

    # Columns identifying each station
    id_columns = [
        "NLC",
        "ASC",
        "Station",
        "Fare Zone"
    ]

    # Quarter-hour columns begin after the first 11 columns
    time_columns = df.columns[11:]

    # Reshape from wide format to long format
    df_long = df.melt(
        id_vars=id_columns,
        value_vars=time_columns,
        var_name="Time",
        value_name="Entries"
    )

    # Load station boarding data so we can identify Underground stations
    station_boarders = pd.read_excel(
        file,
        sheet_name="Station_Boarders",
        header=2
    )

    station_boarders.columns = (
        station_boarders.columns.str.strip()
    )

    # Find Underground station IDs
    tube_station_ids = (
        station_boarders[
            station_boarders["Mode"] == "LU"
        ]["NLC"]
        .unique()
    )

    # Keep only Underground stations
    df_tube = df_long[
        df_long["NLC"].isin(tube_station_ids)
    ].copy()

    # Ensure Fare Zone behaves as a categorical variable
    df_tube["Fare Zone"] = (
        df_tube["Fare Zone"].astype(str)
    )

    # Extract the start of the 15-minute interval
    df_tube["StartTime"] = (
        df_tube["Time"].str[:4]
    )

    df_tube["Hour"] = (
        df_tube["StartTime"]
        .str[:2]
        .astype(int)
    )

    df_tube["Minute"] = (
        df_tube["StartTime"]
        .str[2:]
        .astype(int)
    )

    # Convert time into minutes since midnight
    df_tube["MinutesSinceMidnight"] = (
        df_tube["Hour"] * 60
        + df_tube["Minute"]
    )

    # Record which NUMBAT year each observation came from
    df_tube["Year"] = year

    return df_tube


def prepare_data():
    data_folder = Path("data/raw")

    # Explicitly identify each year's workbook
    file_2022 = data_folder / "NBT22TWT_outputs.xlsx"
    file_2023 = data_folder / "NBT23TWT_outputs.xlsx"
    file_2024 = data_folder / "NBT24TWT_outputs.xlsx"

    # Apply the same cleaning process to every year
    df_2022 = load_year(
        file_2022,
        2022
    )

    df_2023 = load_year(
        file_2023,
        2023
    )

    df_2024 = load_year(
        file_2024,
        2024
    )

    # Features available to the model
    features = [
        "Station",
        "Fare Zone",
        "MinutesSinceMidnight"
    ]

    # 2022 = training data
    X_train = df_2022[features]
    y_train = df_2022["Entries"]

    # 2023 = validation data
    X_val = df_2023[features]
    y_val = df_2023["Entries"]

    # 2024 = final test data
    X_test = df_2024[features]
    y_test = df_2024["Entries"]

    # Keep readable time labels for diagnostics
    time_train = df_2022["Time"]
    time_val = df_2023["Time"]
    time_test = df_2024["Time"]

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        time_train,
        time_val,
        time_test
    )

if __name__ == "__main__":
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

    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)