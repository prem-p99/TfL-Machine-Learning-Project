import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def prepare_data():
    # Find the TfL NUMBAT Excel workbook
    data_folder = Path("data/raw")
    file = list(data_folder.glob("*.xlsx"))[0]

    # Load station entry data
    df = pd.read_excel(
        file,
        sheet_name="Station_Entries",
        header=2
    )

    # Columns that identify each station
    id_columns = [
        "NLC",
        "ASC",
        "Station",
        "Fare Zone"
    ]

    # The 96 quarter-hour columns start after the first 11 columns
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

    # Find the unique station IDs where Mode is London Underground
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

    # Make Fare Zone consistently categorical
    df_tube["Fare Zone"] = df_tube["Fare Zone"].astype(str)

    # Extract the start time from labels such as "0815-0830"
    df_tube["StartTime"] = df_tube["Time"].str[:4]

    # Turn the start time into numerical hour and minute features
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

    # Convert time into a single numerical value
    # Example: 08:15 -> 8 * 60 + 15 = 495
    df_tube["MinutesSinceMidnight"] = (
        df_tube["Hour"] * 60
        + df_tube["Minute"]
    )

    # Features given to the model
    features = [
        "Station",
        "Fare Zone",
        "MinutesSinceMidnight"
    ]

    X = df_tube[features]

    # Target that the model tries to predict
    y = df_tube["Entries"]

    # Keep the readable time labels for later diagnostics
    time_labels = df_tube["Time"]

    # Split everything using exactly the same random split
    (
        X_train,
        X_test,
        y_train,
        y_test,
        time_train,
        time_test
    ) = train_test_split(
        X,
        y,
        time_labels,
        test_size=0.2,
        random_state=42
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        time_train,
        time_test
    )