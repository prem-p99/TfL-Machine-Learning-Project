import pandas as pd
from pathlib import Path


data_folder = Path("data/raw")
files = list(data_folder.glob("*.xlsx"))
file = files[0]

df = pd.read_excel(
    file,
    sheet_name="Station_Entries",
    header=2
)

id_columns = [
    "NLC",
    "ASC",
    "Station",
    "Fare Zone"
]

time_columns = df.columns[11:]

df_long = df.melt(
    id_vars=id_columns,
    value_vars=time_columns,
    var_name="Time",
    value_name="Entries"
)

# print(df_long.head(20))

# print("\nMissing values:")
# print(df_long.isnull().sum())

# print("\nEntry statistics:")
# print(df_long["Entries"].describe())

# print("\nNumber of stations:")
# print(df_long["Station"].nunique())

# print("\nExample stations:")
# print(df_long["Station"].unique()[:20])

station_boarders = pd.read_excel(
    file,
    sheet_name="Station_Boarders",
    header=2
)

tube_boarders = station_boarders[
    station_boarders["Mode"] == "LU"
]

# print("\nUnderground boarding rows:")
# print(tube_boarders.shape)

# print("\nNumber of Underground stations:")
# print(tube_boarders["NLC"].nunique())

# print("\nExample Underground stations:")
# print(tube_boarders["Station"].unique()[:20])

tube_station_ids = tube_boarders["NLC"].unique()

df_tube = df_long[
    df_long["NLC"].isin(tube_station_ids)
].copy()

# print("\nTube-only dataset:")
# print(df_tube.head(20))

# print("\nTube-only shape:")
# print(df_tube.shape)

# print("\nTube stations:")
# print(df_tube["Station"].nunique())

# print("\nUnderground line codes:")
# print(sorted(tube_boarders["Line"].unique()))

import matplotlib.pyplot as plt


# Total Underground entries by 15-minute interval
demand_by_time = (
    df_tube
    .groupby("Time")["Entries"]
    .sum()
    .reset_index()
)

print("\nDemand by time:")
print(demand_by_time.head())


# Total entries by station
demand_by_station = (
    df_tube
    .groupby("Station")["Entries"]
    .sum()
    .sort_values(ascending=False)
)

# print("\nTop 10 busiest stations:")
# print(demand_by_station.head(10))

# plt.figure(figsize=(14, 6))

# plt.plot(
#     demand_by_time["Time"],
#     demand_by_time["Entries"]
# )

# plt.xticks(rotation=90)
# plt.xlabel("15-minute interval")
# plt.ylabel("Total station entries")
# plt.title("London Underground demand throughout the day")

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 6))

# plt.hist(
#     df_tube["Entries"],
#     bins=50
# )

# plt.xlabel("Entries per 15-minute interval")
# plt.ylabel("Number of observations")
# plt.title("Distribution of London Underground station entries")

# plt.tight_layout()
# plt.show()

# top_15 = demand_by_station.head(15).sort_values()

# plt.figure(figsize=(10, 7))

# top_15.plot(kind="barh")

# plt.xlabel("Total daily entries")
# plt.ylabel("Station")
# plt.title("15 busiest London Underground stations")

# plt.tight_layout()
# plt.show()

df_tube["StartTime"] = df_tube["Time"].str[:4]

df_tube["Hour"] = df_tube["StartTime"].str[:2].astype(int)

df_tube["Minute"] = df_tube["StartTime"].str[2:].astype(int)

# print(df_tube[
#     ["Station", "Time", "StartTime", "Hour", "Minute", "Entries"]
# ].head(20))

df_tube["MinutesSinceMidnight"] = (
    df_tube["Hour"] * 60
    + df_tube["Minute"]
)

# print(df_tube[
#     [
#         "Station",
#         "Time",
#         "Hour",
#         "Minute",
#         "MinutesSinceMidnight",
#         "Entries"
#     ]
# ].head(20))

output_file = Path("data/cleaned/tube_entries_cleaned.xlsx")

output_file.parent.mkdir(parents=True, exist_ok=True)

df_tube.to_excel(
    output_file,
    index=False
)

print(f"\nCleaned dataset saved to: {output_file}")