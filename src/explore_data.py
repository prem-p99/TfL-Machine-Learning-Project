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

print(df_long.head(20))

print("\nMissing values:")
print(df_long.isnull().sum())

print("\nEntry statistics:")
print(df_long["Entries"].describe())

print("\nNumber of stations:")
print(df_long["Station"].nunique())

print("\nExample stations:")
print(df_long["Station"].unique()[:20])