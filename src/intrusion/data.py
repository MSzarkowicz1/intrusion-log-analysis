import pandas as pd


def load_df(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["encryption_used"] = df["encryption_used"].fillna("None").astype("category")
    return df
