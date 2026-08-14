import os
import pandas as pd
 
# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
# Path to the raw CSV. Kaggle's AI4I 2020 dataset is usually distributed
# as "ai4i2020.csv" — adjust the filename here if yours differs.
RAW_DATA_PATH = os.path.join("data", "raw", "ai4i2020.csv")
 
# Mapping for the Type column. In AI4I 2020, L/M/H represents product
# quality variant (Low, Medium, High), which is an ORDINAL category
# (there's a natural order L < M < H), not a random label. So instead of
# one-hot encoding (which would make 3 new columns), we map it to a
# single integer column 0/1/2. This keeps things simple and keeps the
# ordering information, which one-hot encoding would throw away.
TYPE_ENCODING = {"L": 0, "M": 1, "H": 2}
 
# Columns that are just row/product identifiers — they carry no
# predictive signal (they're basically like a database primary key),
# so we drop them before modeling.
ID_COLUMNS_TO_DROP = ["UDI", "Product ID"]
 
 
# -----------------------------------------------------------------------
# STEP 1: LOAD
# -----------------------------------------------------------------------
def load_raw_csv(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Reads the raw CSV straight off disk, no cleaning yet.
    Kept separate from load_data() so teammates (or tests) can inspect
    the truly raw data if they want to.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Couldn't find the dataset at '{path}'.\n"
            f"Download 'ai4i2020.csv' from Kaggle "
            f"(stephanmatzka/predictive-maintenance-dataset-ai4i-2020) "
            f"and place it in data/raw/."
        )
    df = pd.read_csv(path)
    return df
 
 
# -----------------------------------------------------------------------
# STEP 2: NULL CHECK
# -----------------------------------------------------------------------
def check_nulls(df: pd.DataFrame) -> pd.Series:
    """
    Counts null values per column and prints a quick summary.
    Returns the per-column null counts (a pandas Series) in case
    someone wants to log or assert on it later.
    """
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
 
    print("----- Null Value Check -----")
    if total_nulls == 0:
        print("No null values found. Dataset is clean on that front.")
    else:
        print("Null values found per column:")
        print(null_counts[null_counts > 0])
    print()
 
    return null_counts
 
 
# -----------------------------------------------------------------------
# STEP 3: CLASS BALANCE CHECK
# -----------------------------------------------------------------------
def check_class_balance(df: pd.DataFrame, target_col: str = "Machine failure") -> pd.Series:
    """
    Counts how many rows are failures (1) vs non-failures (0).
    This matters a lot here: AI4I 2020 is heavily imbalanced
    (failures are rare events, as they should be in real machines),
    so the prediction teammate needs to know this upfront — plain
    accuracy will look great even if the model just predicts "0" every time.
    """
    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100
 
    print("----- Class Balance Check -----")
    for label in counts.index:
        print(f"Machine failure = {label}: {counts[label]} rows "
              f"({percentages[label]:.2f}%)")
    print()
 
    return counts
 
 
# -----------------------------------------------------------------------
# STEP 4: ENCODE THE 'Type' COLUMN
# -----------------------------------------------------------------------
def encode_type_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps the categorical 'Type' column (L/M/H) to integers using
    TYPE_ENCODING, and stores it in a new column 'Type_encoded'.
    We keep the original 'Type' column too (in case someone wants the
    human-readable label for the explainability step), we just add the
    numeric version alongside it for the classifier to use.
    """
    df = df.copy()  # avoid mutating the caller's DataFrame unexpectedly
    df["Type_encoded"] = df["Type"].map(TYPE_ENCODING)
 
    # Sanity check: if any value didn't map (unexpected category showed up),
    # map() would produce NaN here — better to catch that now than let it
    # silently poison the model later.
    if df["Type_encoded"].isnull().any():
        unmapped = df.loc[df["Type_encoded"].isnull(), "Type"].unique()
        raise ValueError(f"Found Type values with no encoding defined: {unmapped}")
 
    return df
 
 
# -----------------------------------------------------------------------
# STEP 5: DROP IRRELEVANT ID COLUMNS
# -----------------------------------------------------------------------
def drop_id_columns(df: pd.DataFrame, columns=ID_COLUMNS_TO_DROP) -> pd.DataFrame:
    """
    Drops columns that are just identifiers (UDI, Product ID).
    errors="ignore" means this won't crash if a column is already
    missing (useful if someone re-runs the pipeline on partially
    processed data).
    """
    df = df.copy()
    df = df.drop(columns=columns, errors="ignore")
    return df
 
 
# -----------------------------------------------------------------------
# MAIN ENTRY POINT: load_data()
# -----------------------------------------------------------------------
def load_data(path: str = RAW_DATA_PATH, verbose: bool = True) -> pd.DataFrame:
    """
    The single function everyone else on the team should call.
 
    Runs the full Day-1 pipeline:
        raw CSV -> null check -> class balance check -> encode Type -> drop IDs
    and returns a clean, ready-to-use DataFrame.
 
    Parameters
    ----------
    path : str
        Path to the raw CSV file.
    verbose : bool
        If True, prints the null-check and class-balance summaries.
        Set to False if you just want the DataFrame with no console output
        (e.g. when imported inside another script).
 
    Returns
    -------
    pd.DataFrame
        Cleaned dataset: no ID columns, Type encoded as Type_encoded,
        nulls checked (AI4I 2020 has none, but we still verify).
    """
    df = load_raw_csv(path)
 
    if verbose:
        check_nulls(df)
        check_class_balance(df)
 
    df = encode_type_column(df)
    df = drop_id_columns(df)
 
    if verbose:
        print("----- load_data() complete -----")
        print(f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
 
    return df
 
 
# -----------------------------------------------------------------------
# SCRIPT MODE: lets you just run `python src/data_pipeline.py` to sanity
# check everything works end-to-end without importing it elsewhere.
# -----------------------------------------------------------------------
if __name__ == "__main__":
    clean_df = load_data()
    print()
    print(clean_df.head())
