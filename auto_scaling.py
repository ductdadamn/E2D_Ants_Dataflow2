import re
import pandas as pd
from pathlib import Path

# =========================
# 1. LOG PARSING
# =========================

LOG_PATTERN = re.compile(
    r'(?P<host>\S+) \S+ \S+ '
    r'\[(?P<time>[^\]]+)\] '
    r'"(?P<method>\S+) (?P<path>\S+) (?P<protocol>[^"]+)" '
    r'(?P<status>\d{3}) (?P<bytes>\S+)'
)

def load_log(file_path):
    print(f"Parsing {file_path} ...")
    rows = []

    with open(file_path, "r", errors="ignore") as f:
        for line in f:
            m = LOG_PATTERN.match(line)
            if not m:
                continue

            d = m.groupdict()
            d["status"] = int(d["status"])
            d["bytes"] = int(d["bytes"]) if d["bytes"].isdigit() else 0
            rows.append(d)

    df = pd.DataFrame(rows)

    df["ts"] = pd.to_datetime(
        df["time"],
        format="%d/%b/%Y:%H:%M:%S %z"
    )

    df = df.drop(columns=["time"])
    df = df.set_index("ts").sort_index()

    print("rows:", len(df))
    print(df.head())

    return df


# =========================
# 2. PER-MINUTE AGGREGATION
# =========================

def aggregate_per_minute(df):
    per_1m = df.resample("1min").agg(
        requests=("host", "count"),
        bytes=("bytes", "sum")
    )

    return per_1m.fillna(0)


# =========================
# 3. STATUS CODE FEATURES (FIXED)
# =========================

def add_status_features(df, per_1m):
    # Create status groups: 200, 300, 400, 500
    df["status_group"] = (df["status"] // 100) * 100

    sg = (
        df
        .groupby([pd.Grouper(freq="1min"), "status_group"])
        .size()
        .unstack(fill_value=0)
    )

    sg.columns = [f"status_{c}" for c in sg.columns]

    per_1m = per_1m.join(sg, how="left").fillna(0)

    return per_1m


# =========================
# 4. MAIN PIPELINE
# =========================

def main():
    base = Path("auto_scaling/data/raw")

    # CHANGE THESE NAMES if your files are named differently
    train_file = base / "train.txt"
    test_file = base / "test.txt"

    train_df = load_log(train_file)

    print("\nStatus counts:\n", train_df["status"].value_counts())
    print("Total bytes:", train_df["bytes"].sum())

    per_1m = aggregate_per_minute(train_df)
    per_1m = add_status_features(train_df, per_1m)

    print("\nSample of per_1m:")
    print(per_1m.head())

    # Optional: save processed data
    output = Path("data/processed")
    output.mkdir(parents=True, exist_ok=True)

    per_1m.to_csv(output / "train_per_1m.csv")
    print("\nSaved to data/processed/train_per_1m.csv")


if __name__ == "__main__":
    main()
