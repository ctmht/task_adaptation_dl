import csv
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import pickle
import re
import os


def read_arff_columns(arff_path: str) -> list[str]:
    cols = []
    with open(arff_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            low = s.lower()
            if low.startswith("@attribute"):
                parts = s.split(None, 2)
                cols.append(parts[1].strip())
            elif low.startswith("@data"):
                break
    return cols


def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    h = np.floor(s / 100)
    m = s % 100
    return (h * 60 + m).astype("float64")

def read_arff_nominal_values(arff_path, attr_name):
    """
    reads the ARFF header and extracts the allowed categorical values for one nominal attribute,
    like all airports for Origin or all carriers for UniqueCarrier
    """
    pat = re.compile(r"^@attribute\s+" + re.escape(attr_name) + r"\s+\{(.*)\}\s*$", re.IGNORECASE)
    with open(arff_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            m = pat.match(s)
            if m:
                inside = m.group(1)
                # split by commas and strip whitespace and quotes
                vals = []
                for v in inside.split(","):
                    v = v.strip()
                    if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                        v = v[1:-1]
                    vals.append(v)
                return vals
            if s.lower().startswith("@data"):
                break
    raise ValueError(f"Could not find nominal values for attribute {attr_name} in header")

def arff_to_train_val_test_h5(
    arff_path: str,
    train_h5: str,
    val_h5: str,
    test_h5: str,
    *,
    target_col: str = "DepDelay",
    splits=(0.8, 0.1, 0.1),
    seed: int = 42,
    chunksize: int = 250_000,
):
    p_train, p_val, p_test = splits
    p_train = float(p_train)
    p_val = float(p_val)

    # start fresh
    for path in [train_h5, val_h5, test_h5]:
        if os.path.exists(path):
            os.remove(path)

    cols = read_arff_columns(arff_path)

    # categorical encoders
    carriers = read_arff_nominal_values(arff_path, "UniqueCarrier")
    origins  = read_arff_nominal_values(arff_path, "Origin")
    dests    = read_arff_nominal_values(arff_path, "Dest")

    # Origin and Dest must share the SAME mapping
    airport_levels = sorted(set(origins) | set(dests))

    carrier_to_id = {c: i for i, c in enumerate(carriers, start=1)}
    airport_to_id = {a: i for i, a in enumerate(airport_levels, start=1)}

    # save integer encoders to decode later if needed
    with open("cat_encoders.pkl", "wb") as f:
        pickle.dump(
            {"carrier_to_id": carrier_to_id, "airport_to_id": airport_to_id,
            "carriers": carriers, "airport_levels": airport_levels},
            f
    )

    # numeric columns
    numeric_cols = [
        "DepDelay",
        "Month",
        "DayofMonth",
        "DayOfWeek",
        "CRSDepTime",
        "CRSArrTime",
        "Distance",
    ]

    # remaining numerical features to z-transform (NOT time ones)
    z_cols = ["Distance"]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    rng = np.random.default_rng(seed)
    rows = []

    with open(arff_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        for line in f:
            if line.strip().lower().startswith("@data"):
                break

        reader = csv.reader(f, quotechar="'", delimiter=",", skipinitialspace=True)

        for row in reader:
            if not row or row[0].strip().startswith("%"):
                continue
            row = [None if x.strip() == "?" else x for x in row]
            rows.append(row)

            if len(rows) >= chunksize:
                df = pd.DataFrame(rows, columns=cols)
                for c in numeric_cols:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=[target_col])

                r = rng.random(len(df))
                train_mask = r < p_train

                x_scaler.partial_fit(df.loc[train_mask, z_cols])
                y_scaler.partial_fit(df.loc[train_mask, [target_col]])

                rows = []

        if rows:
            df = pd.DataFrame(rows, columns=cols)
            for c in numeric_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=[target_col])

            r = rng.random(len(df))
            train_mask = r < p_train
            x_scaler.partial_fit(df.loc[train_mask, z_cols])
            y_scaler.partial_fit(df.loc[train_mask, [target_col]])

    # save scaler
    with open("x_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)
    with open("depdelay_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)

    # Transform time features + write H5 files
    rng = np.random.default_rng(seed)  # reset so split is identical
    rows = []

    with open(arff_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        for line in f:
            if line.strip().lower().startswith("@data"):
                break

        reader = csv.reader(f, quotechar="'", delimiter=",", skipinitialspace=True)

        for row in reader:
            if not row or row[0].strip().startswith("%"):
                continue
            row = [None if x.strip() == "?" else x for x in row]
            rows.append(row)

            if len(rows) >= chunksize:
                df = pd.DataFrame(rows, columns=cols)
                for c in numeric_cols:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=[target_col])

                # encode categoricals (GLOBAL mapping)
                df["UniqueCarrier"] = df["UniqueCarrier"].map(carrier_to_id).fillna(0).astype("int32")
                df["Origin"] = df["Origin"].map(airport_to_id).fillna(0).astype("int32")
                df["Dest"]   = df["Dest"].map(airport_to_id).fillna(0).astype("int32")

                # sin/cos time features
                m = df["Month"]
                dow = df["DayOfWeek"]

                df["month_sin"] = np.sin(2 * np.pi * (m - 1) / 12)
                df["month_cos"] = np.cos(2 * np.pi * (m - 1) / 12)

                df["dow_sin"] = np.sin(2 * np.pi * (dow - 1) / 7)
                df["dow_cos"] = np.cos(2 * np.pi * (dow - 1) / 7)

                dep_min = hhmm_to_minutes(df["CRSDepTime"])
                arr_min = hhmm_to_minutes(df["CRSArrTime"])

                df["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440)
                df["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440)

                df["arr_sin"] = np.sin(2 * np.pi * arr_min / 1440)
                df["arr_cos"] = np.cos(2 * np.pi * arr_min / 1440)

                # DayofMonth assumes non-leap year for simplicity (and lack of year)
                MONTH_DAYS = np.array(
                    [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
                )

                days_in_month = MONTH_DAYS[m.values.astype(int) - 1]
                dom_phase = 2 * np.pi * (df["DayofMonth"] - 1) / days_in_month
                df["dom_sin"] = np.sin(dom_phase)
                df["dom_cos"] = np.cos(dom_phase)

                # z-transform remaining numeric features
                df[z_cols] = x_scaler.transform(df[z_cols])

                y = pd.DataFrame(
                    y_scaler.transform(df[[target_col]]),
                    columns=[target_col],
                    index=df.index,
                )

                df = df.drop(
                    columns=["Month", "DayOfWeek", "CRSDepTime", "CRSArrTime", target_col]
                )
                X = df

                # same split logic (random per row)
                r = rng.random(len(df))
                train_mask = r < p_train
                val_mask = (r >= p_train) & (r < p_train + p_val)
                test_mask = r >= (p_train + p_val)

                # write (use compression everywhere so files are smaller)
                X[train_mask].to_hdf(
                    train_h5,
                    key="X",
                    mode="a",
                    format="table",
                    append=True,
                    index=False,
                    complevel=9,
                    complib="blosc",
                )
                y[train_mask].to_hdf(
                    train_h5,
                    key="y",
                    mode="a",
                    format="table",
                    append=True,
                    index=False,
                    complevel=9,
                    complib="blosc",
                )

                X[val_mask].to_hdf(
                    val_h5,
                    key="X",
                    mode="a",
                    format="table",
                    append=True,
                    index=False,
                    complevel=9,
                    complib="blosc",
                )
                y[val_mask].to_hdf(
                    val_h5,
                    key="y",
                    mode="a",
                    format="table",
                    append=True,
                    index=False,
                    complevel=9,
                    complib="blosc",
                )

                X[test_mask].to_hdf(
                    test_h5,
                    key="X",
                    mode="a",
                    format="table",
                    append=True,
                    index=False,
                    complevel=9,
                    complib="blosc",
                )
                y[test_mask].to_hdf(
                    test_h5,
                    key="y",
                    mode="a",
                    format="table",
                    append=True,
                    index=False,
                    complevel=9,
                    complib="blosc",
                )

                rows = []

        # leftover from chunking
        if rows:
            df = pd.DataFrame(rows, columns=cols)
            for c in numeric_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=[target_col])

            # encode categoricals (GLOBAL mapping)
            df["UniqueCarrier"] = df["UniqueCarrier"].map(carrier_to_id).fillna(0).astype("int32")
            df["Origin"] = df["Origin"].map(airport_to_id).fillna(0).astype("int32")
            df["Dest"]   = df["Dest"].map(airport_to_id).fillna(0).astype("int32")

            m = df["Month"]
            dow = df["DayOfWeek"]

            df["month_sin"] = np.sin(2 * np.pi * (m - 1) / 12)
            df["month_cos"] = np.cos(2 * np.pi * (m - 1) / 12)

            df["dow_sin"] = np.sin(2 * np.pi * (dow - 1) / 7)
            df["dow_cos"] = np.cos(2 * np.pi * (dow - 1) / 7)

            dep_min = hhmm_to_minutes(df["CRSDepTime"])
            arr_min = hhmm_to_minutes(df["CRSArrTime"])

            df["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440)
            df["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440)

            df["arr_sin"] = np.sin(2 * np.pi * arr_min / 1440)
            df["arr_cos"] = np.cos(2 * np.pi * arr_min / 1440)

            # DayofMonth assumes non-leap year for simplicity (and lack of year)
            MONTH_DAYS = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

            days_in_month = MONTH_DAYS[m.values.astype(int) - 1]
            dom_phase = 2 * np.pi * (df["DayofMonth"] - 1) / days_in_month
            df["dom_sin"] = np.sin(dom_phase)
            df["dom_cos"] = np.cos(dom_phase)

            df[z_cols] = x_scaler.transform(df[z_cols])
            y = pd.DataFrame(
                y_scaler.transform(df[[target_col]]),
                columns=[target_col],
                index=df.index,
            )

            df = df.drop(
                columns=["Month", "DayOfWeek", "CRSDepTime", "CRSArrTime", target_col]
            )
            X = df

            r = rng.random(len(df))
            train_mask = r < p_train
            val_mask = (r >= p_train) & (r < p_train + p_val)
            test_mask = r >= (p_train + p_val)

            X[train_mask].to_hdf(
                train_h5,
                key="X",
                mode="a",
                format="table",
                append=True,
                index=False,
                complevel=9,
                complib="blosc",
            )
            y[train_mask].to_hdf(
                train_h5,
                key="y",
                mode="a",
                format="table",
                append=True,
                index=False,
                complevel=9,
                complib="blosc",
            )

            X[val_mask].to_hdf(
                val_h5,
                key="X",
                mode="a",
                format="table",
                append=True,
                index=False,
                complevel=9,
                complib="blosc",
            )
            y[val_mask].to_hdf(
                val_h5,
                key="y",
                mode="a",
                format="table",
                append=True,
                index=False,
                complevel=9,
                complib="blosc",
            )

            X[test_mask].to_hdf(
                test_h5,
                key="X",
                mode="a",
                format="table",
                append=True,
                index=False,
                complevel=9,
                complib="blosc",
            )
            y[test_mask].to_hdf(
                test_h5,
                key="y",
                mode="a",
                format="table",
                append=True,
                index=False,
                complevel=9,
                complib="blosc",
            )


if __name__ == "__main__":
    DATA_PATH = os.path.abspath("./AIRLINES_10M.arff")
    print(DATA_PATH)

    arff_to_train_val_test_h5(
        arff_path=DATA_PATH,
        train_h5="airlines_train.h5",
        val_h5="airlines_val.h5",
        test_h5="airlines_test.h5",
        target_col="DepDelay",
        splits=(0.8, 0.1, 0.1),
        seed=42,
        chunksize=250_000,  # chunking so i don't run out of memory :')
    )

