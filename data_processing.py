import numpy as np
import pandas as pd

from config import DATA_PATH, FEATURES, TARGET, YES_NO_MAP, GENDER_MAP


def normalize_category(value, mapping):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in mapping:
        return mapping[text]
    try:
        number = float(text)
        if number in (0, 1):
            return number
    except (ValueError, TypeError):
        pass
    return np.nan


def load_dataset():
    """Membaca dan melakukan preprocessing dataset."""
    path = DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan.\n"
            f"Silakan letakkan dataset di:\n{DATA_PATH}"
        )

    print("\n" + "=" * 65)
    print("                 PEMUATAN DATASET")
    print("=" * 65)
    print(f"Dataset : {path}")

    df = pd.read_csv(path)

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed > 0:
        raise ValueError(
            f"Dataset penelitian mengandung {n_removed} duplikat. "
            "Gunakan dataset hasil penyaringan."
        )

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Kolom dataset tidak lengkap: " + ", ".join(missing_columns)
        )

    binary_columns = ["jenis_kelamin", "riwayat_keluarga", "merokok"]
    for column in binary_columns:
        mapping = GENDER_MAP if column == "jenis_kelamin" else YES_NO_MAP
        df[column] = df[column].apply(
            lambda value: normalize_category(value, mapping)
        )

    numeric_columns = ["usia", "imt", "sistolik", "diastolik", TARGET]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[TARGET] = df[TARGET].apply(lambda v: normalize_category(v, YES_NO_MAP))
    df = df.dropna(subset=[TARGET])

    if df[FEATURES + [TARGET]].isna().any().any():
        missing = df[FEATURES + [TARGET]].isna().sum()
        raise ValueError(
            "Dataset memiliki nilai kosong/tidak valid:\n"
            + missing[missing > 0].to_string()
        )

    ranges = {
        "usia": (18, 90),
        "imt": (10, 70),
        "sistolik": (70, 250),
        "diastolik": (30, 150),
    }
    for column, (lo, hi) in ranges.items():
        if not df[column].between(lo, hi).all():
            raise ValueError(
                f"Kolom {column} memiliki nilai di luar rentang {lo}-{hi}."
            )

    if not (df["sistolik"] > df["diastolik"]).all():
        raise ValueError("Semua data harus memenuhi sistolik > diastolik.")

    if df[TARGET].nunique() < 2:
        raise ValueError("Target hipertensi harus memiliki kelas 0 dan 1.")

    print(f"Jumlah data        : {len(df)}")
    print(f"Jumlah fitur       : {len(FEATURES)}")
    print(f"Target             : {TARGET}")
    print(f"Kelas 0            : {(df[TARGET] == 0).sum()}")
    print(f"Kelas 1            : {(df[TARGET] == 1).sum()}")
    imbalance_ratio = (df[TARGET] == 0).sum() / (df[TARGET] == 1).sum()
    print(f"Ratio imbalance    : {imbalance_ratio:.2f}:1")

    return df