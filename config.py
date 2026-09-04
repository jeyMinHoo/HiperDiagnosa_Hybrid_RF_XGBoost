from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
RF_MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"
XGB_MODEL_PATH = BASE_DIR / "models" / "xgboost.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
REPORTS_DIR = BASE_DIR / "reports"

FEATURES = [
    "usia",
    "jenis_kelamin",
    "imt",
    "riwayat_keluarga",
    "merokok",
    "sistolik",
    "diastolik",
]
TARGET = "hipertensi"

GENDER_MAP = {
    "l": 1, "laki-laki": 1, "laki laki": 1, "male": 1, "m": 1,
    "p": 0, "perempuan": 0, "female": 0, "f": 0,
}

YES_NO_MAP = {
    "ya": 1, "iya": 1, "yes": 1, "true": 1, "1": 1,
    "tidak": 0, "no": 0, "false": 0, "0": 0,
}
