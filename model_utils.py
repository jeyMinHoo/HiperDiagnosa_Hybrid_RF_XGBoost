import numpy as np
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from config import FEATURES


def compute_scale_pos_weight(y_train) -> float:
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    if positive == 0:
        raise ValueError("Data training tidak memiliki kelas positif (1).")
    return negative / positive


def build_models(scale_pos_weight: float, use_smote: bool = False):
    rf_steps = [("imputer", SimpleImputer(strategy="median"))]
    xgb_steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_smote:
        rf_steps.append(("smote", SMOTE(random_state=42, k_neighbors=5)))
        xgb_steps.append(("smote", SMOTE(random_state=42, k_neighbors=5)))
    rf_steps.append(("model", RandomForestClassifier(
        n_estimators=200, max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )))
    xgb_steps.append(("model", XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0 if use_smote else scale_pos_weight,
        objective="binary:logistic", eval_metric="logloss", random_state=42,
        n_jobs=-1, tree_method="hist",
    )))
    return Pipeline(rf_steps), Pipeline(xgb_steps)


def run_cross_validation(model, X_train, y_train, model_name, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    print(f"\n[CV] {model_name} — {n_splits}-fold Stratified Cross-Validation (data training)")
    print("-" * 65)
    summary = {}
    for metric in scoring:
        scores = results[f"test_{metric}"]
        summary[metric] = {"mean": float(scores.mean()), "std": float(scores.std())}
        print(f"  {metric:<10}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return summary


def get_oof_probabilities(model, X_train, y_train, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def find_optimal_weight(y_train, rf_oof_probability, xgb_oof_probability, objective="auc"):
    weights = np.arange(0.0, 1.01, 0.05)
    best_score, best_weight = -1, 0.5
    print("\n[INFO] Mencari bobot ensemble optimal via OUT-OF-FOLD prediction di TRAINING SET "
          f"(objective={objective})...")
    print("Bobot RF | Bobot XGB | Skor")
    print("-" * 35)
    for weight in weights:
        hybrid_oof = weight * rf_oof_probability + (1 - weight) * xgb_oof_probability
        score = (sklearn.metrics.roc_auc_score(y_train, hybrid_oof)
                 if objective == "auc" else evaluate_model(y_train, hybrid_oof, threshold=0.5)[objective])
        print(f"  {weight:.2f}   |   {1 - weight:.2f}    | {score:.4f}")
        if score > best_score:
            best_score, best_weight = score, float(weight)
    print(f"\n✓ Bobot optimal: RF={best_weight:.2f} / XGB={1 - best_weight:.2f} ({objective}={best_score:.4f})")
    if best_weight in (0.0, 1.0):
        dominant = "Random Forest" if best_weight == 1.0 else "XGBoost"
        print("  [PERHATIAN] Bobot mendarat di titik ekstrem -- ensemble")
        print(f"  efektif menjadi {dominant} saja. Ini VALID kalau salah satu")
        print("  model memang jelas lebih baik, tapi kalau CV summary (bagian")
        print("  cross_validation_on_train di metrics.json) menunjukkan kedua")
        print("  model berdekatan, titik ekstrem ini kemungkinan noise dari")
        print("  data yang terbatas -- pertimbangkan pakai lebih banyak data,")
        print("  atau cek stabilitasnya dengan mengubah random_state.")
    return best_weight, float(best_score)


def evaluate_model(y_true, probability, threshold=0.5):
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    return {
        "accuracy": float(sklearn.metrics.accuracy_score(y_true, prediction)),
        "precision": float(precision), "ppv": float(precision), "npv": float(npv),
        "recall": float(recall), "f1": float(f1),
        "auc": float(sklearn.metrics.roc_auc_score(y_true, probability)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "specificity": float(specificity), "sensitivity": float(recall),
        "youden_j": float(recall + specificity - 1),
    }


def find_optimal_threshold(y_val, probability_val, metric="youden", fn_cost=1.0, fp_cost=1.0, min_precision=None):
    thresholds = np.arange(0.05, 1.0, 0.01)
    best_score, best_threshold, best_metrics = None, 0.5, None
    print(f"\n[INFO] Mencari threshold optimal di VALIDATION SET (metric={metric}"
          + (f", fn_cost={fn_cost}, fp_cost={fp_cost}" if metric == "cost" else "")
          + (f", min_precision={min_precision}" if min_precision else "") + ")...")
    for threshold in thresholds:
        metrics = evaluate_model(y_val, probability_val, threshold)
        if min_precision is not None and metrics["precision"] < min_precision:
            continue
        if metric == "f1":
            score = metrics["f1"]
        elif metric == "recall":
            score = metrics["recall"]
        elif metric == "youden":
            score = metrics["youden_j"]
        elif metric == "cost":
            score = -(fn_cost * metrics["fn"] + fp_cost * metrics["fp"])
        else:
            raise ValueError(f"Metric tidak dikenal: {metric}")
        if best_score is None or score > best_score:
            best_score, best_threshold, best_metrics = score, threshold, metrics
    if best_metrics is None:
        raise ValueError("Tidak ada threshold yang memenuhi syarat min_precision "
                         f"({min_precision}). Turunkan syarat atau perbaiki model.")
    print(f"✓ Threshold optimal (di validation set): {best_threshold:.2f} (score={best_score:.4f})")
    print(f"  Validation -> precision={best_metrics['precision']:.4f}, npv={best_metrics['npv']:.4f}, "
          f"recall={best_metrics['recall']:.4f}, specificity={best_metrics['specificity']:.4f}")
    return float(best_threshold), best_metrics


def get_feature_importance(model):
    return dict(zip(FEATURES, model.named_steps["model"].feature_importances_.tolist()))


def sort_importance(data):
    return dict(sorted(data.items(), key=lambda item: item[1], reverse=True))


def report_feature_importance(sorted_importance, low_importance_threshold=0.03):
    total = sum(sorted_importance.values()) or 1.0
    cumulative, rows, low_importance_features = 0.0, [], []
    for feature, importance in sorted_importance.items():
        share = importance / total
        cumulative += share
        rows.append((feature, importance, share, cumulative))
        if share < low_importance_threshold:
            low_importance_features.append(feature)
    print(f"\n{'Fitur':<25} {'Importance':>10} {'Kontribusi':>11} {'Kumulatif':>10}")
    print("-" * 60)
    for feature, importance, share, cumulative in rows:
        flag = "  <- kontribusi kecil" if share < low_importance_threshold else ""
        print(f"{feature:<25} {importance:>10.4f} {share * 100:>10.2f}% {cumulative * 100:>9.2f}%{flag}")
    if low_importance_features:
        print(f"\n[CATATAN] Fitur dengan kontribusi < {low_importance_threshold * 100:.0f}%: {', '.join(low_importance_features)}")
        print("          AUC yang stuck di ~0.73-0.74 sering menandakan batasan ada")
        print("          di fitur/data, bukan di algoritma. Pertimbangkan: cek fitur")
        print("          klinis lain yang belum tercakup (mis. lingkar pinggang,")
        print("          konsumsi garam, aktivitas fisik), atau buat fitur turunan")
        print("          (mis. rasio sistolik/diastolik, kategori IMT).")
    return {"low_importance_threshold": low_importance_threshold, "low_importance_features": low_importance_features}


def get_permutation_importance(model, X_val, y_val, scoring="roc_auc", n_repeats=10):
    result = permutation_importance(model, X_val, y_val, scoring=scoring, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    return {feature: {"mean": float(mean), "std": float(std)}
            for feature, mean, std in zip(FEATURES, result.importances_mean, result.importances_std)}


def print_metrics(model_name, metrics, threshold=0.5):
    print(f"\n{model_name} (threshold={threshold:.2f})")
    print("-" * 65)
    for label, key in [("Accuracy", "accuracy"), ("Precision(PPV)", "precision"), ("NPV", "npv"),
                       ("Recall", "recall"), ("Sensitivity", "sensitivity"), ("Specificity", "specificity"),
                       ("F1-Score", "f1"), ("AUC", "auc"), ("Youden's J", "youden_j")]:
        print(f"{label:<14}: {metrics[key]:.4f}")
    print("\nConfusion Matrix")
    for label, key in [("True Negative", "tn"), ("False Positive", "fp"), ("False Negative", "fn"), ("True Positive", "tp")]:
        print(f"  {label:<15}: {metrics[key]}")
