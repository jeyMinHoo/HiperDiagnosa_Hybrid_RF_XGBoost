import json

import joblib
from sklearn.model_selection import train_test_split

from config import (
    FEATURES,
    METRICS_PATH,
    RF_MODEL_PATH,
    REPORTS_DIR,
    TARGET,
    XGB_MODEL_PATH,
)
from data_processing import load_dataset
from model_utils import (
    build_models,
    compute_scale_pos_weight,
    evaluate_model,
    find_optimal_threshold,
    find_optimal_weight,
    get_feature_importance,
    get_oof_probabilities,
    get_permutation_importance,
    print_metrics,
    report_feature_importance,
    run_cross_validation,
    sort_importance,
)
from visualization import (
    generate_html_report,
    plot_class_distribution,
    plot_confusion_matrices,
    plot_cv_comparison,
    plot_feature_importance,
    plot_roc_curves,
    plot_threshold_search,
    plot_weight_search,
)

# Re-export the existing public helpers for callers that imported them here.
__all__ = [
    "load_dataset", "compute_scale_pos_weight", "build_models",
    "run_cross_validation", "get_oof_probabilities", "find_optimal_weight",
    "evaluate_model", "find_optimal_threshold", "get_feature_importance",
    "sort_importance", "report_feature_importance",
    "get_permutation_importance", "print_metrics", "train",
]


def train(
    use_smote=False,
    threshold_metric="youden",
    run_cv=True,
    tune_weight=True,
    weight_objective="auc",
    fn_cost=1.0,
    fp_cost=1.0,
    min_precision=None,
    low_importance_threshold=0.03,
):
    """Train, evaluate, save, and report the hybrid RF/XGBoost system."""
    print("\n")
    print("=" * 65)
    print("       HYBRID RANDOM FOREST + XGBOOST")
    print("          SISTEM DETEKSI HIPERTENSI ESENSIAL")
    print("=" * 65)

    df = load_dataset()
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp,
    )

    print("\n" + "=" * 65)
    print("                  PEMBAGIAN DATA")
    print("=" * 65)
    print(f"Data training   : {len(X_train)} (60%)")
    print(f"Data validation : {len(X_val)}  (20%) -> untuk threshold tuning")
    print(f"Data testing    : {len(X_test)} (20%) -> HANYA untuk evaluasi akhir")
    print("Random state    : 42")

    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"\nscale_pos_weight (dihitung dari y_train) : {scale_pos_weight:.4f}")
    random_forest, xgboost = build_models(scale_pos_weight, use_smote=use_smote)

    cv_summary = {}
    if run_cv:
        print("\n" + "=" * 65)
        print("     CROSS-VALIDATION (ESTIMASI PERFORMA DI TRAINING SET)")
        print("=" * 65)
        cv_summary["random_forest"] = run_cross_validation(
            random_forest, X_train, y_train, "Random Forest"
        )
        cv_summary["xgboost"] = run_cross_validation(
            xgboost, X_train, y_train, "XGBoost"
        )

    print("\n" + "=" * 65)
    print("              PROSES TRAINING MODEL FINAL")
    print("=" * 65)
    print("\n[1/2] Training Random Forest...")
    random_forest.fit(X_train, y_train)
    print("      OK Random Forest selesai.")
    print("\n[2/2] Training XGBoost...")
    xgboost.fit(X_train, y_train)
    print("      OK XGBoost selesai.")

    rf_val_probability = random_forest.predict_proba(X_val)[:, 1]
    xgb_val_probability = xgboost.predict_proba(X_val)[:, 1]

    if tune_weight:
        print("\n" + "=" * 65)
        print("     TAHAP 1 — TUNING BOBOT ENSEMBLE (VIA OOF, TRAINING SET)")
        print("=" * 65)
        rf_oof_probability = get_oof_probabilities(random_forest, X_train, y_train)
        xgb_oof_probability = get_oof_probabilities(xgboost, X_train, y_train)
        rf_weight, weight_score = find_optimal_weight(
            y_train, rf_oof_probability, xgb_oof_probability,
            objective=weight_objective,
        )
    else:
        rf_weight, weight_score = 0.5, None

    xgb_weight = 1 - rf_weight
    hybrid_val_probability = (
        rf_weight * rf_val_probability + xgb_weight * xgb_val_probability
    )

    print("\n" + "=" * 65)
    print("     TAHAP 2 — THRESHOLD OPTIMIZATION (DI VALIDATION SET)")
    print("=" * 65)
    optimal_threshold, _ = find_optimal_threshold(
        y_val, hybrid_val_probability, metric=threshold_metric,
        fn_cost=fn_cost, fp_cost=fp_cost, min_precision=min_precision,
    )

    rf_test_probability = random_forest.predict_proba(X_test)[:, 1]
    xgb_test_probability = xgboost.predict_proba(X_test)[:, 1]
    hybrid_test_probability = (
        rf_weight * rf_test_probability + xgb_weight * xgb_test_probability
    )
    rf_metrics = evaluate_model(y_test, rf_test_probability, optimal_threshold)
    xgb_metrics = evaluate_model(y_test, xgb_test_probability, optimal_threshold)
    hybrid_metrics = evaluate_model(y_test, hybrid_test_probability, optimal_threshold)

    rf_importance = get_feature_importance(random_forest)
    xgb_importance = get_feature_importance(xgboost)
    hybrid_importance = {
        feature: rf_weight * rf_importance[feature]
        + xgb_weight * xgb_importance[feature]
        for feature in FEATURES
    }
    print("\n" + "=" * 65)
    print("     TAHAP 3 — LAPORAN FEATURE IMPORTANCE (HYBRID)")
    print("=" * 65)
    importance_report = report_feature_importance(
        sort_importance(hybrid_importance), low_importance_threshold,
    )

    print("\n[INFO] Menghitung permutation importance (cross-check, validation set)...")
    rf_permutation_importance = get_permutation_importance(
        random_forest, X_val, y_val,
    )
    xgb_permutation_importance = get_permutation_importance(
        xgboost, X_val, y_val,
    )

    metrics = {
        "algorithm": "Hybrid Random Forest + XGBoost",
        "ensemble_method": (
            "Soft Voting (bobot di-tuning di validation set)"
            if tune_weight else "Soft Voting 50:50 (fixed)"
        ),
        "rf_weight": float(rf_weight), "xgb_weight": float(xgb_weight),
        "weight_objective": weight_objective if tune_weight else None,
        "weight_tuning_method": "out_of_fold_train" if tune_weight else None,
        "weight_tuning_score_oof_train": weight_score,
        "threshold_metric": threshold_metric,
        "threshold_cost": (
            {"fn_cost": fn_cost, "fp_cost": fp_cost}
            if threshold_metric == "cost" else None
        ),
        "threshold_min_precision": min_precision,
        "optimal_threshold": float(optimal_threshold),
        "threshold_selected_on": "validation_set", "used_smote": use_smote,
        "scale_pos_weight_train": float(scale_pos_weight),
        "dataset_total": len(df), "train_total": len(X_train),
        "validation_total": len(X_val), "test_total": len(X_test),
        "positive_total": int(y.sum()), "negative_total": int((y == 0).sum()),
        "random_state": 42,
        "split": "60/20/20 (train/validation/test), stratified",
        "cross_validation_on_train": cv_summary,
        "rf": rf_metrics, "xgb": xgb_metrics, "hybrid": hybrid_metrics,
        "feature_importance": {
            "random_forest": sort_importance(rf_importance),
            "xgboost": sort_importance(xgb_importance),
            "hybrid_weighted": sort_importance(hybrid_importance),
            "low_importance_report": importance_report,
            "permutation_importance_validation_set": {
                "note": "Cross-check terhadap importance impurity/gain-based di atas "
                        "dari penurunan AUC saat 1 kolom fitur diacak.",
                "scoring": "roc_auc",
                "random_forest": dict(sorted(
                    rf_permutation_importance.items(),
                    key=lambda item: item[1]["mean"], reverse=True,
                )),
                "xgboost": dict(sorted(
                    xgb_permutation_importance.items(),
                    key=lambda item: item[1]["mean"], reverse=True,
                )),
            },
        },
    }

    RF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(random_forest, RF_MODEL_PATH)
    joblib.dump(xgboost, XGB_MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 65)
    print("     TAHAP 5 — MEMBUAT GRAFIK & LAPORAN VISUAL")
    print("=" * 65)
    image_paths = {}
    try:
        image_paths["class_distribution"] = plot_class_distribution(df)
        image_paths["confusion_matrices"] = plot_confusion_matrices(
            rf_metrics, xgb_metrics, hybrid_metrics, optimal_threshold,
        )
        image_paths["roc_curves"] = plot_roc_curves(
            y_test, rf_test_probability, xgb_test_probability, hybrid_test_probability,
        )
        image_paths["feature_importance"] = plot_feature_importance(
            hybrid_importance, rf_permutation_importance, xgb_permutation_importance,
        )
        image_paths["weight_search"] = (
            plot_weight_search(y_train, rf_oof_probability, xgb_oof_probability, rf_weight)
            if tune_weight else None
        )
        image_paths["threshold_search"] = plot_threshold_search(
            y_val, hybrid_val_probability, optimal_threshold,
        )
        image_paths["cv_comparison"] = (
            plot_cv_comparison(cv_summary) if run_cv else None
        )
        generate_html_report(metrics, image_paths)
    except Exception as error:
        print(f"  [PERINGATAN] Gagal membuat sebagian/semua visualisasi: {error}")
        print("  Model & metrics.json tetap tersimpan dengan benar.")

    print("\n\n" + "=" * 65)
    print("       HASIL EVALUASI AKHIR (TEST SET, unseen)")
    print("=" * 65)
    print_metrics("RANDOM FOREST", rf_metrics, optimal_threshold)
    print_metrics("XGBOOST", xgb_metrics, optimal_threshold)
    print_metrics(
        f"HYBRID RF({rf_weight:.2f}) + XGB({xgb_weight:.2f})",
        hybrid_metrics, optimal_threshold,
    )
    print("\n" + "=" * 65)
    print("                  FILE TERSIMPAN")
    print("=" * 65)
    print(f"Random Forest : {RF_MODEL_PATH}")
    print(f"XGBoost       : {XGB_MODEL_PATH}")
    print(f"Metrics       : {METRICS_PATH}")
    print(f"Grafik/Laporan: {REPORTS_DIR}/ (buka summary_report.html di browser)")
    print("\n" + "=" * 65)
    print("             TRAINING SELESAI")
    print("=" * 65)
    return metrics


if __name__ == "__main__":
    train(
        use_smote=False,
        threshold_metric="youden",
        run_cv=True,
        tune_weight=False,
    )
