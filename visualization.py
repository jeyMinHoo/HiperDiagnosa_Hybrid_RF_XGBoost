from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from config import FEATURES, REPORTS_DIR, TARGET
from model_utils import evaluate_model

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#333333",
    "axes.grid": True, "grid.color": "#e0e0e0", "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 120,
})
COLOR_RF = "#2E86AB"
COLOR_XGB = "#E67E22"
COLOR_HYBRID = "#27AE60"
COLOR_NEG = "#95A5A6"
COLOR_POS = "#C0392B"


def _save_fig(fig: plt.Figure, filename: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAFIK] Disimpan: {path}")
    return path


def plot_confusion_matrices(rf_metrics, xgb_metrics, hybrid_metrics, threshold):
    models = [("Random Forest", rf_metrics, COLOR_RF), ("XGBoost", xgb_metrics, COLOR_XGB),
              (f"Hybrid (thr={threshold:.2f})", hybrid_metrics, COLOR_HYBRID)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Confusion Matrix — Test Set (Unseen)", fontsize=14, fontweight="bold")
    for ax, (name, metrics, color) in zip(axes, models):
        matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
        ax.imshow(matrix, cmap="Blues", vmin=0)
        ax.set_title(name)
        ax.grid(False)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Prediksi\nTidak Berisiko", "Prediksi\nBerisiko"])
        ax.set_yticklabels(["Aktual\nTidak Berisiko", "Aktual\nBerisiko"])
        labels = [["TN", "FP"], ["FN", "TP"]]
        max_val = matrix.max()
        for i in range(2):
            for j in range(2):
                value = matrix[i, j]
                ax.text(j, i, f"{labels[i][j]}\n{value}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color="white" if value > max_val * 0.5 else "black")
    fig.tight_layout()
    return _save_fig(fig, "01_confusion_matrices.png")


def plot_roc_curves(y_test, rf_proba, xgb_proba, hybrid_proba):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba, color in [("Random Forest", rf_proba, COLOR_RF), ("XGBoost", xgb_proba, COLOR_XGB), ("Hybrid", hybrid_proba, COLOR_HYBRID)]:
        fpr, tpr, _ = metrics.roc_curve(y_test, proba)
        auc = metrics.roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)"); ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Kurva ROC — Test Set (Unseen)"); ax.legend(loc="lower right"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    return _save_fig(fig, "02_roc_curves.png")


def plot_feature_importance(hybrid_importance, rf_permutation_importance, xgb_permutation_importance):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sorted_items = sorted(hybrid_importance.items(), key=lambda item: item[1])
    feature_names = [item[0] for item in sorted_items]; values = [item[1] for item in sorted_items]
    ax1.barh(feature_names, values, color=COLOR_HYBRID); ax1.set_title("Feature Importance — Hybrid\n(impurity/gain-based)"); ax1.set_xlabel("Importance (proporsi)")
    for index, value in enumerate(values): ax1.text(value, index, f" {value * 100:.1f}%", va="center", fontsize=9)
    ordered = sorted(FEATURES, key=lambda feature: rf_permutation_importance[feature]["mean"] + xgb_permutation_importance[feature]["mean"])
    positions = np.arange(len(ordered)); height = 0.35
    rf_means = [rf_permutation_importance[f]["mean"] for f in ordered]; rf_stds = [rf_permutation_importance[f]["std"] for f in ordered]
    xgb_means = [xgb_permutation_importance[f]["mean"] for f in ordered]; xgb_stds = [xgb_permutation_importance[f]["std"] for f in ordered]
    ax2.barh(positions - height / 2, rf_means, height, xerr=rf_stds, color=COLOR_RF, label="Random Forest", capsize=3)
    ax2.barh(positions + height / 2, xgb_means, height, xerr=xgb_stds, color=COLOR_XGB, label="XGBoost", capsize=3)
    ax2.set_yticks(positions); ax2.set_yticklabels(ordered); ax2.axvline(0, color="#666666", linewidth=0.8)
    ax2.set_title("Permutation Importance — Validation Set\n(penurunan AUC saat fitur diacak)"); ax2.set_xlabel("Penurunan AUC (mean ± std)"); ax2.legend(loc="lower right")
    fig.tight_layout()
    return _save_fig(fig, "03_feature_importance.png")


def plot_weight_search(y_train, rf_oof_probability, xgb_oof_probability, best_weight):
    weights = np.arange(0.0, 1.01, 0.05)
    aucs = [metrics.roc_auc_score(y_train, weight * rf_oof_probability + (1 - weight) * xgb_oof_probability) for weight in weights]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(weights, aucs, color=COLOR_HYBRID, linewidth=2, marker="o", markersize=3)
    best_index = int(round(best_weight / 0.05)); ax.scatter([best_weight], [aucs[best_index]], color=COLOR_POS, s=90, zorder=5, label=f"Bobot optimal: RF={best_weight:.2f} / XGB={1-best_weight:.2f}")
    ax.set_xlabel("Bobot Random Forest (w)  —  Bobot XGBoost = 1 - w"); ax.set_ylabel("AUC gabungan (out-of-fold, training set)"); ax.set_title("Pencarian Bobot Soft-Voting Optimal"); ax.legend(loc="lower center"); ax.set_xlim(-0.02, 1.02)
    fig.tight_layout(); return _save_fig(fig, "04_weight_search.png")


def plot_threshold_search(y_val, hybrid_val_probability, best_threshold):
    thresholds = np.arange(0.05, 1.0, 0.01); metric_values = {key: [] for key in ("precision", "recall", "specificity", "npv")}
    for threshold in thresholds:
        metrics = evaluate_model(y_val, hybrid_val_probability, threshold)
        for key in metric_values: metric_values[key].append(metrics[key])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key, label, color in [("precision", "Precision (PPV)", COLOR_POS), ("recall", "Recall (Sensitivity)", COLOR_HYBRID), ("specificity", "Specificity", COLOR_RF), ("npv", "NPV", COLOR_XGB)]: ax.plot(thresholds, metric_values[key], label=label, color=color, linewidth=1.8)
    ax.axvline(best_threshold, color="#333333", linestyle="--", linewidth=1.3, label=f"Threshold terpilih ({best_threshold:.2f})"); ax.set_xlabel("Threshold"); ax.set_ylabel("Skor"); ax.set_title("Trade-off Metrik vs Threshold — Validation Set"); ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5)); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout(); return _save_fig(fig, "05_threshold_search.png")


def plot_cv_comparison(cv_summary):
    if not cv_summary: return None
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]; labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    rf_means = [cv_summary["random_forest"][m]["mean"] for m in metrics]; rf_stds = [cv_summary["random_forest"][m]["std"] for m in metrics]
    xgb_means = [cv_summary["xgboost"][m]["mean"] for m in metrics]; xgb_stds = [cv_summary["xgboost"][m]["std"] for m in metrics]
    x = np.arange(len(metrics)); width = 0.35; fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, rf_means, width, yerr=rf_stds, label="Random Forest", color=COLOR_RF, capsize=4); ax.bar(x + width / 2, xgb_means, width, yerr=xgb_stds, label="XGBoost", color=COLOR_XGB, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Skor"); ax.set_title("5-Fold Cross-Validation — Training Set (mean ± std)"); ax.legend(); ax.set_ylim(0, 1.05)
    for index, (rf_mean, xgb_mean) in enumerate(zip(rf_means, xgb_means)): ax.text(index - width / 2, rf_mean + 0.02, f"{rf_mean:.3f}", ha="center", fontsize=8); ax.text(index + width / 2, xgb_mean + 0.02, f"{xgb_mean:.3f}", ha="center", fontsize=8)
    fig.tight_layout(); return _save_fig(fig, "06_cross_validation_comparison.png")


def plot_class_distribution(df):
    counts = df[TARGET].value_counts().sort_index(); labels = ["Tidak Hipertensi (0)", "Hipertensi (1)"]; colors = [COLOR_NEG, COLOR_POS]
    fig, ax = plt.subplots(figsize=(6, 5)); bars = ax.bar(labels, counts.values, color=colors); ax.set_title("Distribusi Kelas Target — Seluruh Dataset"); ax.set_ylabel("Jumlah Sampel")
    for bar, count in zip(bars, counts.values): ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count}\n({count / counts.sum() * 100:.1f}%)", ha="center", va="bottom", fontsize=10)
    fig.tight_layout(); return _save_fig(fig, "00_class_distribution.png")


def generate_html_report(metrics, image_paths):
    def metric_row(label, key, fmt="{:.4f}"):
        return f"<tr><td>{label}</td><td>{fmt.format(metrics['rf'][key])}</td><td>{fmt.format(metrics['xgb'][key])}</td><td><b>{fmt.format(metrics['hybrid'][key])}</b></td></tr>"
    rows = "".join(metric_row(label, key) for label, key in [("Accuracy", "accuracy"), ("Precision (PPV)", "precision"), ("NPV", "npv"), ("Recall (Sensitivity)", "recall"), ("Specificity", "specificity"), ("F1-Score", "f1"), ("AUC", "auc"), ("Youden's J", "youden_j")])
    def img_tag(key, caption):
        return "" if key not in image_paths or image_paths[key] is None else f'<div class="chart"><h3>{caption}</h3><img src="{image_paths[key].name}" alt="{caption}"></div>'
    html = f'''<!DOCTYPE html><html lang="id"><head><meta charset="UTF-8"><title>Laporan Training — HiperDiagnosa Hybrid RF+XGBoost</title><style>body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 32px; background: #f4f6f8; color: #222; }} h1 {{ color: #1a1a2e; }} h2 {{ margin-top: 40px; border-bottom: 2px solid #ddd; padding-bottom: 6px; }} .meta {{ color: #666; font-size: 14px; margin-bottom: 24px; }} .badge {{ display: inline-block; background: #27AE60; color: white; padding: 3px 10px; border-radius: 12px; font-size: 12px; margin-right: 6px; }} table {{ border-collapse: collapse; width: 100%; max-width: 700px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }} th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; }} th {{ background: #1a1a2e; color: white; }} tr:hover {{ background: #f9f9f9; }} .chart {{ background: white; padding: 16px; margin-bottom: 24px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); max-width: 1000px; }} .chart img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }} .disclaimer {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 14px 18px; margin: 20px 0; max-width: 700px; font-size: 14px; }}</style></head><body><h1>Laporan Training — HiperDiagnosa</h1><div class="meta"><span class="badge">{metrics['algorithm']}</span><span class="badge">{metrics['ensemble_method']}</span><br><br>Bobot ensemble: RF={metrics['rf_weight']:.2f} / XGB={metrics['xgb_weight']:.2f} &middot; Threshold optimal: {metrics['optimal_threshold']:.2f} &middot; Split: {metrics['split']}<br>Dataset: {metrics['dataset_total']} sampel (train={metrics['train_total']}, val={metrics['validation_total']}, test={metrics['test_total']})</div><div class="disclaimer">⚠️ Ini laporan performa MODEL STATISTIK untuk alat skrining, bukan validasi klinis. Semua metrik di atas dihitung dari test set yang belum pernah dilihat model selama training/tuning.</div><h2>Ringkasan Metrik (Test Set)</h2><table><tr><th>Metrik</th><th>Random Forest</th><th>XGBoost</th><th>Hybrid</th></tr>{rows}</table><h2>Distribusi Data</h2>{img_tag('class_distribution', 'Distribusi Kelas Target')}<h2>Perbandingan Performa</h2>{img_tag('confusion_matrices', 'Confusion Matrix (Test Set)')}{img_tag('roc_curves', 'Kurva ROC (Test Set)')}{img_tag('cv_comparison', '5-Fold Cross-Validation (Training Set)')}<h2>Proses Tuning</h2>{img_tag('weight_search', 'Pencarian Bobot Soft-Voting')}{img_tag('threshold_search', 'Pencarian Threshold')}<h2>Feature Importance</h2>{img_tag('feature_importance', 'Feature Importance (Impurity/Gain vs Permutation)')}</body></html>'''
    REPORTS_DIR.mkdir(parents=True, exist_ok=True); report_path = REPORTS_DIR / "summary_report.html"; report_path.write_text(html, encoding="utf-8"); print(f"  [LAPORAN] Disimpan: {report_path}"); return report_path
