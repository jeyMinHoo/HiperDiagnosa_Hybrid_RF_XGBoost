"""
Flask Application untuk HiperDiagnosa
Improved version dengan better code quality & error handling
"""

import json
import logging
from typing import Dict, Tuple, Optional
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from config import (
    RF_MODEL_PATH,
    XGB_MODEL_PATH,
    METRICS_PATH,
    FEATURES,
    GENDER_MAP,
    YES_NO_MAP,
)

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# FLASK APP INITIALIZATION
# ============================================================

app = Flask(__name__)

# Model caching
_rf_model = None
_xgb_model = None
_metrics_cache = None

# ============================================================
# MODEL LOADING
# ============================================================

def load_models() -> Tuple[Optional[object], Optional[object]]:
    """
    Load trained Random Forest dan XGBoost models.
    
    Returns:
        Tuple[rf_model, xgb_model]: Loaded models or (None, None) if not found
        
    Raises:
        RuntimeError: Jika model file tidak ditemukan
    """
    global _rf_model, _xgb_model
    
    if _rf_model is not None and _xgb_model is not None:
        return _rf_model, _xgb_model
    
    if not RF_MODEL_PATH.exists() or not XGB_MODEL_PATH.exists():
        logger.error(f"Model files not found: RF={RF_MODEL_PATH}, XGB={XGB_MODEL_PATH}")
        return None, None
    
    try:
        _rf_model = joblib.load(RF_MODEL_PATH)
        _xgb_model = joblib.load(XGB_MODEL_PATH)
        logger.info("Models loaded successfully")
        return _rf_model, _xgb_model
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return None, None


def load_metrics() -> Optional[Dict]:
    """
    Load model metrics from JSON file.
    
    Returns:
        Dict: Metrics data or None if not found
    """
    global _metrics_cache
    
    if _metrics_cache is not None:
        return _metrics_cache
    
    if not METRICS_PATH.exists():
        logger.warning("Metrics file not found")
        return None
    
    try:
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            _metrics_cache = json.load(f)
        logger.info("Metrics loaded successfully")
        return _metrics_cache
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}")
        return None

# ============================================================
# INPUT PROCESSING
# ============================================================

def normalize_input(value, mapping, field_name):
    text = str(value).strip().lower()
    if text in mapping:
        return float(mapping[text])
    try:
        number = float(text)
        if number in (0, 1):
            return number
    except (ValueError, TypeError):
        pass
    raise ValueError(f"{field_name} tidak valid")

def normalize_gender_input(value):
    return normalize_input(value, GENDER_MAP, "Jenis kelamin")

def normalize_yes_no_input(value, field_name):
    return normalize_input(value, YES_NO_MAP, field_name)


def validate_numeric_input(
    name: str,
    value: float,
    min_val: float,
    max_val: float,
) -> float:
    """
    Validate numeric input range.
    
    Args:
        name: Field name for error message
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        float: Validated value
        
    Raises:
        ValueError: Jika value di luar range
    """
    try:
        num = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{name} harus berupa angka")
    
    if not (min_val <= num <= max_val):
        raise ValueError(
            f"{name} harus antara {min_val} dan {max_val}, "
            f"tetapi didapat {num}"
        )
    
    return num


def validate_input_data(data: Dict) -> Dict:
    """
    Validate and normalize input data.
    
    Args:
        data: Dictionary berisi input dari user
        
    Returns:
        Dict: Validated dan normalized input
        
    Raises:
        ValueError: Jika ada input yang tidak valid
    """
    errors = []
    
    # Required fields
    required_fields = FEATURES
    for field in required_fields:
        if field not in data:
            errors.append(f"Field '{field}' harus disediakan")
    
    if errors:
        raise ValueError("; ".join(errors))
    
    try:
        # Numeric fields dengan validation range
        # Batas bawah 18: ambang diagnostik tekanan darah untuk anak/remaja
        # berbeda (berbasis persentil usia-tinggi-jenis kelamin, bukan angka
        # mmHg tetap seperti pada dewasa). Alat ini hanya untuk skrining dewasa.
        usia = validate_numeric_input("usia", data["usia"], 18, 90)
        imt = validate_numeric_input("imt", data["imt"], 10, 70)
        sistolik = validate_numeric_input("sistolik", data["sistolik"], 70, 250)
        diastolik = validate_numeric_input("diastolik", data["diastolik"], 30, 150)
        
        # Binary fields
        jenis_kelamin = normalize_gender_input(data["jenis_kelamin"])
        riwayat_keluarga = normalize_yes_no_input(
            data["riwayat_keluarga"],
            "Riwayat keluarga",
        )
        merokok = normalize_yes_no_input(data["merokok"], "Merokok")

        if sistolik <= diastolik:
            raise ValueError("Sistolik harus lebih besar daripada diastolik")
        
        return {
            "usia": usia,
            "jenis_kelamin": jenis_kelamin,
            "imt": imt,
            "riwayat_keluarga": riwayat_keluarga,
            "merokok": merokok,
            "sistolik": sistolik,
            "diastolik": diastolik,
        }
    
    except ValueError as e:
        raise ValueError(f"Input validation error: {str(e)}")

# ============================================================
# PREDICTION HELPERS
# ============================================================

def categorize_risk(probability: float) -> str:
    """
    Categorize hypertension risk based on probability.
    
    Args:
        probability: Predicted probability (0-1)
        
    Returns:
        str: Risk category
    """
    if probability >= 0.75:
        return "Tinggi"
    elif probability >= 0.50:
        return "Sedang"
    else:
        return "Rendah"


def categorize_blood_pressure(
    sistolik: float,
    diastolik: float,
) -> str:
    """
    Categorize blood pressure based on readings, mengikuti pedoman
    ACC/AHA 2017 secara lengkap (versi sebelumnya melewatkan kategori
    "Elevated" sehingga sistolik 120-129 dengan diastolik <80 salah
    tergolong "Normal").

    Args:
        sistolik: Systolic blood pressure
        diastolik: Diastolic blood pressure

    Returns:
        str: Blood pressure category

    References (ACC/AHA 2017 Hypertension Guideline):
        - Normal: < 120 dan < 80
        - Elevated: 120-129 dan < 80
        - Stage 1 HTN: 130-139 atau 80-89
        - Stage 2 HTN: >= 140 atau >= 90
        - Hypertensive Crisis: >= 180 dan/atau >= 120

    Catatan: Kategori ini dihitung dari SATU kali input tekanan darah.
    Diagnosis hipertensi klinis sebenarnya mensyaratkan rata-rata dari
    >=2 pengukuran pada >=2 kesempatan berbeda -- lihat disclaimer di
    endpoint /api/diagnose.
    """
    if sistolik >= 180 or diastolik >= 120:
        return "Sangat Tinggi (Hypertensive Crisis)"
    elif sistolik >= 140 or diastolik >= 90:
        return "Tinggi (Stage 2 Hypertension)"
    elif sistolik >= 130 or diastolik >= 80:
        return "Meningkat (Stage 1 Hypertension)"
    elif sistolik >= 120 and diastolik < 80:
        return "Waspada (Elevated)"
    else:
        return "Normal"


def is_hypertensive_crisis(pressure_category: str) -> bool:
    """
    Menandai kondisi darurat medis berdasarkan kategori tekanan darah
    saja (independen dari skor model ML). Dipakai sebagai safety
    override di endpoint /api/diagnose -- lihat penjelasan di sana.
    """
    return pressure_category == "Sangat Tinggi (Hypertensive Crisis)"

# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def index():
    """Render main page with metrics."""
    try:
        metrics = load_metrics()
        return render_template("index.html", metrics=metrics)
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return f"Error loading page: {str(e)}", 500


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """
    GET /api/metrics
    
    Returns model metrics as JSON.
    
    Response (200):
        {
            "success": true,
            "data": {...metrics...}
        }
        
    Response (404):
        {
            "success": false,
            "error": "Model belum dilatih."
        }
    """
    try:
        metrics = load_metrics()
        if metrics is None:
            return jsonify(
                success=False,
                error="Model belum dilatih. Jalankan train_model.py terlebih dahulu.",
            ), 404
        
        return jsonify(success=True, data=metrics), 200
    
    except Exception as e:
        logger.error(f"Error in /api/metrics: {str(e)}")
        return jsonify(
            success=False,
            error="Terjadi kesalahan server saat membaca metrics",
        ), 500


@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    """
    POST /api/diagnose

    Estimasi RISIKO hipertensi (skrining) berdasarkan fitur input.

    PENTING -- INI BUKAN DIAGNOSIS MEDIS:
    Hasil endpoint ini adalah estimasi risiko dari model statistik,
    bukan diagnosis klinis. Diagnosis hipertensi yang sebenarnya
    mensyaratkan rata-rata >=2 pengukuran tekanan darah pada >=2
    kesempatan berbeda (ACC/AHA 2017; ESC/ESH 2018), sedangkan
    endpoint ini hanya menerima satu kali input. Selalu sertakan
    "disclaimer" dari response ke pengguna akhir.

    SAFETY OVERRIDE:
    Jika pressure_category == "Sangat Tinggi (Hypertensive Crisis)"
    (sistolik >=180 dan/atau diastolik >=120), maka prediction/label/
    risk dipaksa ke kondisi darurat TERLEPAS dari skor model ML.
    Ini mencegah kasus di mana kombinasi fitur lain membuat skor
    model < 0.5 padahal tekanan darah pasien sudah di level darurat
    medis -- kondisi yang tidak boleh pernah dilabel "tidak
    terklasifikasi hipertensi".

    Request body (JSON):
        {
            "usia": 45,
            "jenis_kelamin": "laki-laki",
            "imt": 25.5,
            "riwayat_keluarga": "ya",
            "merokok": "tidak",
            "sistolik": 140,
            "diastolik": 90
        }

    Response (200):
        {
            "success": true,
            "data": {
                "prediction": 1,
                "label": "Berisiko Hipertensi",
                "rf_probability_percent": 65.23,
                "xgb_probability_percent": 72.15,
                "hybrid_probability_percent": 68.69,
                "risk": "Sedang",
                "pressure_category": "Tinggi (Stage 2 Hypertension)",
                "requires_immediate_attention": false,
                "safety_override_applied": false,
                "disclaimer": "..."
            }
        }

    Response (400/500):
        {
            "success": false,
            "error": "Error message"
        }
    """
    try:
        # Parse request
        request_data = request.get_json(force=True)
        if not request_data:
            raise ValueError("Request body harus berisi JSON")
        
        # Load models
        rf_model, xgb_model = load_models()
        if rf_model is None or xgb_model is None:
            logger.error("Models not available")
            return jsonify(
                success=False,
                error="Model belum tersedia. Jalankan train_model.py terlebih dahulu.",
            ), 503
        
        # Validate input
        validated_input = validate_input_data(request_data)
        
        # Create DataFrame for prediction
        X_input = pd.DataFrame([validated_input], columns=FEATURES)
        
        # Get predictions from both models
        rf_probability = float(rf_model.predict_proba(X_input)[0, 1])
        xgb_probability = float(xgb_model.predict_proba(X_input)[0, 1])
        
        # Hybrid prediction (soft voting) -- pakai bobot hasil tuning training
        # (metrics.json -> hybrid.rf_weight/xgb_weight), BUKAN 50:50 hardcode.
        # Sebelumnya kode ini selalu (rf+xgb)/2 walau train_model.py sudah
        # men-tuning bobot optimal (tune_weight=True) -- akibatnya UI yang
        # menampilkan "Soft Voting 50:50" bisa salah kalau bobot hasil
        # tuning bukan 50:50. Fallback ke 0.5/0.5 hanya kalau metrics.json
        # belum ada/belum berisi bobot (mis. training lama belum diulang).
        metrics_for_weight = load_metrics()
        # PERBAIKAN: rf_weight/xgb_weight ternyata disimpan train_model.py
        # di LEVEL ATAS metrics.json (metrics["rf_weight"]), bukan di dalam
        # metrics["hybrid"] seperti asumsi kode sebelumnya -- akibatnya kode
        # lama diam-diam SELALU fallback ke 0.5/0.5 walau metrics.json
        # sebenarnya sudah punya bobot hasil tuning (mis. 0.15/0.85).
        rf_weight = (
            metrics_for_weight.get("rf_weight", 0.5)
            if metrics_for_weight
            else 0.5
        )
        xgb_weight = (
            metrics_for_weight.get("xgb_weight", 0.5)
            if metrics_for_weight
            else 0.5
        )

        if (
            rf_weight < 0
            or xgb_weight < 0
            or abs((rf_weight + xgb_weight) - 1.0) > 1e-6
        ):
            raise RuntimeError("Bobot hybrid pada metrics.json tidak valid")

        hybrid_probability = (
            (rf_weight * rf_probability)
            + (xgb_weight * xgb_probability)
        )

        pressure_category = categorize_blood_pressure(
            validated_input["sistolik"],
            validated_input["diastolik"],
        )
        crisis = is_hypertensive_crisis(pressure_category)

        # Get risk classification from ML model
        # PERBAIKAN: pakai optimal_threshold hasil tuning train_model.py
        # (cost-aware, mempertimbangkan FN lebih mahal dari FP untuk
        # skrining klinis) -- bukan 0.5 hardcode yang mengabaikan semua
        # kerja tuning threshold di training. Fallback ke 0.5 hanya kalau
        # metrics.json belum punya field ini (training lama belum diulang).
        decision_threshold = (
            metrics_for_weight.get("optimal_threshold", 0.5)
            if metrics_for_weight
            else 0.5
        )
        prediction = int(hybrid_probability >= decision_threshold)
        risk = categorize_risk(hybrid_probability)
        diagnosis = (
            "Berisiko Hipertensi"
            if prediction == 1
            else "Risiko Hipertensi Rendah (berdasarkan model)"
        )

        # SAFETY OVERRIDE: tekanan darah krisis (>=180/120) HARUS selalu
        # ditandai darurat, terlepas dari skor probabilitas model ML.
        # Model dilatih dari fitur tidak lengkap (7 variabel) dan tidak
        # boleh jadi satu-satunya penentu saat tanda vital sudah di
        # level darurat medis.
        if crisis:
            prediction = 1
            risk = "Tinggi"
            diagnosis = "Krisis Hipertensi -- SEGERA cari pertolongan medis"

        # Prepare response
        response = {
            "prediction": prediction,
            "label": diagnosis,
            "rf_probability_percent": round(rf_probability * 100, 2),
            "xgb_probability_percent": round(xgb_probability * 100, 2),
            "hybrid_probability_percent": round(hybrid_probability * 100, 2),
            "risk": risk,
            "pressure_category": pressure_category,
            "requires_immediate_attention": crisis,
            "safety_override_applied": crisis,
            "rf_weight_percent": round(rf_weight * 100),
            "xgb_weight_percent": round(xgb_weight * 100),
            "decision_threshold_percent": round(decision_threshold * 100, 1),
            "disclaimer": (
                "Hasil ini adalah estimasi risiko dari model statistik "
                "berdasarkan 7 variabel input, BUKAN diagnosis medis. "
                "Diagnosis hipertensi klinis memerlukan rata-rata >=2 kali "
                "pengukuran tekanan darah pada >=2 kesempatan berbeda oleh "
                "tenaga medis. Konsultasikan hasil ini dengan dokter, "
                "terutama jika kategori tekanan darah Anda 'Meningkat' "
                "atau lebih tinggi."
            ),
        }

        logger.info(f"Diagnosis successful: {diagnosis} (safety_override={crisis})")
        return jsonify(success=True, data=response), 200
    
    except ValueError as e:
        # Input validation error
        logger.warning(f"Input validation error: {str(e)}")
        return jsonify(
            success=False,
            error=f"Input tidak valid: {str(e)}",
        ), 400
    
    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error in /api/diagnose: {str(e)}", exc_info=True)
        return jsonify(
            success=False,
            error="Terjadi kesalahan server saat melakukan diagnosa",
        ), 500


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify(
        success=False,
        error="Endpoint tidak ditemukan",
    ), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify(
        success=False,
        error="Method tidak diizinkan untuk endpoint ini",
    ), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify(
        success=False,
        error="Terjadi kesalahan internal server",
    ), 500

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # PERBAIKAN KEAMANAN: debug=True SEBELUMNYA aktif secara default.
    # Werkzeug debugger interaktif (aktif saat debug=True) memungkinkan
    # eksekusi kode Python arbitrer dari browser siapa pun yang bisa
    # mengakses server -- RCE (Remote Code Execution) kalau server ini
    # pernah terekspos di luar localhost/development machine.
    #
    # Sekarang dikontrol via environment variable FLASK_DEBUG (default
    # "0"/False). Untuk development lokal: set FLASK_DEBUG=1 di environment
    # sebelum menjalankan skrip ini. Untuk PRODUCTION, jangan pernah
    # mengaktifkan debug mode -- gunakan WSGI server (mis. gunicorn/waitress)
    # alih-alih app.run() bawaan Flask, contoh:
    #   gunicorn -w 4 -b 0.0.0.0:5000 app:app
    import os

    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    if debug_mode:
        logger.warning(
            "Menjalankan dengan debug=True -- HANYA untuk development lokal, "
            "JANGAN gunakan mode ini di production/server yang terekspos jaringan."
        )

    logger.info("Starting Flask application...")
    app.run(host="127.0.0.1", port=5000, debug=debug_mode)