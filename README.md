# HiperDiagnosa Hybrid Random Forest + XGBoost

Sistem Pendukung Keputusan untuk skrining awal risiko hipertensi
menggunakan ensemble Random Forest dan XGBoost.

## Versi perbaikan
- Dataset penelitian dibaca dari `data/dataset.csv`; jumlah baris mengikuti
	dataset aktif setelah lolos validasi kualitas data.
- Soft Voting: RF 50% + XGBoost 50%.
- Threshold dipilih dari validation set menggunakan Youden Index.
- Test set hanya digunakan sekali untuk evaluasi akhir.
- Validasi input aplikasi disamakan dengan rentang dataset.
- Mapping jenis kelamin dipisahkan dari mapping Ya/Tidak.
- Model `.joblib` dilatih ulang dengan versi library yang dicantumkan
	di `requirements.txt`.

## Fitur
- usia
- jenis_kelamin
- imt
- riwayat_keluarga
- merokok
- sistolik
- diastolik

## Target
- `hipertensi = 0`
- `hipertensi = 1`

## Training ulang
```bash
pip install -r requirements.txt
python train_model.py
```

## Menjalankan aplikasi
```bash
python app.py
```

Buka `http://127.0.0.1:5000`.

## Deploy ke Render

1. Push repository ini ke GitHub.
2. Di Render, pilih **New > Blueprint** lalu hubungkan repository tersebut.
3. Render akan membaca `render.yaml`, memasang dependensi, dan menjalankan
	aplikasi dengan Gunicorn.
4. Setelah deploy selesai, buka URL HTTPS yang diberikan Render.

Pastikan folder `models/` dan file model `.joblib` ikut ter-commit karena
aplikasi membutuhkannya saat startup/request. Jangan aktifkan `FLASK_DEBUG=1`
di environment production.

## Deploy ke Railway

1. Buka [Railway](https://railway.app) dan pilih **Login with GitHub**.
2. Pilih **New Project > Deploy from GitHub Repo**.
3. Pilih repository `HiperDiagnosa_Hybrid_RF_XGBoost` dan branch `main`.
4. Railway akan mendeteksi `requirements.txt` dan menjalankan `Procfile`.
5. Di menu **Settings > Networking**, pilih **Generate Domain**.

Railway menyediakan nilai `PORT` secara otomatis. Jangan mengubahnya menjadi
port tetap karena Gunicorn harus mengikuti port tersebut.

> Sistem ini adalah alat skrining berbasis model statistik, bukan
> pengganti diagnosis dokter.
