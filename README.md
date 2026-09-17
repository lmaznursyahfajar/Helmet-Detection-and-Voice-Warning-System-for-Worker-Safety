# 🪖 Helmet Detection System

Aplikasi berbasis web interaktif menggunakan **Streamlit** dan model komputer visi **YOLO** untuk mendeteksi penggunaan helm keselamatan (seperti pada area konstruksi atau pengendara). Sistem ini secara otomatis mendeteksi pelanggaran (kondisi tanpa helm), memberikan peringatan suara (*voice warning*) menggunakan Text-to-Speech (gTTS), serta mencatat riwayat pelanggaran ke dalam file Excel.

---

## ✨ Fitur Utama

- **🔍 Multi-Mode Input**:
  - **📷 Gambar**: Unggah file gambar (`.jpg`, `.jpeg`, `.png`) untuk diproses.
  - **🎥 Video**: Unggah file video (`.mp4`, `.avi`, `.mov`) lengkap dengan indikator kemajuan (*progress bar*).
  - **📹 Webcam**: Deteksi langsung secara *real-time* menggunakan kamera internal/eksternal.
- **🔊 Peringatan Suara Otomatis**: Memutar pesan suara (*"Harap gunakan helm untuk keselamatan Anda"*) via gTTS ketika pelanggaran terdeteksi, dilengkapi fitur *cooldown* 5 detik untuk mencegah suara berulang secara berlebihan.
- **📊 Logging Pelanggaran Otomatis**: Setiap pelanggaran yang terdeteksi pada media akan otomatis dicatat ke dalam berkas Excel (`pelanggaran_helm.xlsx`) beserta timestamp waktu kejadian.
- **⚙️ Konfigurasi Fleksibel**: Pengaturan ambang batas kepercayaan (*Confidence Threshold*) melalui *slider* di sidebar.

---

## 📂 Berkas & Asset yang Dibutuhkan

Pastikan file berikut berada di dalam direktori proyek Anda sebelum menjalankan aplikasi:

1. **`app.py`**: File utama aplikasi Streamlit.
2. **`bestt.pt`**: Weights model YOLO yang telah dilatih untuk mendeteksi helm / kepala.
3. **`pelanggaran_helm.xlsx`**: File catat riwayat pelanggaran (dibuat otomatis jika belum ada).

---

## 📦 Prasyarat & Instalasi

### 1. Buat File `requirements.txt`

Buat file bernama `requirements.txt` pada direktori proyek Anda dan tambahkan pustaka berikut:

```text
streamlit
ultralytics
opencv-python
numpy
pillow
pandas
openpyxl
gtts
