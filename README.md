# HelmGuard — Sistem Monitoring Kepatuhan Helm Keselamatan

Aplikasi Streamlit untuk mendeteksi kepatuhan penggunaan helm keselamatan
menggunakan model YOLO, dilengkapi dashboard, alarm suara, log pelanggaran,
dan bukti foto otomatis. Dirancang untuk dipakai langsung di area kerja
atau tambang, bukan hanya demo.

## Apa yang berubah dari versi awal

- **Log tidak lagi bikin lag.** Versi awal membuka & menulis ulang file
  Excel penuh setiap ada satu frame pelanggaran — di mode video/webcam ini
  bisa membekukan aplikasi. Sekarang log ditulis sebagai baris CSV yang
  ditambahkan langsung (append), jauh lebih ringan untuk stream terus-menerus.
- **Alarm tidak lagi bergantung penuh pada internet.** gTTS butuh koneksi
  internet setiap kali dipanggil — masalah nyata di lokasi tambang dengan
  sinyal terbatas. Sekarang suara di-cache, dan jika gTTS gagal/tidak ada
  internet, sistem otomatis memakai nada bip peringatan yang dibuat secara
  lokal tanpa koneksi sama sekali.
- **Bug warna kotak deteksi pada mode gambar sudah diperbaiki** (sebelumnya
  kotak merah bisa tampil biru karena urutan channel warna RGB vs BGR tertukar).
- **Kelas "tidak pakai helm" tidak lagi di-hardcode** ke `"head"` — sekarang
  dibaca otomatis dari model dan bisa dipilih sendiri, jadi berfungsi untuk
  model YOLO apa pun, bukan cuma satu model tertentu.
- **Fitur baru untuk kebutuhan lapangan:** dashboard tren pelanggaran,
  bukti foto otomatis per kejadian, riwayat yang bisa difilter & diekspor
  (CSV/Excel), dukungan input CCTV via RTSP, frame-skip untuk mempercepat
  pemrosesan video, dan halaman Pengaturan untuk mengganti model/lokasi/teks
  peringatan tanpa mengedit kode.

## Instalasi

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Letakkan berkas bobot model (`.pt`) hasil training YOLO Anda di folder
aplikasi ini (sejajar dengan `app.py`). Aplikasi akan otomatis
mendeteksinya di halaman **Pengaturan**, atau bisa diunggah langsung dari
antarmuka.

## Menjalankan aplikasi

```bash
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

## Catatan penting untuk deployment di tambang/lokasi kerja

1. **Kamera lokal vs CCTV.** Mode "Webcam" hanya bisa mengakses kamera pada
   perangkat yang menjalankan Streamlit secara langsung — ini cocok kalau
   aplikasi dijalankan di PC/laptop/mini-PC yang terpasang di lokasi. Kalau
   aplikasi dijalankan di server terpusat dan kamera ada di lapangan,
   gunakan tab **CCTV (RTSP)** dan masukkan URL stream kamera
   (`rtsp://user:pass@ip:554/stream1`).
2. **Koneksi internet terbatas.** Alarm suara punya cadangan offline
   otomatis (lihat di atas). Model YOLO dan seluruh proses deteksi berjalan
   lokal, tidak butuh internet sama sekali setelah paket ter-install.
3. **Penyimpanan.** Log (`logs/pelanggaran_log.csv`) dan bukti foto
   (`snapshots/`) tersimpan lokal di folder aplikasi. Untuk pemakaian
   jangka panjang, jadwalkan backup berkala folder ini, atau pindahkan ke
   penyimpanan jaringan.
4. **Performa.** Untuk video/CCTV resolusi tinggi, naikkan nilai
   "Proses tiap N frame" di tab Video, atau turunkan resolusi input kamera,
   supaya deteksi tetap real-time di perangkat dengan GPU terbatas.
5. **Multi-lokasi.** Isi "Nama lokasi/situs" di halaman Pengaturan agar log
   dari beberapa titik pemasangan bisa dibedakan saat digabungkan.

## Struktur folder

```
helmet-monitor/
├── app.py                  # Aplikasi utama
├── requirements.txt
├── .streamlit/config.toml  # Tema bawaan Streamlit
├── logs/pelanggaran_log.csv
└── snapshots/               # Bukti foto otomatis
```
