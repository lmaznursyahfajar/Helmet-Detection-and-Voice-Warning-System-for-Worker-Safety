from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
import warnings
import pandas as pd
import datetime
import os
import base64
from gtts import gTTS
import time

warnings.filterwarnings("ignore")

# ====== STYLE ======
st.set_page_config(page_title="Helmet Detection", page_icon="🪖", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .stFileUploader label { font-size: 18px; color: #00ffcc; font-weight: bold; }
    .stImage img { border-radius: 10px; }
    .title { text-align: center; font-size: 36px; font-weight: bold; color: #00ffcc; }
    .subtitle { text-align: center; font-size: 18px; color: #cccccc; }
    </style>
""", unsafe_allow_html=True)

# ====== CONFIG ======
MODEL_PATH = "bestt.pt"  # ganti path model
model = YOLO(MODEL_PATH)
LOG_FILE = "pelanggaran_helm.xlsx"
NO_HELM_CLASSES = ["head"]  # class yang dianggap tidak pakai helm

# ====== LOG FILE ======
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["Waktu", "File", "Jumlah Pelanggaran"]).to_excel(LOG_FILE, index=False)

# ====== COOLDOWN SUARA ======
last_voice_time = 0  # waktu terakhir suara diputar
VOICE_COOLDOWN = 5   # jeda 5 detik

# ====== PLAY ASSISTANT VOICE ======
def play_assistant_voice():
    tts = gTTS("Harap gunakan helm untuk keselamatan Anda", lang="id")
    tts_file = "assistant_voice.mp3"
    tts.save(tts_file)

    with open(tts_file, "rb") as f:
        voice_bytes = f.read()
    voice_b64 = base64.b64encode(voice_bytes).decode()
    st.markdown(f"""
        <audio autoplay>
        <source src="data:audio/mp3;base64,{voice_b64}" type="audio/mp3">
        </audio>
    """, unsafe_allow_html=True)

# ====== LOG PELANGGARAN ======
def catat_pelanggaran(file_name, jumlah):
    waktu = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.read_excel(LOG_FILE)
    df = pd.concat([df, pd.DataFrame([[waktu, file_name, jumlah]], columns=df.columns)], ignore_index=True)
    df.to_excel(LOG_FILE, index=False)

# ====== DETEKSI ======
def proses_frame(frame, min_conf=0.5, file_name=""):
    global last_voice_time

    results = model(frame)
    jumlah_pelanggaran = 0
    frame_det = frame.copy()

    for r in results:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            label = model.names[int(cls)]
            if conf >= min_conf:
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0)
                if label in NO_HELM_CLASSES:
                    jumlah_pelanggaran += 1
                    color = (0, 0, 255)
                cv2.rectangle(frame_det, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_det, f"{label} {conf*100:.1f}%",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Hanya mainkan suara jika cooldown selesai
    if jumlah_pelanggaran > 0:
        if time.time() - last_voice_time > VOICE_COOLDOWN:
            play_assistant_voice()
            last_voice_time = time.time()
        if file_name:
            catat_pelanggaran(file_name, jumlah_pelanggaran)

    return frame_det

# ====== HEADER ======
st.markdown("<div class='title'>🪖 Helmet Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deteksi helm pekerja konstruksi + peringatan suara asisten</div>", unsafe_allow_html=True)
st.write("")

# ====== SIDEBAR ======
mode = st.sidebar.radio("Pilih Mode Deteksi:", ["Gambar", "Video", "Webcam"])
conf_thres = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# ====== MODE GAMBAR ======
if mode == "Gambar":
    uploaded_file = st.file_uploader("📷 Upload gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        hasil = proses_frame(image, min_conf=conf_thres, file_name=uploaded_file.name)
        st.image(hasil, caption="Hasil Deteksi", use_container_width=True)

# ====== MODE VIDEO ======
elif mode == "Video":
    uploaded_file = st.file_uploader("🎥 Upload video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        progress = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            hasil = proses_frame(frame, min_conf=conf_thres, file_name=uploaded_file.name)
            stframe.image(hasil, channels="BGR", use_container_width=True)

            processed += 1
            progress.progress(min(processed / frame_count, 1.0))

        cap.release()

# ====== MODE WEBCAM ======
elif mode == "Webcam":
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    st.warning("Tekan **Stop** di toolbar Streamlit untuk menghentikan webcam.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        hasil = proses_frame(frame, min_conf=conf_thres)
        stframe.image(hasil, channels="BGR", use_container_width=True)
    cap.release()
