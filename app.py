"""
HelmGuard — Sistem Monitoring Kepatuhan APD (Helm Keselamatan)
================================================================
Aplikasi Streamlit berbasis YOLO untuk mendeteksi kepatuhan penggunaan
helm keselamatan di area kerja / tambang, lengkap dengan dashboard,
peringatan suara, log pelanggaran, dan bukti foto (snapshot).

Jalankan dengan:
    streamlit run app.py
"""

import base64
import csv
import io
import struct
import time
import wave
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ----------------------------------------------------------------------
# Dependensi opsional — aplikasi tetap bisa dibuka (dengan pesan yang
# jelas) walau salah satu dari ini belum terpasang, alih-alih crash.
# ----------------------------------------------------------------------
try:
    from ultralytics import YOLO
    YOLO_READY = True
except ImportError:
    YOLO_READY = False

try:
    from gtts import gTTS
    GTTS_READY = True
except ImportError:
    GTTS_READY = False


# ========================================================================
# KONFIGURASI & PATH
# ========================================================================
APP_DIR = Path(__file__).resolve().parent
LOG_DIR = APP_DIR / "logs"
SNAPSHOT_DIR = APP_DIR / "snapshots"
LOG_FILE = LOG_DIR / "pelanggaran_log.csv"
LOG_COLUMNS = [
    "waktu", "lokasi", "sumber", "jumlah_pelanggaran",
    "total_terdeteksi", "confidence_min", "snapshot",
]

LOG_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="HelmGuard — Monitoring APD",
    page_icon="⛑️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ========================================================================
# TEMA VISUAL
# ========================================================================
def inject_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

        :root{
            --bg:#14171b; --panel:#1b1f24; --panel-alt:#20262d;
            --border:#2a3138; --text:#e8eaec; --muted:#8b95a1;
            --amber:#f2a93b; --red:#e15241; --green:#49a67c;
        }

        html, body, [class*="css"]  { font-family:'Inter',sans-serif; }
        .stApp { background:var(--bg); color:var(--text); }
        section[data-testid="stSidebar"] { background:var(--panel); border-right:1px solid var(--border); }

        .hazard-strip{
            height:6px; margin:-1rem -1rem 1.4rem -1rem;
            background:repeating-linear-gradient(135deg, var(--amber) 0 14px, var(--bg) 14px 28px);
        }

        .app-header{ display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:1.6rem; flex-wrap:wrap; gap:.8rem;}
        .app-title{ font-size:26px; font-weight:700; color:var(--text); margin:0; letter-spacing:-.01em;}
        .app-subtitle{ font-size:14px; color:var(--muted); margin-top:2px;}
        .site-tag{ font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--amber); border:1px solid var(--border); background:var(--panel); padding:5px 10px; border-radius:3px;}

        .status-row{ display:flex; gap:10px; flex-wrap:wrap; }
        .status-pill{ display:inline-flex; align-items:center; gap:7px; padding:5px 11px; border-radius:3px; font-size:12px; font-family:'IBM Plex Mono',monospace; border:1px solid var(--border); background:var(--panel); color:var(--text);}
        .status-dot{ width:8px; height:8px; border-radius:50%; flex-shrink:0;}
        .status-dot.on{ background:var(--green); box-shadow:0 0 6px var(--green);}
        .status-dot.off{ background:var(--muted); }
        .status-dot.alert{ background:var(--red); box-shadow:0 0 6px var(--red); animation:pulse 1s infinite;}
        @keyframes pulse{ 0%,100%{opacity:1;} 50%{opacity:.35;} }

        .kpi-row{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:1.6rem;}
        .kpi-card{ flex:1; min-width:170px; background:var(--panel); border:1px solid var(--border); border-radius:4px; padding:18px 20px;}
        .kpi-value{ font-family:'IBM Plex Mono',monospace; font-size:30px; font-weight:600; color:var(--text); line-height:1.1;}
        .kpi-label{ font-size:12.5px; color:var(--muted); margin-top:6px;}
        .kpi-card.danger .kpi-value{ color:var(--red); }
        .kpi-card.safe .kpi-value{ color:var(--green); }
        .kpi-card.amber .kpi-value{ color:var(--amber); }

        .panel{ background:var(--panel); border:1px solid var(--border); border-radius:4px; padding:20px; margin-bottom:1rem;}
        .panel-title{ font-size:14px; font-weight:600; color:var(--text); margin-bottom:12px; padding-bottom:10px; border-bottom:1px solid var(--border);}

        .empty-state{ color:var(--muted); font-size:13.5px; padding:22px 4px; text-align:center; border:1px dashed var(--border); border-radius:4px;}

        div[data-testid="stMetricValue"]{ font-family:'IBM Plex Mono',monospace; }
        .stButton>button{ border-radius:3px; border:1px solid var(--border); font-weight:500; }
        .stButton>button[kind="primary"]{ background:var(--amber); border-color:var(--amber); color:#14171b; }

        footer{visibility:hidden;}
        #MainMenu{visibility:hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_pill(label, state):
    """state: 'on' | 'off' | 'alert'"""
    return f'<span class="status-pill"><span class="status-dot {state}"></span>{label}</span>'


def kpi_card(label, value, variant=""):
    return f'<div class="kpi-card {variant}"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>'


# ========================================================================
# STATE AWAL
# ========================================================================
DEFAULTS = {
    "site_name": "Area Kerja — Site A",
    "conf_threshold": 0.5,
    "iou_threshold": 0.45,
    "voice_text": "Perhatian, harap gunakan helm untuk keselamatan Anda",
    "voice_cooldown": 6,
    "sound_enabled": True,
    "snapshot_enabled": True,
    "violation_classes": [],
    "helmet_classes": [],
    "person_classes": [],
    "detection_mode": "class",
    "webcam_running": False,
    "session_total": 0,
    "session_violation": 0,
    "last_voice_time": 0.0,
    "last_log_time": 0.0,
    "model_path": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ========================================================================
# MODEL
# ========================================================================
@st.cache_resource(show_spinner="Memuat model deteksi...")
def load_model(model_path: str):
    return YOLO(model_path)


def available_weights():
    return sorted([p.name for p in APP_DIR.glob("*.pt")])


# ========================================================================
# AUDIO — suara peringatan dengan fallback offline (tanpa internet)
# ========================================================================
@st.cache_data(show_spinner=False)
def generate_voice_bytes(text: str):
    """Coba pakai gTTS (butuh internet). Jika gagal, pakai nada bip
    yang dibuat langsung tanpa koneksi internet, supaya alarm tetap
    berfungsi di lokasi dengan sinyal terbatas seperti tambang."""
    if GTTS_READY:
        try:
            buf = io.BytesIO()
            gTTS(text=text, lang="id").write_to_fp(buf)
            return buf.getvalue(), "audio/mp3"
        except Exception:
            pass
    return generate_beep_tone(), "audio/wav"


def generate_beep_tone(freq=1000, duration=0.35, sr=22050, repeats=2, gap=0.12):
    n_samples = int(sr * duration)
    n_gap = int(sr * gap)
    samples = []
    for _ in range(repeats):
        for i in range(n_samples):
            val = int(32767 * 0.5 * np.sin(2 * np.pi * freq * i / sr))
            samples.append(val)
        samples.extend([0] * n_gap)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack("<%dh" % len(samples), *samples))
    return buf.getvalue()


def play_alert(audio_bytes, mime):
    b64 = base64.b64encode(audio_bytes).decode()
    key = int(time.time() * 1000)
    st.markdown(
        f'<audio autoplay key="{key}"><source src="data:{mime};base64,{b64}" type="{mime}"></audio>',
        unsafe_allow_html=True,
    )


# ========================================================================
# LOG PELANGGARAN (CSV — ringan, aman ditulis per-event)
# ========================================================================
def append_log(sumber, jumlah_pelanggaran, total_terdeteksi, snapshot_path=""):
    is_new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(LOG_COLUMNS)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            st.session_state.site_name,
            sumber,
            jumlah_pelanggaran,
            total_terdeteksi,
            st.session_state.conf_threshold,
            snapshot_path,
        ])


@st.cache_data(ttl=5)
def load_log():
    if not LOG_FILE.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.read_csv(LOG_FILE)
    df["waktu"] = pd.to_datetime(df["waktu"], errors="coerce")
    return df


def save_snapshot(frame_bgr):
    fname = f"pelanggaran_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    path = SNAPSHOT_DIR / fname
    cv2.imwrite(str(path), frame_bgr)
    return str(path.relative_to(APP_DIR))


# ========================================================================
# INFERENSI
# ========================================================================
def classify_semantics(class_names):
    """Kelompokkan nama kelas model ke peran semantik, supaya sistem tidak
    pernah menebak buta kelas mana yang berarti 'pelanggaran'.
    Mengembalikan (kelas_pelanggaran, kelas_helm, kelas_orang, kelas_lain)."""
    violation, helmet, person, other = [], [], [], []
    for c in class_names:
        lc = c.lower().replace(" ", "").replace("_", "").replace("-", "")
        if any(k in lc for k in ["nohelmet", "nohardhat", "nohelm", "without", "tanpahelm", "unsafe"]) or lc == "head":
            violation.append(c)
        elif any(k in lc for k in ["helmet", "hardhat", "helm"]):
            helmet.append(c)
        elif any(k in lc for k in ["person", "orang", "pekerja", "worker", "human"]):
            person.append(c)
        else:
            other.append(c)
    return violation, helmet, person, other


def boxes_overlap_ratio(box_a, box_b):
    """Rasio luas irisan terhadap luas box_b — dipakai untuk menilai apakah
    sebuah kotak helm berada pada zona kepala kotak orang."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / area_b


def head_zone(person_box, ratio=0.4):
    x1, y1, x2, y2 = person_box
    return (x1, y1, x2, y1 + (y2 - y1) * ratio)


def run_detection(model, frame_bgr, conf, iou, det_config):
    """frame_bgr: numpy array BGR (konvensi OpenCV).
    det_config: {"mode": "class"|"overlap", "violation_classes": [...],
                 "helmet_classes": [...], "person_classes": [...]}.
    Mengembalikan (frame_tergambar_BGR, jumlah_pelanggaran, total_terdeteksi)."""
    results = model(frame_bgr, conf=conf, iou=iou, verbose=False)
    drawn = frame_bgr.copy()
    boxes = []
    for r in results:
        for box, cls, cf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            boxes.append({"label": model.names[int(cls)], "box": (x1, y1, x2, y2), "conf": float(cf)})

    if det_config["mode"] == "overlap":
        drawn, violations, total = _draw_overlap_mode(drawn, boxes, det_config)
    else:
        drawn, violations, total = _draw_class_mode(drawn, boxes, det_config["violation_classes"])

    if violations > 0:
        banner = f"PELANGGARAN TERDETEKSI: {violations}"
        cv2.rectangle(drawn, (0, 0), (drawn.shape[1], 34), (65, 82, 225), -1)
        cv2.putText(drawn, banner, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    return drawn, violations, total


def _draw_box(drawn, box, color, tag):
    x1, y1, x2, y2 = box
    cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(drawn, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
    cv2.putText(drawn, tag, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_class_mode(drawn, boxes, violation_classes):
    """Kotak dianggap pelanggaran murni berdasarkan nama kelasnya
    (cocok untuk model dengan kelas eksplisit seperti 'no-helmet'/'head')."""
    violations, total = 0, 0
    for b in boxes:
        total += 1
        is_violation = b["label"] in violation_classes
        color = (65, 82, 225) if is_violation else (124, 166, 73)  # BGR: merah / hijau
        if is_violation:
            violations += 1
        _draw_box(drawn, b["box"], color, f"{b['label']} {b['conf'] * 100:.0f}%")
    return drawn, violations, total


def _draw_overlap_mode(drawn, boxes, det_config):
    """Untuk model yang hanya punya kelas 'helm' dan 'orang' tanpa kelas
    eksplisit 'tanpa helm': seseorang dianggap patuh hanya jika ada kotak
    helm yang tumpang tindih signifikan dengan zona kepalanya."""
    helmet_boxes = [b["box"] for b in boxes if b["label"] in det_config["helmet_classes"]]
    person_boxes = [b for b in boxes if b["label"] in det_config["person_classes"]]

    violations, total = 0, 0
    for p in person_boxes:
        total += 1
        zone = head_zone(p["box"])
        has_helmet = any(boxes_overlap_ratio(zone, hb) > 0.5 for hb in helmet_boxes)
        color = (124, 166, 73) if has_helmet else (65, 82, 225)
        if not has_helmet:
            violations += 1
        tag = "helm terpasang" if has_helmet else "tidak pakai helm"
        _draw_box(drawn, p["box"], color, f"{tag} {p['conf'] * 100:.0f}%")

    for hb in helmet_boxes:
        cv2.rectangle(drawn, (hb[0], hb[1]), (hb[2], hb[3]), (58, 169, 242), 1)

    return drawn, violations, total


def handle_violation_event(frame_bgr, violations, total, sumber):
    """Bunyikan alarm (dengan cooldown) + catat log + simpan snapshot."""
    if violations == 0:
        return
    now = time.time()

    if st.session_state.sound_enabled and now - st.session_state.last_voice_time > st.session_state.voice_cooldown:
        audio_bytes, mime = generate_voice_bytes(st.session_state.voice_text)
        play_alert(audio_bytes, mime)
        st.session_state.last_voice_time = now

    if now - st.session_state.last_log_time > 3:
        snap_path = save_snapshot(frame_bgr) if st.session_state.snapshot_enabled else ""
        append_log(sumber, violations, total, snap_path)
        st.session_state.last_log_time = now
        load_log.clear()


# ========================================================================
# HEADER
# ========================================================================
def render_header(model_ready):
    st.markdown('<div class="hazard-strip"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(
            f"""
            <div class="app-header">
                <div>
                    <p class="app-title">⛑️ HelmGuard</p>
                    <p class="app-subtitle">Sistem monitoring kepatuhan helm keselamatan</p>
                </div>
                <span class="site-tag">{st.session_state.site_name}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        model_state = "on" if model_ready else "off"
        sound_state = "on" if st.session_state.sound_enabled else "off"
        pills = (
            status_pill("Model " + ("aktif" if model_ready else "belum siap"), model_state)
            + status_pill("Alarm " + ("aktif" if st.session_state.sound_enabled else "nonaktif"), sound_state)
            + status_pill(datetime.now().strftime("%H:%M:%S — %d %b %Y"), "off")
        )
        st.markdown(f'<div class="status-row" style="justify-content:flex-end;margin-top:14px;">{pills}</div>', unsafe_allow_html=True)


# ========================================================================
# HALAMAN: DASHBOARD
# ========================================================================
def page_dashboard():
    df = load_log()
    today = pd.Timestamp.now().normalize()

    total_pelanggaran = int(df["jumlah_pelanggaran"].sum()) if not df.empty else 0
    hari_ini = int(df[df["waktu"] >= today]["jumlah_pelanggaran"].sum()) if not df.empty else 0
    minggu_ini = int(df[df["waktu"] >= today - timedelta(days=7)]["jumlah_pelanggaran"].sum()) if not df.empty else 0
    kejadian = len(df) if not df.empty else 0

    st.markdown(
        '<div class="kpi-row">'
        + kpi_card("Pelanggaran hari ini", hari_ini, "danger" if hari_ini else "safe")
        + kpi_card("Pelanggaran 7 hari terakhir", minggu_ini, "amber")
        + kpi_card("Total pelanggaran tercatat", total_pelanggaran)
        + kpi_card("Jumlah kejadian terlog", kejadian)
        + "</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown('<div class="panel"><div class="panel-title">Tren pelanggaran per jam (7 hari terakhir)</div>', unsafe_allow_html=True)
        if df.empty:
            st.markdown('<div class="empty-state">Belum ada data. Jalankan deteksi untuk mulai mencatat riwayat.</div>', unsafe_allow_html=True)
        else:
            recent = df[df["waktu"] >= today - timedelta(days=7)].copy()
            if recent.empty:
                st.markdown('<div class="empty-state">Belum ada pelanggaran dalam 7 hari terakhir.</div>', unsafe_allow_html=True)
            else:
                recent["jam"] = recent["waktu"].dt.floor("h")
                trend = recent.groupby("jam")["jumlah_pelanggaran"].sum()
                st.bar_chart(trend, color="#e15241")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="panel"><div class="panel-title">Kejadian terbaru</div>', unsafe_allow_html=True)
        if df.empty:
            st.markdown('<div class="empty-state">Belum ada kejadian tercatat.</div>', unsafe_allow_html=True)
        else:
            recent5 = df.sort_values("waktu", ascending=False).head(5)
            for _, row in recent5.iterrows():
                st.markdown(
                    f'<div style="padding:9px 0;border-bottom:1px solid var(--border);font-size:13px;">'
                    f'<span style="font-family:\'IBM Plex Mono\',monospace;color:var(--muted);">{row["waktu"].strftime("%d/%m %H:%M")}</span> — '
                    f'<b style="color:var(--red);">{int(row["jumlah_pelanggaran"])} pelanggaran</b> · {row["sumber"]}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)


# ========================================================================
# HALAMAN: DETEKSI LANGSUNG
# ========================================================================
def page_detection(model):
    class_names = list(model.names.values()) if model else []
    if class_names and class_names != st.session_state.get("_last_class_names"):
        violation_guess, helmet_guess, person_guess, _ = classify_semantics(class_names)
        st.session_state.violation_classes = violation_guess
        st.session_state.helmet_classes = helmet_guess
        st.session_state.person_classes = person_guess
        st.session_state.detection_mode = "class" if violation_guess else ("overlap" if helmet_guess and person_guess else "class")
        st.session_state._last_class_names = class_names

    mode_options = {
        "class": "Berdasarkan kelas (model punya kelas eksplisit 'tanpa helm')",
        "overlap": "Berdasarkan posisi (model hanya punya kelas 'helm' & 'orang')",
    }
    with st.expander("⚙️ Parameter deteksi", expanded=False):
        st.caption(
            f"Kelas terdeteksi pada model — helm: `{', '.join(st.session_state.helmet_classes) or '-'}` · "
            f"orang: `{', '.join(st.session_state.person_classes) or '-'}` · "
            f"tanpa-helm: `{', '.join(st.session_state.violation_classes) or '-'}`"
        )
        st.session_state.detection_mode = st.radio(
            "Metode deteksi pelanggaran", list(mode_options.keys()),
            format_func=lambda k: mode_options[k],
            index=list(mode_options.keys()).index(st.session_state.detection_mode)
            if st.session_state.detection_mode in mode_options else 0,
            horizontal=False,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.session_state.conf_threshold = st.slider("Ambang keyakinan (confidence)", 0.1, 1.0, st.session_state.conf_threshold, 0.05)
        with c2:
            st.session_state.iou_threshold = st.slider("Ambang IoU", 0.1, 1.0, st.session_state.iou_threshold, 0.05)

        if st.session_state.detection_mode == "class":
            st.session_state.violation_classes = st.multiselect(
                "Kelas yang berarti TIDAK memakai helm", class_names, default=st.session_state.violation_classes,
                help="Pilih hanya kelas yang menandakan kepala tanpa helm (mis. 'head' atau 'no-helmet'). "
                     "Jangan pilih kelas 'helmet' — helm yang terdeteksi berarti PATUH, bukan pelanggaran.",
            )
            if not st.session_state.violation_classes:
                st.warning("Belum ada kelas 'tanpa helm' dipilih — semua deteksi akan dianggap patuh. Pilih kelasnya di atas.")
        else:
            hc1, hc2 = st.columns(2)
            with hc1:
                st.session_state.helmet_classes = st.multiselect(
                    "Kelas yang berarti helm terpasang", class_names, default=st.session_state.helmet_classes
                )
            with hc2:
                st.session_state.person_classes = st.multiselect(
                    "Kelas yang berarti orang/pekerja", class_names, default=st.session_state.person_classes
                )
            if not st.session_state.helmet_classes or not st.session_state.person_classes:
                st.warning("Pilih kelas 'helm' dan 'orang' agar sistem bisa menyimpulkan siapa yang tidak memakai helm.")

    tab_img, tab_vid, tab_cam, tab_cctv = st.tabs(["📷 Gambar", "🎥 Video", "🎦 Webcam", "📡 CCTV (RTSP)"])

    with tab_img:
        detect_image(model)
    with tab_vid:
        detect_video(model)
    with tab_cam:
        detect_live(model, source=0, label="Webcam lokal")
    with tab_cctv:
        detect_cctv(model)


def current_det_config():
    return {
        "mode": st.session_state.detection_mode,
        "violation_classes": st.session_state.violation_classes,
        "helmet_classes": st.session_state.helmet_classes,
        "person_classes": st.session_state.person_classes,
    }


def detect_image(model):
    uploaded = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"], key="img_uploader")
    if uploaded is None:
        st.markdown('<div class="empty-state">Unggah foto area kerja untuk memeriksa kepatuhan helm.</div>', unsafe_allow_html=True)
        return
    pil_img = Image.open(uploaded).convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    drawn, violations, total = run_detection(
        model, frame_bgr, st.session_state.conf_threshold, st.session_state.iou_threshold, current_det_config()
    )
    st.image(drawn, channels="BGR", use_container_width=True)
    show_result_summary(violations, total)
    handle_violation_event(drawn, violations, total, uploaded.name)


def detect_video(model):
    uploaded = st.file_uploader("Unggah video", type=["mp4", "avi", "mov", "mkv"], key="vid_uploader")
    frame_skip = st.slider("Proses tiap N frame (percepat pemrosesan)", 1, 10, 3, key="vid_skip")
    if uploaded is None:
        st.markdown('<div class="empty-state">Unggah rekaman CCTV / video area kerja untuk dianalisis.</div>', unsafe_allow_html=True)
        return

    if st.button("▶ Proses video", type="primary", key="vid_start"):
        tmp_path = APP_DIR / f"_tmp_{uploaded.name}"
        tmp_path.write_bytes(uploaded.read())
        cap = cv2.VideoCapture(str(tmp_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        frame_slot = st.empty()
        progress = st.progress(0)
        summary_slot = st.empty()
        idx, session_violations, session_total = 0, 0, 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            if idx % frame_skip == 0:
                drawn, violations, total = run_detection(
                    model, frame, st.session_state.conf_threshold, st.session_state.iou_threshold, current_det_config()
                )
                frame_slot.image(drawn, channels="BGR", use_container_width=True)
                session_violations += violations
                session_total += total
                handle_violation_event(drawn, violations, total, uploaded.name)
                summary_slot.markdown(
                    f'<div class="status-row">{status_pill(f"{session_total} objek terdeteksi", "off")}'
                    f'{status_pill(f"{session_violations} pelanggaran", "alert" if session_violations else "on")}</div>',
                    unsafe_allow_html=True,
                )
            progress.progress(min(idx / total_frames, 1.0))

        cap.release()
        tmp_path.unlink(missing_ok=True)
        st.success("Pemrosesan video selesai.")


def detect_live(model, source, label):
    st.caption(
        "Mode ini mengakses kamera pada perangkat yang menjalankan aplikasi. "
        "Untuk deployment di server/cloud, kamera lokal tidak tersedia — gunakan tab **CCTV (RTSP)**."
    )
    run_key = f"run_{label.replace(' ', '_')}"
    start, stop = st.columns(2)
    if start.button("▶ Mulai " + label, type="primary", key=run_key + "_start"):
        st.session_state.webcam_running = True
    if stop.button("■ Berhenti", key=run_key + "_stop"):
        st.session_state.webcam_running = False

    frame_slot = st.empty()
    summary_slot = st.empty()

    if not st.session_state.webcam_running:
        st.markdown('<div class="empty-state">Kamera tidak aktif. Klik Mulai untuk memantau secara langsung.</div>', unsafe_allow_html=True)
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("Tidak bisa membuka kamera. Periksa apakah kamera sedang dipakai aplikasi lain atau izin akses kamera.")
        st.session_state.webcam_running = False
        return

    session_total, session_violations = 0, 0
    while cap.isOpened() and st.session_state.webcam_running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Sinyal kamera terputus.")
            break
        drawn, violations, total = run_detection(
            model, frame, st.session_state.conf_threshold, st.session_state.iou_threshold, current_det_config()
        )
        frame_slot.image(drawn, channels="BGR", use_container_width=True)
        session_total += total
        session_violations += violations
        handle_violation_event(drawn, violations, total, label)
        summary_slot.markdown(
            f'<div class="status-row">{status_pill(f"{session_total} objek terdeteksi", "off")}'
            f'{status_pill(f"{session_violations} pelanggaran", "alert" if session_violations else "on")}</div>',
            unsafe_allow_html=True,
        )
    cap.release()


def detect_cctv(model):
    st.caption("Masukkan URL stream RTSP/HTTP dari CCTV di lokasi kerja (mis. rtsp://user:pass@192.168.1.10:554/stream1).")
    rtsp_url = st.text_input("URL kamera CCTV", key="rtsp_url", placeholder="rtsp://...")
    if not rtsp_url:
        st.markdown('<div class="empty-state">Masukkan URL CCTV untuk memulai pemantauan jarak jauh.</div>', unsafe_allow_html=True)
        return
    detect_live(model, source=rtsp_url, label="CCTV")


def show_result_summary(violations, total):
    state = "alert" if violations else "on"
    st.markdown(
        f'<div class="status-row" style="margin-top:10px;">'
        f'{status_pill(f"{total} objek terdeteksi", "off")}'
        f'{status_pill(f"{violations} pelanggaran ditemukan" if violations else "Semua patuh", state)}'
        f"</div>",
        unsafe_allow_html=True,
    )


# ========================================================================
# HALAMAN: RIWAYAT PELANGGARAN
# ========================================================================
def page_history():
    df = load_log()
    if df.empty:
        st.markdown('<div class="empty-state">Belum ada riwayat pelanggaran. Log akan muncul di sini setelah deteksi berjalan.</div>', unsafe_allow_html=True)
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        date_range = st.date_input("Rentang tanggal", value=(df["waktu"].min().date(), df["waktu"].max().date()))
    with c2:
        sources = st.multiselect("Sumber", sorted(df["sumber"].unique()), default=sorted(df["sumber"].unique()))
    with c3:
        min_viol = st.number_input("Minimal jumlah pelanggaran", min_value=0, value=0)

    filtered = df.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[(filtered["waktu"].dt.date >= start) & (filtered["waktu"].dt.date <= end)]
    filtered = filtered[filtered["sumber"].isin(sources)]
    filtered = filtered[filtered["jumlah_pelanggaran"] >= min_viol]
    filtered = filtered.sort_values("waktu", ascending=False)

    st.markdown(f'<div class="panel"><div class="panel-title">{len(filtered)} kejadian ditemukan</div>', unsafe_allow_html=True)
    st.dataframe(
        filtered.rename(columns={
            "waktu": "Waktu", "lokasi": "Lokasi", "sumber": "Sumber",
            "jumlah_pelanggaran": "Pelanggaran", "total_terdeteksi": "Total Terdeteksi",
            "confidence_min": "Confidence", "snapshot": "Snapshot",
        }),
        use_container_width=True, hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("⬇ Unduh sebagai CSV", filtered.to_csv(index=False).encode("utf-8"), "riwayat_pelanggaran.csv", "text/csv")
    with col_dl2:
        excel_buf = io.BytesIO()
        filtered.to_excel(excel_buf, index=False, engine="openpyxl")
        st.download_button("⬇ Unduh sebagai Excel", excel_buf.getvalue(), "riwayat_pelanggaran.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with_snap = filtered[filtered["snapshot"].astype(str).str.len() > 0]
    if not with_snap.empty:
        st.markdown('<div class="panel"><div class="panel-title">Bukti foto terbaru</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (_, row) in enumerate(with_snap.head(8).iterrows()):
            snap_path = APP_DIR / row["snapshot"]
            if snap_path.exists():
                with cols[i % 4]:
                    st.image(str(snap_path), caption=row["waktu"].strftime("%d/%m %H:%M"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🗑 Hapus seluruh riwayat log"):
        st.session_state.confirm_clear = True
    if st.session_state.get("confirm_clear"):
        st.warning("Tindakan ini akan menghapus semua riwayat log secara permanen.")
        cc1, cc2 = st.columns(2)
        if cc1.button("Ya, hapus semua", type="primary"):
            LOG_FILE.unlink(missing_ok=True)
            load_log.clear()
            st.session_state.confirm_clear = False
            st.rerun()
        if cc2.button("Batal"):
            st.session_state.confirm_clear = False


# ========================================================================
# HALAMAN: PENGATURAN
# ========================================================================
def page_settings():
    st.markdown('<div class="panel"><div class="panel-title">Umum</div>', unsafe_allow_html=True)
    st.session_state.site_name = st.text_input("Nama lokasi / situs", st.session_state.site_name)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Model deteksi</div>', unsafe_allow_html=True)
    weights = available_weights()
    if weights:
        chosen = st.selectbox("Berkas bobot model (.pt) di folder aplikasi", weights)
        st.session_state.model_path = str(APP_DIR / chosen)
    else:
        st.info("Tidak ada berkas .pt ditemukan di folder aplikasi. Unggah salah satu di bawah ini.")
    uploaded_model = st.file_uploader("Atau unggah berkas model (.pt)", type=["pt"], key="model_uploader")
    if uploaded_model is not None:
        save_path = APP_DIR / uploaded_model.name
        save_path.write_bytes(uploaded_model.read())
        st.session_state.model_path = str(save_path)
        load_model.clear()
        st.success(f"Model '{uploaded_model.name}' disimpan dan diaktifkan.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Peringatan suara</div>', unsafe_allow_html=True)
    st.session_state.sound_enabled = st.toggle("Aktifkan alarm suara", st.session_state.sound_enabled)
    st.session_state.voice_text = st.text_area("Teks peringatan", st.session_state.voice_text)
    st.session_state.voice_cooldown = st.slider("Jeda antar peringatan (detik)", 2, 30, st.session_state.voice_cooldown)
    if not GTTS_READY:
        st.caption("Paket gTTS tidak terpasang — sistem otomatis memakai nada bip cadangan yang tidak butuh internet.")
    if st.button("🔊 Uji alarm"):
        audio_bytes, mime = generate_voice_bytes(st.session_state.voice_text)
        play_alert(audio_bytes, mime)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Bukti foto (snapshot)</div>', unsafe_allow_html=True)
    st.session_state.snapshot_enabled = st.toggle("Simpan foto otomatis saat pelanggaran terdeteksi", st.session_state.snapshot_enabled)
    st.caption(f"Foto disimpan di folder: {SNAPSHOT_DIR}")
    st.markdown("</div>", unsafe_allow_html=True)


# ========================================================================
# MAIN
# ========================================================================
def main():
    inject_theme()

    if not YOLO_READY:
        st.error("Paket **ultralytics** belum terpasang. Jalankan `pip install -r requirements.txt` lalu muat ulang aplikasi.")
        return

    model = None
    model_path = st.session_state.model_path or (str(APP_DIR / available_weights()[0]) if available_weights() else None)
    if model_path and Path(model_path).exists():
        try:
            model = load_model(model_path)
            st.session_state.model_path = model_path
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")

    render_header(model_ready=model is not None)

    with st.sidebar:
        st.markdown("#### Navigasi")
        page = st.radio(
            "menu", ["📊 Dashboard", "🔍 Deteksi Langsung", "📋 Riwayat Pelanggaran", "⚙️ Pengaturan"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("HelmGuard v1.0 · Dibangun dengan Ultralytics YOLO + Streamlit")

    if model is None:
        st.warning("Model deteksi belum tersedia. Buka menu **Pengaturan** untuk memilih atau mengunggah berkas model (.pt).")
        if page == "⚙️ Pengaturan":
            page_settings()
        return

    if page == "📊 Dashboard":
        page_dashboard()
    elif page == "🔍 Deteksi Langsung":
        page_detection(model)
    elif page == "📋 Riwayat Pelanggaran":
        page_history()
    elif page == "⚙️ Pengaturan":
        page_settings()


if __name__ == "__main__":
    main()
