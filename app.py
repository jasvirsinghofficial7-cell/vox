"""

app.py — VoxAuth Master Application (Biometric Security Edition)

Features: NUCLEAR Dark Blue/Cyan Theme, Multi-Lingual Wizard, App Vault, Noise Reduction, Animations

"""

import json

import os

import multiprocessing

import queue

import threading

import time

import numpy as np

import torch

import torchaudio

import io

import noisereduce as nr



# 🚨 TORCHAUDIO FIX: Prevents "Failed to fetch dynamically imported module" error

if not hasattr(torchaudio, 'list_audio_backends'):

    torchaudio.list_audio_backends = lambda: ["soundfile"]

if not hasattr(torchaudio, 'get_audio_backend'):

    torchaudio.get_audio_backend = lambda: "soundfile"



import sounddevice as sd

import streamlit as st

from streamlit_option_menu import option_menu

from scipy.spatial.distance import cosine

from speechbrain.inference.speaker import EncoderClassifier

from speechbrain.inference.separation import SepformerSeparation



# Our custom DB layer (Ensure database.py is in the same folder)

from database import (

    init_db, register_user, authenticate_user,

    load_voiceprint_from_db, save_voiceprint_to_db,

    update_user_profile, load_user_profile,
    
)

from audio_utils import normalize_audio, audio_to_embedding, get_vbcable_device_id, DEVICE, COMPUTE_NODE





# 🚀 DYNAMIC COMPUTE ALLOCATION 🚀

# DEVICE and COMPUTE_NODE are imported from audio_utils



os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

if DEVICE == "cpu":

    os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())

    torch.set_num_threads(multiprocessing.cpu_count())



# ─────────────────────────────────────────────────────────────────

# PAGE CONFIG & CSS (Aggressive Animated Dark Theme)

# ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="VoxAuth – Biometric Security", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")



# 🔥 NUCLEAR CSS OVERRIDE 🔥

st.markdown("""

<style>

    /* Force Background App & Main Container */

    [data-testid="stAppViewContainer"], .stApp {

        background-color: #0b0c10 !important;

        background-image: radial-gradient(circle at 50% 0%, #151b2b 0%, #0b0c10 60%) !important;

        color: #e5e7eb !important;

    }

   

    /* Force Header to be transparent */

    [data-testid="stHeader"] { background: transparent !important; }

   

    /* Force Sidebar Dark */

    [data-testid="stSidebar"] {

        background-color: #0f131c !important;

        border-right: 1px solid #1e2638 !important;

    }



    /* Typography Overrides */

    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-family: 'Inter', sans-serif; letter-spacing: 0.5px; font-weight: 700; }

    p, label, .stMarkdown, .stText { color: #9ba3b5 !important; }

   

    /* Input Fields - Sleek & Dark */

    .stTextInput > div > div > input, .stSelectbox > div > div > div {

        background-color: #11151e !important;

        border: 1px solid #232b3d !important;

        border-radius: 10px !important;

        color: #ffffff !important;

        padding-left: 15px !important;

    }

    .stTextInput > div > div > input:focus { border: 1px solid #3273f6 !important; box-shadow: 0 0 8px rgba(50, 115, 246, 0.3) !important; }

   

    /* Premium Blue Buttons */

    .stButton > button {

        background: linear-gradient(135deg, #3273f6 0%, #1d4ed8 100%) !important;

        color: #ffffff !important;

        border: none !important;

        border-radius: 10px !important;

        font-weight: 600 !important;

        padding: 12px 24px !important;

        box-shadow: 0 4px 14px rgba(50, 115, 246, 0.3) !important;

        transition: all 0.3s ease !important;

    }

    .stButton > button:hover {

        background: linear-gradient(135deg, #4a84f7 0%, #2563eb 100%) !important;

        box-shadow: 0 6px 20px rgba(50, 115, 246, 0.5) !important;

        transform: translateY(-2px) !important;

    }

   

    /* Live Box (Cards) */

    .live-box {

        background: #11151e !important;

        padding: 28px !important;

        border-radius: 16px !important;

        border: 1px solid rgba(255, 255, 255, 0.1) !important;

        box-shadow: 0 8px 24px rgba(0,0,0,0.6) !important;

        margin-bottom: 25px !important;

    }

    .live-box:hover { border: 1px solid rgba(50, 115, 246, 0.5) !important; }

   

    /* Metric Cards */

    [data-testid="stMetricValue"] { font-size: 2.5rem !important; color: #0df0e3 !important; text-shadow: 0 0 15px rgba(13, 240, 227, 0.2) !important; font-weight: 800 !important; }

    [data-testid="stMetricLabel"] { font-size: 1.0rem !important; color: #8b929e !important; text-transform: uppercase !important; }

   

    /* Progress Bars */

    .stProgress > div > div > div > div { background: linear-gradient(90deg, #3273f6, #0df0e3) !important; }

    .stProgress > div > div { background-color: #1e2638 !important; border-radius: 8px !important; }



    /* Custom Radio Buttons */

    div.row-widget.stRadio > div { background: #11151e !important; padding: 15px !important; border-radius: 12px !important; border: 1px solid #1e2638 !important; color: white !important;}

    hr { border-color: #232b3d !important; }

   

    /* 🔴 ANIMATIONS 🔴 */

   

    @keyframes pulse-border {

        0% { box-shadow: 0 0 0 0 rgba(50, 115, 246, 0.4); }

        70% { box-shadow: 0 0 0 15px rgba(50, 115, 246, 0); }

        100% { box-shadow: 0 0 0 0 rgba(50, 115, 246, 0); }

    }

    .pulsing-box {

        animation: pulse-border 2s infinite !important;

        border: 1px solid #3273f6 !important;

    }

   

    @keyframes scan-text {

        0%, 100% { opacity: 1; text-shadow: 0 0 10px #0df0e3; }

        50% { opacity: 0.4; text-shadow: none; }

    }

    .scanning-text {

        animation: scan-text 1.5s infinite !important;

        color: #0df0e3 !important;

        font-weight: bold !important;

    }

</style>

""", unsafe_allow_html=True)



init_db()



# Session States initialization

if "logged_in" not in st.session_state: st.session_state.logged_in = False

if "current_user" not in st.session_state: st.session_state.current_user = None

if "enroll_eng" not in st.session_state: st.session_state.enroll_eng = None

if "enroll_hin" not in st.session_state: st.session_state.enroll_hin = None

if "enroll_opt" not in st.session_state: st.session_state.enroll_opt = None

if "monitoring" not in st.session_state: st.session_state.monitoring = False



# Naya Session State for App Locker Toggles 🔥

if "app_locks" not in st.session_state:

    st.session_state.app_locks = {

        "Google Chrome": {"icon": "🌐", "locked": False, "process": "chrome.exe"},

        "Notepad": {"icon": "📝", "locked": True, "process": "notepad.exe"},

        "VS Code": {"icon": "💻", "locked": False, "process": "Code.exe"},

        "Command Prompt": {"icon": "⚙️", "locked": False, "process": "cmd.exe"},

        "WhatsApp": {"icon": "💬", "locked": True, "process": "WhatsApp.exe"}

    }



# ─────────────────────────────────────────────────────────────────

# SHARED UTILITIES & AI MODELS

# ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=f"🧠 Initializing Neural Engine on {COMPUTE_NODE}...")

def load_ai_model():

    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": DEVICE})



@st.cache_resource(show_spinner=f"🎧 Initializing Sepformer AI on {COMPUTE_NODE}...")

def load_separation_model():

    return SepformerSeparation.from_hparams(source="speechbrain/sepformer-wsj02mix", run_opts={"device": DEVICE})



def record_audio_with_progress(duration: int, fs: int = 16000) -> np.ndarray:

    progress_text = f"🎤 Recording {duration}s... Please speak now."

    bar = st.progress(0, text=progress_text)

    try:

        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")

        for pct in range(100):

            time.sleep(duration / 100)

            bar.progress(pct + 1, text=progress_text)

        sd.wait()

        bar.empty()

        return recording.flatten()

    except Exception as e:

        bar.empty()

        st.error(f"⚠️ Microphone Error: {e}")

        return np.zeros(int(duration * fs), dtype="float32")



# normalize_audio, audio_to_embedding, get_vbcable_device_id are imported from audio_utils



# ─────────────────────────────────────────────────────────────────

# PAGE A: AUTHENTICATION

# ─────────────────────────────────────────────────────────────────

def show_auth_page():

    st.markdown("<h1 style='text-align:center;font-size:3.5rem;margin-bottom:0;'>🎙️ VoxAuth</h1>", unsafe_allow_html=True)

    st.markdown("<p style='text-align:center;color:#0df0e3;letter-spacing:2px;text-transform:uppercase;'>Neural Biometric Security</p>", unsafe_allow_html=True)

    st.write("")

    _, mid, _ = st.columns([1, 1.2, 1])

    with mid:

        tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

        with tab_login:

            username_l = st.text_input("Security ID", key="login_user")

            password_l = st.text_input("Passcode", key="login_pass", type="password")

            if st.button("Access Command Center →", use_container_width=True):

                ok, msg = authenticate_user(username_l, password_l)

                if ok:

                    st.session_state.logged_in = True

                    st.session_state.current_user = username_l.strip().lower()

                    with open("active_user.txt", "w") as f: f.write(st.session_state.current_user)


                    st.rerun()

                else: st.error(msg)

        with tab_register:

            username_r = st.text_input("Choose Security ID", key="reg_user")
            
            mobile_r = st.text_input("Mobile Number", key="reg_mobile", placeholder="+1234567890")

            password_r = st.text_input("Set Passcode", key="reg_pass", type="password")

            confirm_r  = st.text_input("Confirm Passcode", key="reg_confirm", type="password")

            if st.button("Initialize Profile →", use_container_width=True):

                if password_r == confirm_r:

                    ok, msg = register_user(username_r, password_r, mobile_r)

                    if ok: st.success(msg)

                    else: st.error(msg)

                else: st.error("Passwords do not match!")



# ─────────────────────────────────────────────────────────────────
# PAGE B: ENROLLMENT WIZARD
# ─────────────────────────────────────────────────────────────────
def show_enrollment_page(classifier, username: str):
    st.title("🛡️ Identity Onboarding Wizard")
    st.write("Establish your biometric baseline. English and Hindi are mandatory for robust security.")
    
    existing_vp = load_voiceprint_from_db(username)
    
    if existing_vp is None:
        if st.session_state.enroll_eng is None:
            st.markdown("### Step 1/3: Mandatory English Baseline")
            st.info("Read these 2 sentences clearly:\n1. The quick brown fox jumps over the lazy dog.\n2. My voice is my password, and it continuously secures my digital identity.")
            if st.button("🎤 Record English (15s)"):
                raw = record_audio_with_progress(15, 16000)
                if np.max(np.abs(raw)) > 0:
                    clean_raw = nr.reduce_noise(y=raw, sr=16000)
                    st.session_state.enroll_eng = audio_to_embedding(normalize_audio(clean_raw), classifier)
                    st.success("✅ English profile captured!")
                    st.rerun()

        elif st.session_state.enroll_hin is None:
            st.markdown("### Step 2/3: Mandatory Hindi Baseline")
            st.info(f"Read these 2 sentences clearly:\n1. नमस्ते, मेरा नाम {username} है और मैं इस सिस्टम को अपनी आवाज़ दे रहा हूँ।\n2. मेरी आवाज़ ही मेरी सबसे बड़ी पहचान और मेरा सुरक्षित पासवर्ड है।")
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🎤 Record Hindi (15s)"):
                    raw = record_audio_with_progress(15, 16000)
                    if np.max(np.abs(raw)) > 0:
                        clean_raw = nr.reduce_noise(y=raw, sr=16000)
                        st.session_state.enroll_hin = audio_to_embedding(normalize_audio(clean_raw), classifier)
                        st.success("✅ Hindi profile captured!")
                        st.rerun()
            with col2:
                if st.button("🔄 Retake English", key="retake_eng"):
                    st.session_state.enroll_eng = None
                    st.rerun()

        else:
            st.markdown("### Step 3/3: Optional Regional Language (Bonus Accuracy)")
            scripts = {
                "Skip (None)": "",
                "Punjabi": f"1. ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ, ਮੇਰਾ ਨਾਮ {username} ਹੈ।\n2. ਮੇਰੀ ਆਵਾਜ਼ ਹੀ ਮੇਰੀ ਪਛਾਣ ਅਤੇ ਮੇਰਾ ਪਾਸਵਰਡ ਹੈ.",
                "Marathi": f"1. नमस्कार, माझे नाव {username} आहे.\n2. माझा आवाज हीच माझी ओळख आणि माझा पासवर्ड आहे.",
                "Bengali": f"1. নমস্কার, আমার নাম {username}।\n2. আমার কণ্ঠস্বরই আমার পরিচয় এবং আমার পাসওয়ার্ড।",
                "Gujarati": f"1. નમસ્તે, મારું નામ {username} છે.\n2. મારો અવાજ મારી ઓળખ અને મારો પાસવર્ડ છે."
            }
            lang = st.selectbox("Select Additional Language (Optional):", list(scripts.keys()))
            
            if lang != "Skip (None)":
                st.info(f"Read these 2 sentences clearly:\n**{scripts[lang]}**")
                if st.button(f"🎤 Record {lang} (15s)"):
                    raw = record_audio_with_progress(15, 16000)
                    if np.max(np.abs(raw)) > 0:
                        clean_raw = nr.reduce_noise(y=raw, sr=16000)
                        st.session_state.enroll_opt = audio_to_embedding(normalize_audio(clean_raw), classifier)
                        st.success(f"✅ {lang} profile captured!")

            st.markdown("---")
            if st.button("🔐 Encrypt & Lock Final Identity Vector", use_container_width=True):
                embeddings_to_fuse = [st.session_state.enroll_eng, st.session_state.enroll_hin]
                if st.session_state.enroll_opt is not None: embeddings_to_fuse.append(st.session_state.enroll_opt)
                master_emb = np.mean(embeddings_to_fuse, axis=0)
                save_voiceprint_to_db(username, master_emb)
                st.balloons()
                st.success("🎉 Identity Locked Successfully! System Ready.")
                st.rerun()
    else:
        st.success("✅ Master Voiceprint is Active & Secured.")
        st.divider()
        st.subheader("🚀 Enhance System Accuracy")
        st.write("Kya aap apni voice authentication ko aur better banane ke liye aur sample dena chahte hain?")
        
        if st.button("🎤 Record Bonus Sample (30s)"):
            st.info("Speak freely or read any text for the next 30 seconds...")
            bonus_raw = record_audio_with_progress(30, 16000)
            if np.max(np.abs(bonus_raw)) > 0:
                with st.spinner("Processing deep neural features..."):
                    clean_bonus = nr.reduce_noise(y=bonus_raw, sr=16000)
                    bonus_emb = audio_to_embedding(normalize_audio(clean_bonus), classifier)
                    new_master_emb = np.mean([existing_vp, bonus_emb], axis=0)
                    save_voiceprint_to_db(username, new_master_emb)
                    st.success("🎉 Voiceprint Upgraded! Aapki profile ab aur bhi stronger hai.")
        
        st.divider()
        if st.button("⚠️ Reset Voiceprint Profile"):
            st.session_state.enroll_eng = None
            st.session_state.enroll_hin = None
            st.session_state.enroll_opt = None
            st.warning("Profile reset initialized. Contact admin to wipe DB record.")


# ─────────────────────────────────────────────────────────────────
# PAGE C: VERIFICATION
# ─────────────────────────────────────────────────────────────────
def show_verification_page(classifier, username: str):
    st.title("🔐 Access Verification")
    st.write("Standard rapid biometric handshake.")

    vp = load_voiceprint_from_db(username)
    if vp is None: return st.error("❌ Action Required: Enroll Identity First.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Handshake Protocol")
        if st.button("🎙️ Initialize Scan (5s)", use_container_width=True):
            status_ph = st.empty()
            status_ph.markdown("<div class='live-box pulsing-box'><span class='scanning-text'>⏳ Listening for biometric signature...</span></div>", unsafe_allow_html=True)
            raw_audio = record_audio_with_progress(5, 16000)
            
            status_ph.markdown(f"<div class='live-box'><span style='color:#3273f6;'>⚙️ Extracting Neural Features on {COMPUTE_NODE}...</span></div>", unsafe_allow_html=True)
            clean_audio = nr.reduce_noise(y=raw_audio, sr=16000)
            live_emb = audio_to_embedding(normalize_audio(clean_audio), classifier)
            similarity = 1 - cosine(vp.flatten(), live_emb.flatten())
            pct = similarity * 100

            status_ph.empty()
            st.markdown("### Telemetry")
            mc1, mc2 = st.columns(2)
            mc1.metric("Match Confidence", f"{pct:.1f}%")
            st.progress(min(int(pct), 100))

            if similarity >= 0.65:
                mc2.metric("Status", "VERIFIED", delta="Secure")
                st.success(f"ACCESS GRANTED. Welcome, {username.upper()}")
            else:
                mc2.metric("Status", "DENIED", delta="-Mismatch", delta_color="inverse")
                st.error("ACCESS DENIED. Signature conflict.")

    with col2:
        st.markdown(f"""
<div class='live-box'>
    <h4 style='color:#3273f6; margin-top:0;'>Node Status</h4>
    <hr>
    <p style='margin:5px 0;'>👤 User: <b style="color:white;">{username.upper()}</b></p>
    <p style='margin:5px 0;'>🟢 Engine: <b style="color:white;">Standard Auth</b></p>
    <p style='margin:5px 0;'>🔒 Encryption: <b style="color:white;">AES-256</b></p>
</div>
""", unsafe_allow_html=True)


def save_locks_to_file():
    # Fix: Pura ka pura dictionary waise hi save karo jaisa watcher ko chahiye
    with open("app_locks.json", "w") as f:
        json.dump(st.session_state.app_locks, f)
# ─────────────────────────────────────────────────────────────────
# PAGE D: APP VAULT (🔥 ZERO-TRUST APP LOCKER 🔥)
# ─────────────────────────────────────────────────────────────────
def show_app_locker_page(classifier, username: str):
    st.title("🛡️ Zero-Trust App Vault")
    st.write("Toggle switches to enable strict Voice Biometric locks on specific OS applications.")
    
    vp = load_voiceprint_from_db(username)
    if vp is None:
        return st.error("❌ You must enroll your Voiceprint before securing applications.")

    # Save initial state
    save_locks_to_file()

    st.markdown("<br>", unsafe_allow_html=True)

    for app_name, app_data in st.session_state.app_locks.items():
        with st.container():
            # 🔥 Fix: 3 columns ki jagah ab sirf 2 columns rakhe hain (Launch button hata diya)
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"#### {app_data['icon']} {app_name}")
                st.caption(f"Target Process: `{app_data['process']}`")
            
            with col2:
                current_state = app_data["locked"]
                new_state = st.toggle("Secure", value=current_state, key=f"tog_{app_name}")
                
                if new_state != current_state:
                    st.session_state.app_locks[app_name]["locked"] = new_state
                    save_locks_to_file() # File update taaki watcher ko pata chale
                    status_text = "LOCKED 🔒" if new_state else "UNLOCKED 🔓"
                    st.toast(f"{app_name} is now {status_text}", icon="✅")
                    st.rerun()

    st.divider()
    st.info("💡 Tip: Just open these apps normally from your Desktop or Start Menu. The Background OS Watcher will automatically intercept and lock them!")

# ─────────────────────────────────────────────────────────────────

# PAGE E: VoxAuth with Voice Sep (🔥 LIVE ENGINE WITH TOGGLES 🔥)

# ─────────────────────────────────────────────────────────────────

def show_voxauth_live_page(classifier, separator, username: str):

    st.title("🛡️ VoxAuth Secure Stream Architecture")

    st.write("Real-time telemetry and advanced stream isolation protocol.")

   

    vp = load_voiceprint_from_db(username)

    if vp is None: return st.error("❌ Identity Vector Missing. Please enroll.")

   

    cable_id = get_vbcable_device_id()

    cable_status = f"🟢 <b style='color:#0df0e3;'>VB-Cable Linked</b>" if cable_id is not None else "⚠️ <b style='color:orange;'>Local Audio Fallback (Speakers)</b>"

   

    st.markdown("<div style='margin-bottom: 25px;'>", unsafe_allow_html=True)

    auth_mode = st.radio(

        "Select Operation Protocol:",

        [

            "⚡ Protocol Alpha: Voice Authentication WITHOUT Separation",

            "🛡️ Protocol Beta: Voice Authentication WITH Separation"

        ],

        horizontal=True

    )

    st.markdown("</div>", unsafe_allow_html=True)



    col1, col2 = st.columns([2, 1])



    with col1:

        st.markdown("### Execution Control")

        toggle_state = st.toggle("🚀 Power On VoxAuth Engine", value=st.session_state.monitoring)

       

        if toggle_state != st.session_state.monitoring:

            st.session_state.monitoring = toggle_state

            if not toggle_state:

                try: sd.stop()

                except: pass

            st.rerun()



        st.markdown("<br>### Live Data Stream", unsafe_allow_html=True)

        mc1, mc2, mc3 = st.columns(3)

        score_metric = mc1.empty()

        action_metric = mc2.empty()

        latency_metric = mc3.empty()

       

        score_metric.metric("Confidence Index", "0.0%")

        action_metric.metric("Gateway Status", "Standby")

        latency_metric.metric("Processing Latency", "0 ms")



        ui_box = st.empty()

        if not st.session_state.monitoring:

            ui_box.markdown(f"<div class='live-box'>{cable_status}<br><br><span style='color:#8b929e;'>Status: Toggle switch to initiate command...</span></div>", unsafe_allow_html=True)

        else:

            ui_box.markdown(f"<div class='live-box pulsing-box'>{cable_status}<br><br><span class='scanning-text'>Engine Active: Awaiting acoustic input...</span></div>", unsafe_allow_html=True)



    with col2:

        engine_status = "Alpha (TDNN)" if "Alpha" in auth_mode else "Beta (Sepformer)"

        latency_warning = "Real-Time" if DEVICE == "cuda" else "High Delay (Local CPU)"

       

        st.markdown(f"""
<div class='live-box' style='height: 100%;'>
    <h4 style='color:#3273f6; margin-top:0;'>System Architecture</h4>
    <hr>
    <p>👤 Entity: <b style="color:white;">{username.upper()}</b></p>
    <p>🟢 Protocol: <b style="color:white;">{engine_status}</b></p>
    <p>⚙️ Compute Node: <b style='color:#0df0e3;'>{COMPUTE_NODE}</b></p>
    <p>⏱️ Expected Latency: <b style="color:white;">{latency_warning if 'Beta' in auth_mode else 'Real-Time'}</b></p>
</div>
""", unsafe_allow_html=True)



    if st.session_state.monitoring:

        audio_queue, playback_queue = queue.Queue(), queue.Queue()



        def player_thread():

            out_device = cable_id if cable_id is not None else None

            try:

                with sd.OutputStream(samplerate=16000, channels=1, dtype="float32", device=out_device) as stream:

                    while True:

                        audio = playback_queue.get()

                        if audio is None: break

                        stream.write(audio)

            except Exception as e: print(f"Playback Error: {e}")

               

        threading.Thread(target=player_thread, daemon=True).start()



        def audio_cb(indata, frames, time_info, status):

            audio_queue.put(indata.copy().flatten())



        CHUNK_DURATION = 1.5

        VERIFY_THRESH = 0.45

        HOLD_THRESH = 0.40



        try:

            with sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=int(CHUNK_DURATION * 16000), callback=audio_cb):

                while st.session_state.monitoring:

                    chunk = audio_queue.get()

                    start_proc = time.time()

                   

                    if np.max(np.abs(chunk)) < 0.005:

                        score_metric.metric("Confidence Index", "0.0%")

                        action_metric.metric("Gateway Status", "Monitoring")

                        latency_metric.metric("Processing Latency", "0 ms")

                        ui_box.markdown(f"<div class='live-box pulsing-box'>{cable_status}<br><br><span class='scanning-text'>Acoustic State: Sub-threshold (Silence)</span></div>", unsafe_allow_html=True)

                        continue



                    # 🔊 Noise Reduction

                    clean_chunk = nr.reduce_noise(y=chunk, sr=16000, prop_decrease=0.8)



                    if auth_mode == "⚡ Protocol Alpha: Voice Authentication WITHOUT Separation":

                        ui_box.markdown(f"<div class='live-box pulsing-box' style='border-left: 4px solid #3273f6;'>{cable_status}<br><br><span class='scanning-text'>Analyzing Acoustic Signature...</span></div>", unsafe_allow_html=True)

                        live_emb = audio_to_embedding(normalize_audio(clean_chunk), classifier)

                        best_sim = 1 - cosine(vp.flatten(), live_emb.flatten())

                        owner_track = clean_chunk

                       

                    else:

                        compute_msg = "Deep Neural Isolation via GPU..." if DEVICE == "cuda" else "Simulating Deep Isolation on CPU..."

                        ui_box.markdown(f"<div class='live-box pulsing-box' style='border-left: 4px solid #3273f6;'>{cable_status}<br><br><span class='scanning-text'>{compute_msg}</span></div>", unsafe_allow_html=True)

                       

                        chunk_tensor = torch.from_numpy(clean_chunk).float().unsqueeze(0).to(DEVICE)

                        est_sources = separator.separate_batch(chunk_tensor)

                       

                        src1 = est_sources[:, :, 0].detach().cpu().numpy().flatten()

                        src2 = est_sources[:, :, 1].detach().cpu().numpy().flatten()

                       

                        sim1 = 1 - cosine(vp.flatten(), audio_to_embedding(normalize_audio(src1), classifier).flatten())

                        sim2 = 1 - cosine(vp.flatten(), audio_to_embedding(normalize_audio(src2), classifier).flatten())

                       

                        if sim1 > sim2: owner_track, best_sim = src1, sim1

                        else: owner_track, best_sim = src2, sim2



                    pct = best_sim * 100

                    latency_ms = int((time.time() - start_proc) * 1000)

                   

                    score_metric.metric("Confidence Index", f"{pct:.1f}%")

                    latency_metric.metric("Processing Latency", f"{latency_ms} ms")

                   

                    if best_sim >= VERIFY_THRESH:

                        action_metric.metric("Gateway Status", "🟢 VERIFIED")

                        ui_box.markdown(f"<div class='live-box' style='background: rgba(76, 175, 80, 0.1); border: 1px solid #4caf50;'>{cable_status}<br><br><b style='color:#4caf50; font-size: 1.2rem;'>✅ IDENTITY CONFIRMED (>43%).</b> Audio Delivery Active.</div>", unsafe_allow_html=True)

                        playback_queue.put(owner_track)

                    elif best_sim >= HOLD_THRESH:

                        action_metric.metric("Gateway Status", "🟡 SUSTAINING")

                        ui_box.markdown(f"<div class='live-box' style='background: rgba(255, 193, 7, 0.1); border: 1px solid #ffc107;'>{cable_status}<br><br><b style='color:#ffc107; font-size: 1.2rem;'>⚠️ BUFFERING (>35%).</b> Maintaining Connection flow.</div>", unsafe_allow_html=True)

                        playback_queue.put(owner_track)

                    else:

                        action_metric.metric("Gateway Status", "🔴 BLOCKED")

                        ui_box.markdown(f"<div class='live-box' style='background: rgba(244, 67, 54, 0.1); border: 1px solid #f44336;'>{cable_status}<br><br><b style='color:#f44336; font-size: 1.2rem;'>🚨 UNAUTHORIZED VOICE (<35%).</b> Stream Muted.</div>", unsafe_allow_html=True)



            playback_queue.put(None)

        except Exception as e:

            st.error(f"⚠️ Architecture Failure. Diagnostics: {e}")

            st.session_state.monitoring = False



# ─────────────────────────────────────────────────────────────────
# PAGE F: PROFILE SETTINGS
# ─────────────────────────────────────────────────────────────────
def show_profile_page(username: str):
    st.title("👤 Profile Settings")
    st.write("Update your profile information.")
    
    profile = load_user_profile(username)
    current_mobile = profile.get("mobile_number") or ""
    
    with st.container():
        st.markdown("<div class='live-box'>", unsafe_allow_html=True)
        with st.form("profile_update_form"):
            new_username = st.text_input("Username", value=username)
            new_mobile = st.text_input("Mobile Number", value=current_mobile, placeholder="+1234567890")
            new_password = st.text_input("New Passcode (leave blank to keep current)", type="password")
            confirm_password = st.text_input("Confirm New Passcode", type="password")
            
            if st.form_submit_button("Update Profile", use_container_width=True):
                if new_password and new_password != confirm_password:
                    st.error("Passwords do not match!")
                else:
                    ok, msg = update_user_profile(username, new_username, new_password, new_mobile)
                    if ok:
                        st.success(msg)
                        if new_username.strip().lower() != username.strip().lower():
                            st.session_state.current_user = new_username.strip().lower()
                            with open("active_user.txt", "w") as f: 
                                f.write(st.session_state.current_user)
                            # We can keep them logged in but force complete refresh of UI
                            st.rerun()
                    else:
                        st.error(msg)
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────

# MAIN ROUTER (Sidebar Setup)

# ─────────────────────────────────────────────────────────────────

if not st.session_state.logged_in:

    show_auth_page()

else:

    with st.sidebar:

        st.markdown(

            f"<div style='padding:20px 0;text-align:center;'>"

            f"<h2 style='color:#ffffff; margin:0;'>VoxAuth Dashboard</h2>"

            f"<p style='color:#3273f6; font-size:0.9rem; letter-spacing:1px; margin-top:0;'>{st.session_state.current_user.upper()}</p>"

            f"</div>",

            unsafe_allow_html=True,

        )

       

        # 🌌 NAV MENU (App Vault added here!) 🌌

        selected = option_menu(

            "Modules",

            ["Profile", "Enrollment", "Verification", "App Vault", "VoxAuth with Voice Sep"],

            icons=["person", "fingerprint", "shield-lock", "grid", "activity"],

            default_index=4,

            styles={

                "container": { "padding": "5px!important", "background-color": "transparent" },

                "icon": { "color": "#6b7280", "font-size": "18px" },

                "nav-link": {

                    "font-size": "14px", "text-align": "left", "margin": "8px 0px", "padding": "12px",

                    "color": "#9ca3af", "background-color": "transparent", "border-radius": "8px",

                    "font-weight": "500", "transition": "all 0.3s",

                },

                "nav-link-selected": {

                    "background-color": "rgba(50, 115, 246, 0.15)", "color": "#3273f6", "font-weight": "700",

                    "border-left": "4px solid #3273f6"

                },

            }

        )

        st.write("")

        st.write("")

        if st.button("Logout", use_container_width=True):

            st.session_state.logged_in = False

            st.session_state.enroll_eng = None

            st.session_state.enroll_hin = None

            st.session_state.enroll_opt = None

            st.rerun()



    classifier = load_ai_model()

    separator = load_separation_model()

    user = st.session_state.current_user



    # Page Routing

    if selected == "Profile": show_profile_page(user)

    elif selected == "Enrollment": show_enrollment_page(classifier, user)

    elif selected == "Verification": show_verification_page(classifier, user)

    elif selected == "App Vault": show_app_locker_page(classifier, user)

    elif selected == "VoxAuth with Voice Sep": show_voxauth_live_page(classifier, separator, user)
