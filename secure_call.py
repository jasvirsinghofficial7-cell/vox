import streamlit as st
import time
import numpy as np
import torch
import sounddevice as sd
import queue
import threading
import noisereduce as nr
from scipy.spatial.distance import cosine

from database import load_voiceprint_from_db, get_user_ip
from audio_utils import normalize_audio, audio_to_embedding, get_vbcable_device_id, COMPUTE_NODE, DEVICE
from network_utils import AudioServer, AudioClient, get_local_ip

def show_secure_call_page(classifier, separator, username: str):
    st.title("📞 Secure Call Interface")
    st.write("Real-time voice separation and authentication for active communication lines.")

    vp = load_voiceprint_from_db(username)
    if vp is None:
        return st.error("❌ Identity Vector Missing. Please enroll before making a secure call.")

    if "call_active" not in st.session_state:
        st.session_state.call_active = False
    if "network_mode" not in st.session_state:
        st.session_state.network_mode = "Host Call"
    if "target_ip" not in st.session_state:
        st.session_state.target_ip = ""
    if "lookup_username" not in st.session_state:
        st.session_state.lookup_username = ""

    cable_id = get_vbcable_device_id()
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Network Configuration")
        st.info("💡 Establish P2P Network Line.")
        
        net_mode = st.radio("Mode:", ["Host Call", "Join Call"], horizontal=True)
        st.session_state.network_mode = net_mode
        
        target_ip_input = ""
        if net_mode == "Host Call":
            local_ip = get_local_ip()
            st.success(f"📺 Host Mode: Your secure IP is **{local_ip}**")
        else:
            st.markdown("Lookup target by Username or manually enter IP:")
            c1, c2 = st.columns([2, 1])
            lookup_u = c1.text_input("Target Username", value=st.session_state.lookup_username, label_visibility="collapsed", placeholder="Enter Username")
            if lookup_u != st.session_state.lookup_username:
                st.session_state.lookup_username = lookup_u
            
            if c2.button("Lookup IP"):
                res_ip = get_user_ip(st.session_state.lookup_username)
                if res_ip:
                    st.session_state.target_ip = res_ip
                    st.success(f"Found IP: {res_ip}")
                else:
                    st.error("User not found or IP unknown.")

            target_ip_input = st.text_input("Enter Target IP Address:", placeholder="192.168.x.x", value=st.session_state.target_ip)
            st.session_state.target_ip = target_ip_input
            
        st.markdown("<br>### Call Engine Control", unsafe_allow_html=True)
        
        # Start/End Call Toggle
        disable_start = (net_mode == "Join Call" and not target_ip_input.strip())
        is_calling = st.toggle("🚀 Boot Secure Transmission Engine", value=st.session_state.call_active, disabled=disable_start)
        
        if is_calling != st.session_state.call_active:
            st.session_state.call_active = is_calling
            if not is_calling:
                try: 
                    sd.stop()
                    if "network_node" in st.session_state:
                        st.session_state.network_node.stop()
                        del st.session_state.network_node
                except: 
                    pass
            st.rerun()

        st.markdown("<br>### Secure Line Status", unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns(3)
        conf_metric = mc1.empty()
        caller_metric = mc2.empty()
        speaker_metric = mc3.empty()
        
        conf_metric.metric("Match Confidence", "0.0%")
        caller_metric.metric("Line Status", "Disconnected")
        speaker_metric.metric("Active Speaker", "None")

        call_box = st.empty()
        
        if not st.session_state.call_active:
            call_box.markdown(
                "<div class='live-box'><span style='color:#8b929e;'>Call inactive. Toggle above to initiate connection...</span></div>", 
                unsafe_allow_html=True
            )
        else:
            call_box.markdown(
                "<div class='live-box pulsing-box'><span class='scanning-text'>Establishing Secure Line... Listening...</span></div>", 
                unsafe_allow_html=True
            )

    with col2:
        net_target = "Listening..." if st.session_state.network_mode == "Host Call" else st.session_state.target_ip
        st.markdown(f"""
        <div class='live-box' style='height: 100%;'>
            <h4 style='color:#3273f6; margin-top:0;'>Call Architecture</h4>
            <hr>
            <p>👤 Owner: <b style="color:white;">{username.upper()}</b></p>
            <p>📡 Route: <b style="color:white;">{net_target if net_target else 'Unresolved'}</b></p>
            <p>🛡️ Engine: <b style="color:white;">Sepformer P2P</b></p>
            <p>⚙️ Processor: <b style='color:#0df0e3;'>{COMPUTE_NODE}</b></p>
            <p>🔊 Output: <b style='color:orange;'>{"VB-Cable" if cable_id is not None else "Local Speakers"}</b></p>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.call_active:
        if "network_node" not in st.session_state:
            # Initialize Network
            if st.session_state.network_mode == "Host Call":
                st.session_state.network_node = AudioServer(host_ip="0.0.0.0", port=5005)
            else:
                st.session_state.network_node = AudioClient(target_ip=st.session_state.target_ip, port=5005)
            
        network_node = st.session_state.network_node
        
        audio_queue = queue.Queue()
        playback_queue = network_node.receive_queue

        def player_thread():
            out_device = cable_id if cable_id is not None else None
            try:
                with sd.OutputStream(samplerate=16000, channels=1, dtype="float32", device=out_device) as stream:
                    while st.session_state.call_active:
                        audio = playback_queue.get()
                        if audio is None: 
                            break
                        stream.write(audio)
            except Exception as e:
                print(f"Playback Error: {e}")

        threading.Thread(target=player_thread, daemon=True).start()

        def audio_cb(indata, frames, time_info, status):
            if status:
                print(f"Status: {status}")
            audio_queue.put(indata.copy().flatten())

        CHUNK_DURATION = 1.5
        VERIFY_THRESH = 0.43

        try:
            with sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=int(CHUNK_DURATION * 16000), callback=audio_cb):
                while st.session_state.call_active:
                    chunk = audio_queue.get()
                    
                    if chunk is None: break
                    if np.max(np.abs(chunk)) < 0.005:
                        conf_metric.metric("Match Confidence", "0.0%")
                        caller_metric.metric("Line Status", "Silence")
                        speaker_metric.metric("Active Speaker", "None")
                        call_box.markdown(
                            "<div class='live-box pulsing-box'><span class='scanning-text'>Call connected. Waiting for voice...</span></div>", 
                            unsafe_allow_html=True
                        )
                        continue

                    # Process audio: Noise Reduction
                    clean_chunk = nr.reduce_noise(y=chunk, sr=16000, prop_decrease=0.8)

                    call_box.markdown(
                        "<div class='live-box pulsing-box' style='border-left: 4px solid #3273f6;'><span class='scanning-text'>Separating Voices & Authenticating...</span></div>", 
                        unsafe_allow_html=True
                    )

                    chunk_tensor = torch.from_numpy(clean_chunk).float().unsqueeze(0).to(DEVICE)
                    est_sources = separator.separate_batch(chunk_tensor)

                    src1 = est_sources[:, :, 0].detach().cpu().numpy().flatten()
                    src2 = est_sources[:, :, 1].detach().cpu().numpy().flatten()

                    sim1 = 1 - cosine(vp.flatten(), audio_to_embedding(normalize_audio(src1), classifier).flatten())
                    sim2 = 1 - cosine(vp.flatten(), audio_to_embedding(normalize_audio(src2), classifier).flatten())

                    if sim1 > sim2: 
                        owner_track, best_sim = src1, sim1
                        unknown_track = src2
                    else: 
                        owner_track, best_sim = src2, sim2
                        unknown_track = src1

                    pct = best_sim * 100
                    conf_metric.metric("Match Confidence", f"{pct:.1f}%")

                    if best_sim >= VERIFY_THRESH:
                        caller_metric.metric("Line Status", "🟢 VERIFIED")
                        speaker_metric.metric("Active Speaker", username.upper())
                        call_box.markdown(
                            f"<div class='live-box' style='background: rgba(76, 175, 80, 0.1); border: 1px solid #4caf50;'><b style='color:#4caf50; font-size: 1.2rem;'>✅ AUTHORIZED VOICE DETECTED.</b> Transmitting over network.</div>", 
                            unsafe_allow_html=True
                        )
                        network_node.send_queue.put(owner_track)
                    else:
                        caller_metric.metric("Line Status", "🔴 UNKNOWN")
                        speaker_metric.metric("Active Speaker", "Unknown")
                        call_box.markdown(
                            f"<div class='live-box' style='background: rgba(244, 67, 54, 0.1); border: 1px solid #f44336;'><b style='color:#f44336; font-size: 1.2rem;'>🚨 THREAT DETECTED: Unknown Voice.</b> Blocked from transmission.</div>", 
                            unsafe_allow_html=True
                        )

            playback_queue.put(None)
            network_node.stop()
            if "network_node" in st.session_state:
                del st.session_state.network_node
        except Exception as e:
            st.error(f"⚠️ Call Audio Failure: {e}")
            st.session_state.call_active = False
            if "network_node" in st.session_state:
                st.session_state.network_node.stop()
                del st.session_state.network_node

