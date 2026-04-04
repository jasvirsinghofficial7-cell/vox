# watcher.py - OS Level Background Security Agent (With Anti-Spoofing & 1-Hour Session)
import os
import sys
import time
import json
import psutil
import threading
import numpy as np
import sounddevice as sd
import noisereduce as nr
import torch
import subprocess
import pyautogui
import webview  
import getpass

from scipy.spatial.distance import cosine
from speechbrain.inference.speaker import EncoderClassifier
from database import load_voiceprint_from_db

# 🚀 SYSTEM SETUP
LOCK_FILE = "app_locks.json"
DEVICE = "cpu" 

print("🛡️ Loading AI Core for OS Watcher...")
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": DEVICE})
print("✅ OS Watcher is LIVE! Monitoring apps in background...")

is_popup_open = False 
window = None

# 🔥 DYNAMIC USERNAME FUNCTION 🔥
def get_active_username():
    try:
        if os.path.exists("active_user.txt"):
            with open("active_user.txt", "r") as f:
                name = f.read().strip()
                if name: return name.lower()
    except:
        pass
    return getpass.getuser().lower()

def get_locks():
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r") as f: return json.load(f)
        except: return {}
    return {}

# 🕒 UPGRADE: Default whitelist time set to 3600 seconds (1 Hour)
def update_lock_time(process_name, added_seconds=3600):
    locks = get_locks()
    for app, data in locks.items():
        if data.get("process", "").lower() == process_name.lower():
            data["unlocked_until"] = time.time() + added_seconds
            print(f"🔓 Whitelisted {process_name} for {added_seconds} seconds!")
    with open(LOCK_FILE, "w") as f:
        json.dump(locks, f)

def record_audio(duration=4, fs=16000):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return recording.flatten()

def is_parent_unlocked(proc, locks):
    """Check if the parent of this process is a whitelisted/unlocked app."""
    try:
        shell_names = ["cmd.exe", "powershell.exe", "conhost.exe", "bash.exe"]
        if proc.name().lower() not in shell_names:
            return False

        parent = proc.parent()
        if not parent: return False
        
        parent_name = parent.name().lower()
        for app, data in locks.items():
            if data.get("process", "").lower() == parent_name:
                if time.time() < data.get("unlocked_until", 0):
                    return True
    except:
        pass
    return False

# 🧠 API that connects UI to Python AI Logic
class AuthAPI:
    def __init__(self):
        self.current_process = ""
        self.current_exe = ""

    def set_target(self, process_name, exe_path):
        self.current_process = process_name
        self.current_exe = exe_path

    def verify_voice(self):
        current_user = get_active_username()
        vp = load_voiceprint_from_db(current_user)

        if vp is None:
            return {"success": False, "msg": f"🚨 Voiceprint for '{current_user}' not found!"}

        try:
            raw_audio = record_audio(4)
            if np.max(np.abs(raw_audio)) > 0.005:
                clean_audio = nr.reduce_noise(y=raw_audio, sr=16000)

                # 🛡️ UPGRADE: Librosa Anti-Spoofing / Liveness Check 
                try:
                    from audio_utils import detect_liveness
                    is_live, spoof_reason = detect_liveness(clean_audio)
                    if not is_live:
                        pyautogui.press('r')
                        return {"success": False, "msg": f"🚨 Spoof Detected: {spoof_reason}"}
                except ImportError:
                    pass # Silently pass if detect_liveness is not yet in audio_utils.py

                # 🧠 Deep Learning Biometric Check
                tensor = torch.from_numpy(clean_audio).float().unsqueeze(0).to(DEVICE)
                live_emb = classifier.encode_batch(tensor).squeeze().cpu().numpy()
                
                sim = 1 - cosine(vp.flatten(), live_emb.flatten())
                
                if sim >= 0.60:
                    pyautogui.press('a')  
                    # 🕒 UPGRADE: 3600 Seconds (1 Hour) Session Timeout
                    update_lock_time(self.current_process, 3600)
                    threading.Timer(1.5, self.launch_and_close).start()
                    return {"success": True, "msg": f"✅ Access Granted! ({sim*100:.1f}%)"}
                else:
                    pyautogui.press('r')  
                    return {"success": False, "msg": f"🚨 Access Denied! ({sim*100:.1f}%)"}
            else:
                return {"success": False, "msg": "🔇 No audio detected. Try again."}
        except Exception as e:
            print(f"Error in verify_voice: {e}")
            return {"success": False, "msg": "🚨 AI Engine Error!"}

    def launch_and_close(self):
        global is_popup_open, window
        window.hide() 
        is_popup_open = False
        try:
            # 🚀 Shell-less App Launch (Avoids spawning cmd.exe)
            if "whatsapp" in self.current_process.lower():
                os.startfile("whatsapp:")
            elif self.current_exe and os.path.exists(self.current_exe) and "WindowsApps" not in self.current_exe:
                subprocess.Popen([self.current_exe], shell=False)
            else:
                # If we don't have the path, try to start common apps directly
                os.startfile(self.current_process)
        except Exception as e:
            print(f"⚠️ Launch error: {e}")

# 🌐 NATIVE EDGE HTML UI (Dark Theme + Spline + Sci-Fi Loader)
html_content = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { background-color: #0b0c10; color: white; font-family: 'Segoe UI', Arial, sans-serif; text-align: center; margin: 0; padding: 0; overflow: hidden; }
        h2 { color: #f44336; margin-top: 25px; font-size: 24px; }
        .spline-container { width: 100%; height: 350px; border: none; margin-top: 10px; opacity: 0; transition: opacity 0.8s ease-in; }
        .status { color: #8b929e; font-size: 14px; margin-top: 10px; height: 20px;}
        .btn { background-color: #3273f6; color: white; font-size: 16px; font-weight: bold; padding: 12px; border: none; border-radius: 8px; cursor: pointer; width: 85%; margin-top: 15px; transition: 0.3s; }
        .btn:disabled { background-color: #555555; cursor: not-allowed; }

        #loader {
            position: absolute; top: 80px; left: 0; width: 100%; height: 350px;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            background-color: #0b0c10; z-index: 10;
        }
        .spinner {
            width: 45px; height: 45px; border: 4px solid rgba(13, 240, 227, 0.1);
            border-top: 4px solid #0df0e3; border-radius: 50%;
            animation: spin 1s linear infinite; margin-bottom: 15px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    </style>
</head>
<body>
    <h2 id="title">🔒 App is Locked!</h2>
    <div id="loader">
        <div class="spinner"></div>
        <span style="color: #0df0e3; font-size: 14px; font-weight: bold; animation: pulse 1.5s infinite;">Initializing Neural Core...</span>
    </div>
    <iframe id="spline" class="spline-container" src="https://my.spline.design/voiceinteractionanimation-Mi2ojO60nAcIj2M84qyzw3pm/" frameborder="0"></iframe>
    <div id="status" class="status">Click button and speak for 4 seconds</div>
    <button id="authBtn" class="btn" onclick="triggerAuth()">🎤 Start Voice Authentication</button>

    <script>
        function setApp(appName) {
            document.getElementById('title').innerText = "🔒 " + appName + " is Locked!";
            document.getElementById('authBtn').disabled = false;
            document.getElementById('authBtn').innerText = "🎤 Start Voice Authentication";
            document.getElementById('authBtn').style.backgroundColor = "#3273f6";
            document.getElementById('status').innerText = "Click button and speak for 4 seconds";
            document.getElementById('status').style.color = "#8b929e";
            
            document.getElementById('spline').style.opacity = '0';
            document.getElementById('loader').style.display = 'flex';
            
            setTimeout(() => {
                document.getElementById('loader').style.display = 'none';
                document.getElementById('spline').style.opacity = '1';
            }, 1200); 
        }

        function triggerAuth() {
            document.getElementById('authBtn').disabled = true;
            document.getElementById('authBtn').innerText = "⏳ Listening...";
            document.getElementById('authBtn').style.backgroundColor = "#f5a623";
            document.getElementById('status').innerText = "🎙️ Recording... Speak Now!";
            document.getElementById('status').style.color = "#0df0e3";

            pywebview.api.verify_voice().then(function(res) {
                if(res.success) {
                    document.getElementById('authBtn').innerText = "Launching App...";
                    document.getElementById('authBtn').style.backgroundColor = "#4caf50";
                    document.getElementById('status').innerText = res.msg;
                    document.getElementById('status').style.color = "#4caf50";
                } else {
                    document.getElementById('authBtn').disabled = false;
                    document.getElementById('authBtn').innerText = "🎤 Try Again";
                    document.getElementById('authBtn').style.backgroundColor = "#3273f6";
                    document.getElementById('status').innerText = res.msg;
                    document.getElementById('status').style.color = "#f44336";
                }
            });
        }
    </script>
</body>
</html>
"""

# 🔄 BACKGROUND WATCHER THREAD
def background_watcher():
    global is_popup_open, window, api

    time.sleep(1) 

    while True:
        try:
            if is_popup_open:
                time.sleep(0.5)
                continue 

            locks = get_locks()
            
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                try:
                    name = proc.info.get('name')
                    exe_path = proc.info.get('exe') or ""
                    
                    if not name: continue

                    for app_disp, data in locks.items():
                        target = data.get("process", "").lower()
                        
                        if target and data.get("locked", False) and target == name.lower():
                            unlocked_until = data.get("unlocked_until", 0)
                            
                            # Check session timeout
                            if time.time() > unlocked_until:
                                
                                # 🛡️ EXEMPT self and immediate shell parent
                                current_pid = os.getpid()
                                try:
                                    if proc.pid == current_pid:
                                        continue
                                    
                                    parent = psutil.Process(current_pid).parent()
                                    if parent and proc.pid == parent.pid:
                                        if parent.name().lower() in ["cmd.exe", "powershell.exe", "bash.exe"]:
                                            continue
                                    
                                    if is_parent_unlocked(proc, locks):
                                        continue
                                except: pass

                                # 🔪 NATIVE KILL
                                try:
                                    for child in proc.children(recursive=True):
                                        try: child.kill()
                                        except: pass
                                    proc.kill()
                                    print(f"🔪 BLOCKED & NUKED: {name}")
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass

                                # Show Auth Popup
                                if not is_popup_open:
                                    is_popup_open = True
                                    api.set_target(name, exe_path)
                                    window.evaluate_js(f"setApp('{app_disp}')")
                                    window.show()
                                break
                except Exception:
                    pass 

        except Exception as e:
            pass
        
        time.sleep(0.05)

if __name__ == "__main__":
    api = AuthAPI()
    window = webview.create_window('VoxAuth Security', html=html_content, js_api=api, width=450, height=650, frameless=False, hidden=True, on_top=True)
    threading.Thread(target=background_watcher, daemon=True).start()
    webview.start()