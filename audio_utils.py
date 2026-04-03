import torch
import numpy as np
import sounddevice as sd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_NODE = "NVIDIA RTX GPU" if DEVICE == "cuda" else "Local CPU Cluster"

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio

def audio_to_embedding(audio_np: np.ndarray, classifier) -> np.ndarray:
    tensor = torch.from_numpy(audio_np).float().unsqueeze(0).to(DEVICE)
    return classifier.encode_batch(tensor).squeeze().cpu().numpy()

def get_vbcable_device_id():
    try:
        for i, dev in enumerate(sd.query_devices()):
            if "CABLE Input" in dev['name'] and dev['max_output_channels'] > 0: 
                return i
    except Exception: 
        pass
    return None
