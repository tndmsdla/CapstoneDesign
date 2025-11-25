import torch
import torchaudio
import sys
import os

# 필요한 모듈 임포트 (inference_avsr.py와 같은 환경)
from datamodule.transforms import AudioTransform
from datamodule.data_module import collate_LLM

# === 테스트할 파일 경로 ===
AUDIO_PATH = "tests/swwv9a.wav" 
# =======================

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2]} 

def debug_audio_logic():
    print(f"[INFO] Starting Audio Debugging: {AUDIO_PATH}")
    
    # 1. Load
    try:
        waveform, sample_rate = torchaudio.load(AUDIO_PATH, normalize=True)
        print(f"[DEBUG] Loaded Shape: {waveform.shape} | Rate: {sample_rate}Hz")
    except Exception as e:
        print(f"[ERROR] Failed to load audio: {e}")
        return

    # 2. Resample (16k)
    if sample_rate != 16000:
        print(f"[DEBUG] Resampling {sample_rate}Hz -> 16000Hz...")
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    # 3. Mono Convert
    if waveform.shape[0] > 1:
        print(f"[DEBUG] Stereo detected ({waveform.shape[0]}ch). Converting to Mono...")
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 4. Transpose (Channels, Time) -> (Time, Channels)
    audio = waveform.transpose(1, 0)
    print(f"[DEBUG] Transposed Shape: {audio.shape}")

    # 5. Transform
    audio_transform = AudioTransform("test", snr_target=999999) 
    audio = audio_transform(audio)
    print(f"[DEBUG] After Transform: {audio.shape}")

    # 6. Collate Simulation
    batch_data = {"audio": audio, "tokens": ""}
    batch_list = [batch_data]
    tokenizer = MockTokenizer()
    
    print("[INFO] Running collate_LLM...")
    try:
        batch = collate_LLM(batch_list, tokenizer, modality="audio", is_trainval=False)
        audio_tensor = batch["audio"]
        print(f"[DEBUG] Batch Shape (Raw): {audio_tensor.shape}")
    except Exception as e:
        print(f"[WARN] Collate failed ({e}). Simulating stack...")
        audio_tensor = torch.stack([audio]).unsqueeze(0) 
        print(f"[DEBUG] Batch Shape (Simulated): {audio_tensor.shape}")

    # 7. Dimension Fix Check
    print("[INFO] Checking Dimensions...")
    final_shape = audio_tensor.shape
    
    # 4차원이면 교정 (예: [1, 1, Time, 1]) -> [1, Time, 1]
    if audio_tensor.dim() == 4 and audio_tensor.shape[1] == 1:
        print(f"[WARN] 4D Tensor detected {final_shape}. Applying squeeze(1)...")
        audio_tensor = audio_tensor.squeeze(1)
        print(f"[INFO] Corrected Shape: {audio_tensor.shape}")
        
        if audio_tensor.dim() == 3:
            print("[SUCCESS] Final shape is valid (3D).")
        else:
            print("[ERROR] Shape is still invalid.")
            
    elif audio_tensor.dim() == 3:
        print(f"[SUCCESS] Shape is already valid (3D): {final_shape}")
    else:
        print(f"[ERROR] Unexpected shape: {final_shape}")

if __name__ == "__main__":
    debug_audio_logic()