import torch
import sys
import os

from datamodule.av_dataset import load_video
from datamodule.transforms import VideoTransform
from datamodule.data_module import collate_LLM

# === 테스트할 파일 경로 ===
VIDEO_PATH = "tests/swwv9a.mpg" 
# =======================

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2]} 

def debug_video_logic():
    print(f"[INFO] Starting Video Debugging: {VIDEO_PATH}")
    
    # 1. Load
    try:
        video = load_video(VIDEO_PATH)
        print(f"[DEBUG] Loaded Shape: {video.shape} (Time, H, W)")
    except Exception as e:
        print(f"[ERROR] Failed to load video: {e}")
        return

    # 2. Transform
    video_transform = VideoTransform("test")
    video = video_transform(video)
    print(f"[DEBUG] After Transform: {video.shape}")

    # 3. Add Channel Dim
    if len(video.shape) == 3:
        print("[DEBUG] Adding Channel dim (unsqueeze)...")
        video = video.unsqueeze(1)
    print(f"[DEBUG] After Channel Add: {video.shape} (Time, 1, H, W)")

    # 4. Downsample
    downsample_ratio = 2
    video = video[: video.size(0) // downsample_ratio * downsample_ratio]
    
    # 5. Collate Simulation
    batch_data = {"video": video, "tokens": ""}
    batch_list = [batch_data]
    tokenizer = MockTokenizer()
    
    print("[INFO] Running collate_LLM...")
    try:
        batch = collate_LLM(batch_list, tokenizer, modality="video", is_trainval=False)
        video_tensor = batch["video"]
        print(f"[DEBUG] Batch Shape (Raw): {video_tensor.shape}")
    except Exception as e:
        print(f"[WARN] Collate failed ({e}). Simulating stack...")
        video_tensor = torch.stack([video])
        print(f"[DEBUG] Batch Shape (Simulated): {video_tensor.shape}")

    # 6. Dimension Fix Check
    print("[INFO] Checking Dimensions...")
    final_shape = video_tensor.shape
    
    # 6차원이면 교정 (예: [1, 1, T, C, H, W]) -> [1, T, C, H, W]
    if video_tensor.dim() == 6 and video_tensor.shape[1] == 1:
        print(f"[WARN] 6D Tensor detected {final_shape}. Applying squeeze(1)...")
        video_tensor = video_tensor.squeeze(1)
        print(f"[INFO] Corrected Shape: {video_tensor.shape}")
        
        if video_tensor.dim() == 5:
            print("[SUCCESS] Final shape is valid (5D).")
        else:
            print("[ERROR] Shape is still invalid.")
            
    elif video_tensor.dim() == 5:
        print(f"[SUCCESS] Shape is already valid (5D): {final_shape}")
    else:
        print(f"[ERROR] Unexpected shape: {final_shape}")

if __name__ == "__main__":
    debug_video_logic()