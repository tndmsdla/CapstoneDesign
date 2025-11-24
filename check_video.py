import torch
import argparse
import sys
import os

# 필요한 모듈 임포트 (inference_avsr.py와 같은 환경이어야 함)
from datamodule.av_dataset import load_video
from datamodule.transforms import VideoTransform
from datamodule.data_module import collate_LLM

# === 테스트할 파일 경로 (본인 경로에 맞게 수정) ===
VIDEO_PATH = "tests/swwv9a.mpg" 
# ==============================================

class MockTokenizer:
    """collate_LLM을 속이기 위한 가짜 토크나이저"""
    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2]} 

def debug_video_logic():
    print(f"🚀 [Start] Debugging video processing logic for: {VIDEO_PATH}")
    
    # 1. 비디오 로드
    try:
        video = load_video(VIDEO_PATH)
        print(f"1. Load Shape: {video.shape} (Time, H, W)")
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return

    # 2. Transform 적용
    video_transform = VideoTransform("test")
    video = video_transform(video)
    print(f"2. After Transform: {video.shape}")

    # 3. [수정된 로직] 채널 차원 추가
    # inference_avsr.py에 추가한 로직과 동일해야 함
    if len(video.shape) == 3:
        print("   -> Adding Channel dim...")
        video = video.unsqueeze(1)
    print(f"3. After Channel Add: {video.shape} (Time, 1, H, W)")

    # 4. 다운샘플링 (비디오만 해당)
    downsample_ratio = 2
    video = video[: video.size(0) // downsample_ratio * downsample_ratio]
    print(f"4. After Downsample: {video.shape}")

    # =========================================================
    # 🕵️ collate_LLM 시뮬레이션 (배치 생성)
    # =========================================================
    
    batch_data = {"video": video, "tokens": ""}
    batch_list = [batch_data]
    
    # 가짜 토크나이저 사용
    tokenizer = MockTokenizer()
    
    print("\n📦 Running collate_LLM...")
    try:
        batch = collate_LLM(batch_list, tokenizer, modality="video", is_trainval=False)
        video_tensor = batch["video"]
        print(f"5. Batch Shape (Raw): {video_tensor.shape}")
    except Exception as e:
        print(f"❌ Collate Error: {e}")
        # collate가 실패하면 수동으로 stack해서 시뮬레이션
        video_tensor = torch.stack([video])
        print(f"5. Batch Shape (Simulated): {video_tensor.shape}")

    # =========================================================
    # 🚨 최종 검증 (차원 교정 로직 테스트)
    # =========================================================
    
    print("\n🛠️ Testing Fix Logic...")
    
    final_shape = video_tensor.shape
    
    # 6차원이면 교정 필요
    if video_tensor.dim() == 6 and video_tensor.shape[1] == 1:
        print(f"⚠️ [ISSUE] 6차원 데이터 감지! ({final_shape})")
        print("   -> squeeze(1) 적용 중...")
        
        # 교정 수행
        video_tensor = video_tensor.squeeze(1)
        print(f"✅ [FIXED] Final Shape: {video_tensor.shape}")
        
        if video_tensor.dim() == 5:
            print("🎉 성공! 이제 모델에 들어갈 수 있습니다.")
        else:
            print("❌ 여전히 이상합니다.")
            
    elif video_tensor.dim() == 5:
        print(f"✅ [PASS] 이미 완벽한 5차원입니다. ({final_shape})")
    else:
        print(f"❓ [UNKNOWN] 예상 밖의 모양입니다: {final_shape}")

if __name__ == "__main__":
    debug_video_logic()