import torch
from models.lightning import AVSRLightningModel
from utils.io import load_audio_video  # (직접 만든 헬퍼 or load 함수)
import argparse
import os

# 1 인자 설정
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help="Path to pre-trained checkpoint")
parser.add_argument("--video_path", type=str, required=True, help="Path to video file (.mpg or .mp4)")
parser.add_argument("--audio_path", type=str, required=True, help="Path to audio file (.wav)")
parser.add_argument("--output_txt", type=str, default="inference_result.txt", help="Where to save the transcription")
args = parser.parse_args()

# 2️ 모델 로드
print(f"Loading model checkpoint from {args.checkpoint}...")
model = AVSRLightningModel.load_from_checkpoint(args.checkpoint, strict=False)
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

# 3️ 입력 데이터 로드
print("Preparing input...")
video_tensor, audio_tensor = load_audio_video(args.video_path, args.audio_path)

# 4️ 추론 (inference)
print("Running inference...")
with torch.no_grad():
    predicted_text = model.infer(video_tensor, audio_tensor)

print(f"Prediction: {predicted_text}")

# 5️ 파일로 저장
with open(args.output_txt, "w") as f:
    f.write(predicted_text + "\n")

print(f"✅ Inference result saved to {args.output_txt}")
