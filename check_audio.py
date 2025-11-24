import torchaudio
import sys

# 확인하고 싶은 파일 경로 (사용 중인 파일명으로 수정하세요)
file_path = "tests/swwv9a.wav" 

try:
    waveform, sample_rate = torchaudio.load(file_path)
    print(f"📂 파일 경로: {file_path}")
    print(f"📊 텐서 모양(Shape): {waveform.shape}")
    print(f"🔊 채널 수: {waveform.shape[0]}")
    print(f"Hz 샘플 레이트: {sample_rate}")
    
    if waveform.shape[0] == 1:
        print("✅ 결과: 모노(Mono)입니다.")
    elif waveform.shape[0] == 2:
        print("⚠️ 결과: 스테레오(Stereo)입니다. (변환 필요)")
    else:
        print(f"⚠️ 결과: 다채널({waveform.shape[0]})입니다. (변환 필요)")

except Exception as e:
    print(f"❌ 파일을 여는 중 에러 발생: {e}")