import warnings
warnings.filterwarnings("ignore")

import os
import sys

import matplotlib.pyplot as plt
import torchaudio

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

sys.path.append(CURRENT_DIR)

from datamodule.grid_dataset import GRIDDataset

torchaudio.set_audio_backend("soundfile")

ROOT_DIR = os.environ.get("GRID_DATA_ROOT", os.path.join(REPO_ROOT, "GRID"))

dataset = GRIDDataset(root_dir=ROOT_DIR, modality="audiovisual")

print("총 샘플 개수:", len(dataset))
sample = dataset[0]
print("정답 문장:", sample["tokens"])

# 오디오 저장 테스트
torchaudio.save("test_audio.wav", sample["audio"].T, 25000)
print("✅ test_audio.wav 저장 완료")

# 첫 프레임 표시
frame = sample["video"][0].permute(1, 2, 0).numpy()
plt.imshow(frame)
plt.title("First Frame")
plt.axis("off")
plt.show()
