import os
from kaggle.api.kaggle_api_extended import KaggleApi

# 1️⃣ Kaggle API 토큰 경로 설정
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser("C:/Users/djs66/.kaggle")

# 2️⃣ Kaggle API 초기화
api = KaggleApi()
api.authenticate()

# 3️⃣ 데이터셋 다운로드
dataset = "jedidiahangekouakou/grid-corpus-dataset-for-training-lipnet"
save_dir = "C:/Users/djs66/Desktop/grid_dataset"

os.makedirs(save_dir, exist_ok=True)
api.dataset_download_files(dataset, path=save_dir, unzip=True)

print(f"✅ GRID 데이터셋이 {save_dir} 에 성공적으로 다운로드되었습니다!")
