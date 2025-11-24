import os
import argparse
import torch
import jiwer  # WER 계산 라이브러리
from tqdm import tqdm
import pandas as pd
import re

# [중요] 우리가 최적화한 추론 엔진 가져오기
from inference_avsr import load_model_from_checkpoint, inference_single_file

def parse_eval_args():
    parser = argparse.ArgumentParser(description="GRID Dataset Evaluation Script")
    
    # === 필수 경로 ===
    parser.add_argument("--data_dir", type=str, required=True, help="GRID 데이터셋(.mpg, .align)이 있는 폴더 경로")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pretrain_avhubert_enc_video_path", type=str, required=True)
    parser.add_argument("--llm_model", type=str, default="models/Meta-Llama-3.1-8B")
    
    # === 실험 옵션 ===
    parser.add_argument("--use_uadf", action="store_true", help="UADF 사용 여부 (비교 실험용)")
    parser.add_argument("--output_csv", type=str, default="eval_result.csv", help="결과 저장 파일명")
    
    # === 고정/기본값 (inference_avsr.py와 동일하게 유지) ===
    parser.add_argument("--modality", type=str, default="audiovisual")
    parser.add_argument("--video_path", type=str, default=None) # 루프 돌면서 바뀜
    parser.add_argument("--audio_path", type=str, default=None) # 루프 돌면서 바뀜
    parser.add_argument("--pretrain_avhubert_enc_audio_path", type=str, default=None)
    parser.add_argument("--pretrain_avhubert_enc_audiovisual_path", type=str, default=None)
    parser.add_argument("--audio_encoder_name", type=str, default="openai/whisper-medium.en")
    parser.add_argument("--downsample-ratio-video", type=int, default=2)
    parser.add_argument("--downsample-ratio-audio", type=int, default=4)
    parser.add_argument("--max-dec-tokens", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--use-lora-avhubert", action="store_true")
    parser.add_argument("--single-projector-avhubert", action="store_true")
    parser.add_argument("--grid-resample-audio", action="store_true")
    parser.add_argument("--uadf-fusion-method", type=str, default="uncertainty")
    parser.add_argument("--uadf-temperature", type=float, default=1.0)
    parser.add_argument("--prompt-audio", type=str, default="Transcribe speech to text.")
    parser.add_argument("--prompt-video", type=str, default="Transcribe video to text.")
    parser.add_argument("--prompt-audiovisual", type=str, default="Transcribe speech and video to text.")
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--unfrozen-modules", nargs="*", default=["peft_llm"]) 
    parser.add_argument("--add_PETF_LLM", type=str, default="lora")           
    parser.add_argument("--reduction-lora", type=int, default=64)             
    parser.add_argument("--alpha", type=int, default=8)                       
    parser.add_argument("--downsample-ratio-audiovisual", type=int, default=3)
    parser.add_argument("--pretrained-model-path", type=str, default=None)
    parser.add_argument("--use-half-precision", action="store_true")
    parser.add_argument("--low-cpu-mem-usage", action="store_true", default=True)
    parser.add_argument("--load-in-8bit", action="store_true", default=False) 
    parser.add_argument("--cpu-offload", action="store_true")
    
    return parser.parse_args()

def get_ground_truth(align_path):
    """ .align 파일 파싱하여 정답 문장 추출 """
    words = []
    try:
        with open(align_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                # 포맷: [시작시간] [끝시간] [단어]
                if len(parts) >= 3:
                    word = parts[2]
                    # sil(묵음), sp(짧은 정적) 제외
                    if word not in ["sil", "sp"]:
                        words.append(word)
        return " ".join(words).lower() # 소문자 통일
    except Exception as e:
        print(f"⚠️ 정답 파일 읽기 실패 ({align_path}): {e}")
        return ""

def clean_text(text):
    """ 특수문자 제거 및 소문자 변환 (WER 계산용) """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text) # 영문, 숫자, 공백만 남김
    return text.strip()

def main():
    args = parse_eval_args()
    
    # 1. 모델 로드 (한 번만 수행)
    print("🚀 모델 로딩 중...")
    # 우리가 만든 inference_avsr.py의 함수를 사용하므로 OOM 걱정 없음!
    model = load_model_from_checkpoint(args.checkpoint, args)
    print("✅ 모델 준비 완료!")

    # 2. 파일 목록 수집 (.mpg 파일 기준)
    video_files = [f for f in os.listdir(args.data_dir) if f.endswith('.mpg') or f.endswith('.mp4')]
    video_files.sort()
    
    results = []
    total_wer = 0
    count = 0

    print(f"📂 총 {len(video_files)}개 파일 평가 시작... (UADF 적용 여부: {args.use_uadf})")

    # 3. 평가 루프
    for vid_file in tqdm(video_files):
        video_path = os.path.join(args.data_dir, vid_file)
        # .mpg -> .align 확장자 변경
        align_path = os.path.splitext(video_path)[0] + ".align"
        
        # 정답 파일이 없으면 스킵
        if not os.path.exists(align_path):
            continue
            
        ground_truth = get_ground_truth(align_path)
        if not ground_truth: continue # 정답 내용이 없으면 스킵

        # 경로 설정 (오디오는 비디오 파일에서 추출)
        args.video_path = video_path
        args.audio_path = video_path 
        
        try:
            # 추론 실행
            prediction = inference_single_file(args, model)
            
            # 전처리 (소문자, 특수문자 제거)
            ground_truth_clean = clean_text(ground_truth)
            prediction_clean = clean_text(prediction)
            
            # WER 계산
            wer = jiwer.wer(ground_truth_clean, prediction_clean)
            
            results.append({
                "file": vid_file,
                "ground_truth": ground_truth_clean,
                "prediction": prediction_clean,
                "wer": wer
            })
            
            total_wer += wer
            count += 1
            
        except Exception as e:
            print(f"❌ Error processing {vid_file}: {e}")

    # 4. 결과 집계 및 저장
    if count > 0:
        avg_wer = total_wer / count
        print(f"\n{'='*40}")
        print(f"📊 최종 평가 결과 (UADF: {args.use_uadf})")
        print(f"   - 총 파일 수: {count}")
        print(f"   - 평균 WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
        print(f"{'='*40}")
        
        # CSV 저장
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
        print(f"💾 상세 결과 저장됨: {args.output_csv}")
    else:
        print("⚠️ 평가할 유효한 데이터가 없습니다.")

if __name__ == "__main__":
    main()