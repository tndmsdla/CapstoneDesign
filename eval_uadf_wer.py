import os
import argparse
import torch
import jiwer
from tqdm import tqdm
import pandas as pd
import re

# 추론 엔진 가져오기
from inference_avsr import load_model_from_checkpoint, inference_single_file

def parse_eval_args():
    parser = argparse.ArgumentParser(description="GRID Dataset Evaluation Script")
    
    # === 필수 경로 ===
    parser.add_argument("--data_dir", type=str, required=True, help="데이터셋 폴더 (tests/ 또는 전체 데이터셋 루트)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pretrain_avhubert_enc_video_path", type=str, required=True)
    parser.add_argument("--llm_model", type=str, default="models/Meta-Llama-3.1-8B")
    
    # === 타겟 스피커 (전체 데이터셋일 때만 사용됨) ===
    parser.add_argument("--speaker", type=str, default="s1", help="평가할 스피커 폴더명 (예: s1)")

    # === 실험 옵션 ===
    parser.add_argument("--use_uadf", action="store_true", help="UADF 사용 여부")
    parser.add_argument("--output_csv", type=str, default="eval_result.csv", help="결과 저장 파일명")
    
    # === 고정/기본값 ===
    parser.add_argument("--modality", type=str, default="audiovisual")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--audio_path", type=str, default=None)
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
    try:
        words = []
        with open(align_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    word = parts[2]
                    if word not in ["sil", "sp"]:
                        words.append(word)
        return " ".join(words).lower()
    except Exception as e:
        # print(f"⚠️ 정답 파일 읽기 실패 ({align_path}): {e}")
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

def main():
    args = parse_eval_args()
    
    # 1. 모델 로드
    # print("🚀 모델 로딩 중...")
    model = load_model_from_checkpoint(args.checkpoint, args)
    # print("✅ 모델 준비 완료!")

    # 2. 경로 자동 감지 로직 (핵심 수정!)
    target_speaker = args.speaker
    
    # (A) 전체 데이터셋 구조인지 확인 (s1 폴더가 있는지)
    full_structure_path = os.path.join(args.data_dir, target_speaker)
    
    if os.path.exists(full_structure_path) and os.path.isdir(full_structure_path):
        print(f"📂 전체 데이터셋 구조 감지 (Target: {target_speaker})")
        video_dir = full_structure_path
        audio_dir = os.path.join(args.data_dir, "audio", target_speaker)
        align_dir = os.path.join(args.data_dir, "alignments", target_speaker)
    else:
        # (B) 테스트 폴더 구조 (파일들이 data_dir에 바로 있음)
        # print(f"📂 단일(Flat) 폴더 구조 감지 (Target: {args.data_dir})")
        video_dir = args.data_dir
        audio_dir = args.data_dir
        align_dir = args.data_dir

    # 경로 유효성 최종 확인
    if not os.path.exists(video_dir):
        print(f"❌ 폴더를 찾을 수 없습니다: {video_dir}")
        return

    # 3. 평가 루프
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mpg') or f.endswith('.mp4')]
    video_files.sort()
    
    results = []
    total_wer = 0
    count = 0

    print(f"▶️ 총 {len(video_files)}개 파일 평가 시작 (UADF: {args.use_uadf})")

    for vid_file in tqdm(video_files):
        file_id = os.path.splitext(vid_file)[0]
        
        video_path = os.path.join(video_dir, vid_file)
        audio_path = os.path.join(audio_dir, file_id + ".wav")
        align_path = os.path.join(align_dir, file_id + ".align")
        
        if not os.path.exists(audio_path): continue
        if not os.path.exists(align_path): continue
            
        ground_truth = get_ground_truth(align_path)
        if not ground_truth: continue

        args.video_path = video_path
        args.audio_path = audio_path 
        
        try:
            prediction = inference_single_file(args, model).lower().strip()
            ground_truth_clean = clean_text(ground_truth)
            prediction_clean = clean_text(prediction)
            
            wer = jiwer.wer(ground_truth_clean, prediction_clean)
            
            results.append({
                "file": file_id,
                "ground_truth": ground_truth_clean,
                "prediction": prediction_clean,
                "wer": wer
            })
            
            total_wer += wer
            count += 1
            
        except Exception as e:
            print(f"❌ Error: {e}")

    if count > 0:
        avg_wer = total_wer / count
        print(f"\n{'='*40}")
        print(f"📊 최종 평가 결과 (UADF: {args.use_uadf})")
        print(f"   - 파일 수: {count}")
        print(f"   - 평균 WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
        print(f"{'='*40}")
        
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
        print(f"💾 결과 저장됨: {args.output_csv}")
    else:
        print("⚠️ 평가할 데이터가 없습니다.")

if __name__ == "__main__":
    main()