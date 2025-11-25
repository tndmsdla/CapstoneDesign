import os
import argparse
import torch
import jiwer
from tqdm import tqdm
import pandas as pd
import re
import contextlib

# 추론 엔진 가져오기
from inference_avsr import load_model_from_checkpoint, inference_single_file

def parse_eval_args():
    parser = argparse.ArgumentParser(description="GRID Dataset Evaluation Script")
    
    # === 필수 경로 ===
    parser.add_argument("--data_dir", type=str, required=True, help="데이터셋 루트 폴더 (예: data/)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pretrain_avhubert_enc_video_path", type=str, required=True)
    parser.add_argument("--llm_model", type=str, default="models/Meta-Llama-3.1-8B")
    
    # === 타겟 스피커 ===
    parser.add_argument("--speaker", type=str, default="s1", help="평가할 스피커 폴더명 (예: s10_processed)")

    # === 실험 옵션 ===
    parser.add_argument("--use_uadf", action="store_true", help="UADF 사용 여부")
    parser.add_argument("--output_csv", type=str, default="eval_result.csv", help="결과 저장 파일명")
    
    # === 기본값 설정 ===
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
        return ""

def num_to_word(text):
    mapping = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text

def clean_text(text):
    text = text.lower()
    text = num_to_word(text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    args = parse_eval_args()
    
    print("🚀 모델 로딩 중...")
    model = load_model_from_checkpoint(args.checkpoint, args)
    print("✅ 모델 준비 완료!")

    # ================= [경로 설정 수정됨] =================
    target_speaker_folder = args.speaker # 예: s10_processed
    
    # 1. 스피커 폴더 경로 (예: data/s10_processed)
    video_dir = os.path.join(args.data_dir, target_speaker_folder)
    
    # 2. 정답 파일 폴더 경로 (예: data/s10_processed/align)
    align_dir = os.path.join(video_dir, "align")

    if not os.path.exists(video_dir):
        print(f"❌ 스피커 폴더를 찾을 수 없습니다: {video_dir}")
        return
    if not os.path.exists(align_dir):
        print(f"❌ 정답(align) 폴더를 찾을 수 없습니다: {align_dir}")
        return
    # ======================================================

    # 비디오 파일 목록 수집
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mpg') or f.endswith('.mp4')]
    video_files.sort()
    
    results = []
    total_wer = 0
    count = 0

    print(f"[INFO] Processing {len(video_files)} files in '{target_speaker_folder}'...")

    for vid_file in tqdm(video_files, desc="Evaluating", unit="file"):
        file_id = os.path.splitext(vid_file)[0]
        
        # 경로 조합
        video_path = os.path.join(video_dir, vid_file)
        
        # 정답 파일 찾기 (align 폴더 안에서 찾음)
        align_path = os.path.join(align_dir, file_id + ".align")
        
        # 정답 파일 없으면 스킵
        if not os.path.exists(align_path): continue
        
        ground_truth = get_ground_truth(align_path)
        if not ground_truth: continue

        # [핵심] 비디오 경로를 오디오 경로로도 사용 (inference_avsr가 알아서 처리함)
        args.video_path = video_path
        args.audio_path = video_path 
        
        try:
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                prediction = inference_single_file(args, model).lower().strip()
            
            ground_truth_clean = clean_text(ground_truth)
            prediction_clean = clean_text(prediction)
            
            wer = jiwer.wer(ground_truth_clean, prediction_clean)
            results.append({"file": file_id, "gt": ground_truth_clean, "pred": prediction_clean, "wer": wer})
            
            total_wer += wer
            count += 1
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n[WARN] Failed on {file_id}: {e}")

    if count > 0:
        avg_wer = total_wer / count
        print("\n" + "="*40)
        print(f"[SUMMARY] Speaker: {target_speaker_folder} | UADF: {args.use_uadf}")
        print(f"[SUMMARY] Count: {count}")
        print(f"[SUMMARY] Avg WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
        print("="*40 + "\n")
        
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
        print(f"[INFO] Report saved to {args.output_csv}")
    else:
        print("[WARN] No valid data processed.")

if __name__ == "__main__":
    main()