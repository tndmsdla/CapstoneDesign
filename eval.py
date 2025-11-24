#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 19:10:25 2024

@author: umbertocappellazzo
"""

import logging
import torch
from argparse import ArgumentParser

from datamodule.data_module import DataModule_LLM, collate_LLM
from datamodule.av_dataset import load_video, load_audio, cut_or_pad
from datamodule.transforms import AudioTransform, VideoTransform
from pytorch_lightning import Trainer
from models.lightning import ModelModule_LLM
from pytorch_lightning.loggers import WandbLogger
from python_speech_features import logfbank
import torch.nn.functional as F
import torchaudio
import numpy as np

def get_trainer(args):
    return Trainer(precision='bf16-true',
                   num_nodes=1,
                   devices=1,
                   accelerator="gpu",
                   logger=WandbLogger(name=args.exp_name, project=args.project_wandb)
                   )
 
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--exp-name",
        default= None,
        type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--project-wandb",
        default=None ,
        type=str,
        help="Name of the wandb project.",
    )
    parser.add_argument(
        "--modality",
        default=None,
        type=str,
        help="Type of input modality",
        choices=["audio", "video", "audiovisual"],
    )
    parser.add_argument(
        "--pretrained-model-path",                      
        default= None,
        type=str,
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--root-dir",
        default=None,
        type=str,
        help="Root directory of preprocessed dataset",
    )
    parser.add_argument(
        "--test-file",
        default="lrs3_test_transcript_lengths_seg24s_LLM_lowercase.csv",
        type=str,
        help="Filename of testing label list.",
    )
    parser.add_argument(
        "--pretrain-avhubert-enc-video-path",
        default= None,
        type=str,                                                               
    )
    parser.add_argument(
        "--pretrain-avhubert-enc-audio-path",
        default= None,
        type=str,
    )
    parser.add_argument(
        "--pretrain-avhubert-enc-audiovisual-path",
        default= None,
        type=str,
    )
    parser.add_argument(
        "--use-lora-avhubert",
        default = False,
        type = bool,
        help= "Whether to apply LoRA to the transformer module of AV-HuBERT."
        )
    parser.add_argument(
        "--single-projector-avhubert",
        default= False,
        type=bool,
        help="""This parameter is used only when modality == audiovisual_avhubert. If set to True, a single audio-visual projector
                is trained on top of the audio-visual features output by AV-HuBERT. If set to False, audio and video features
                are computed twice with AV-HuBERT with the other modality set to None""",
    )
    parser.add_argument(
        "--llm-model",
        default= None,
        type=str,
        help="LLM model name",
        choices= ["TinyLlama/TinyLlama_v1.1", "meta-llama/Llama-2-13b-hf", 
                  "meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3.1-8B"
                 ]
    )
    parser.add_argument(
        "--audio-encoder-name",
        default = None,
        type = str
        )
    parser.add_argument(
        "--intermediate-size",
        default= 2048,
        type=int,
        help="Intermediate size of the projector.",
    )
    parser.add_argument(
        "--prompt-audio",
        default= "Transcribe speech to text.",
        type=str,
        help="The prompt for the LLM.",
    )
    parser.add_argument(
        "--prompt-video",
        default= "Transcribe video to text.",
        type=str,
        help="The prompt for the LLM.",
    )
    parser.add_argument(
        "--prompt-audiovisual",
        default= "Transcribe speech and video to text.",
        type=str,
        help="The prompt for the LLM.",
    )
    parser.add_argument(
        "--unfrozen_modules",
        nargs="*",
        default= [None], #  "peft_llm","lora_avhubert"
        help="Which modules to train.",
        choices = [None, "peft_llm", "lora_avhubert"]
    )
    parser.add_argument(
        "--add_PETF_LLM",
        default= None,
        type= str,
        help="Whether to add a PEFT module to the LLM.",
        choices= [None, "lora"]
    )
    parser.add_argument(
        "--reduction_lora",
        default= None,
        type=int,
        help="Rank for LoRA."
    )
    parser.add_argument(
        "--alpha",
        default= None,
        type=int,
        help="Alpha for LoRA."
    )
    parser.add_argument(
        "--downsample-ratio-audio",
        default=3,
        type=int,
        help="Downsample ratio.",
    )
    parser.add_argument(
        "--downsample-ratio-video",
        default=3,
        type=int,
        help="Downsample ratio.",
    )
    parser.add_argument(
        "--downsample-ratio-audiovisual",
        default=3,
        type=int,
        help="Downsample ratio.",
    )
    parser.add_argument(
        "--max-dec-tokens",
        default= 32,
        type=int,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--num-beams",
        default= 15,
        type=int,
        help="Beams used for beam search",
    )
    parser.add_argument(
        "--train-num-buckets",
        type=int,
        default=400,
        help="Bucket size for the training set",
    )
    parser.add_argument(
        "--decode-snr-target",
        type=float,
        default= 999999,  
        help="Level of signal-to-noise ratio (SNR)",
        choices= [999999,5,2,0,-2,-5]
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag to use debug level for logging",
    )
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Inference on a single file instead of dataset",
    )
    parser.add_argument(
        "--video-path",
        default=None,
        type=str,
        help="Path to video file for single file inference (.mpg or .mp4)",
    )
    parser.add_argument(
        "--audio-path",
        default=None,
        type=str,
        help="Path to audio file for single file inference (.wav)",
    )
    
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def stacker(feats, stack_order):
    """
    Concatenating consecutive audio frames
    Args:
    feats - numpy.ndarray of shape [T, F]
    stack_order - int (number of neighboring frames to concatenate
    Returns:
    feats - numpy.ndarray of shape [T', F']
    """
    feat_dim = feats.shape[1]
    if len(feats) % stack_order != 0:
        res = stack_order - len(feats) % stack_order
        res = np.zeros([res, feat_dim]).astype(feats.dtype)
        feats = np.concatenate([feats, res], axis=0)
    feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
    return feats


def single_file_inference(args, modelmodule):
    """단일 파일에 대한 inference 수행"""
    logging.info("단일 파일 inference 모드로 실행합니다...")
    
    # Transform 설정
    is_avhubert_audio = True if args.modality == 'audiovisual_avhubert' or (args.audio_encoder_name == 'av-hubert' and args.modality =='audio') else False
    video_transform = VideoTransform("test") if args.modality in ["video", "audiovisual", "audiovisual_avhubert"] else None
    audio_transform = AudioTransform("test", snr_target=args.decode_snr_target, is_avhubert_audio=is_avhubert_audio) if args.modality in ["audio", "audiovisual", "audiovisual_avhubert"] else None
    
    # 데이터 로드 및 전처리
    batch_data = {}
    rate_ratio = 640
    
    if args.modality == "video":
        if not args.video_path:
            raise ValueError("--video-path가 필요합니다.")
        video = load_video(args.video_path)
        video = video_transform(video)
        if args.downsample_ratio_video:
            video = video[: video.size(0) // args.downsample_ratio_video * args.downsample_ratio_video]
        batch_data["video"] = video
        batch_data["tokens"] = ""  # inference에서는 텍스트가 없음
        
    elif args.modality == "audio":
        if not args.audio_path:
            raise ValueError("--audio-path가 필요합니다.")
        # 단일 파일 inference에서는 직접 오디오 로드
        waveform, sample_rate = torchaudio.load(args.audio_path, normalize=True)
        audio = waveform.transpose(1, 0)
        audio = audio_transform(audio)
        
        if is_avhubert_audio:
            device = audio.device
            audio = logfbank(audio)
            audio = torch.tensor(stacker(audio, 4), dtype=torch.float32, device=device)
            with torch.no_grad():
                audio = F.layer_norm(audio, audio.shape[1:])
        
        batch_data["audio"] = audio
        batch_data["tokens"] = ""
        
    elif args.modality == "audiovisual":
        if not args.video_path or not args.audio_path:
            raise ValueError("--video-path와 --audio-path가 모두 필요합니다.")
        video = load_video(args.video_path)
        # 단일 파일 inference에서는 직접 오디오 로드
        waveform, sample_rate = torchaudio.load(args.audio_path, normalize=True)
        audio = waveform.transpose(1, 0)
        audio = cut_or_pad(audio, len(video) * rate_ratio)
        
        video = video_transform(video)
        audio = audio_transform(audio)
        
        if args.downsample_ratio_video:
            video = video[: video.size(0) // args.downsample_ratio_video * args.downsample_ratio_video]
        
        batch_data["video"] = video
        batch_data["audio"] = audio
        batch_data["tokens"] = ""
        
    elif args.modality == "audiovisual_avhubert":
        if not args.video_path or not args.audio_path:
            raise ValueError("--video-path와 --audio-path가 모두 필요합니다.")
        video = load_video(args.video_path)
        # 단일 파일 inference에서는 직접 오디오 로드
        waveform, sample_rate = torchaudio.load(args.audio_path, normalize=True)
        audio = waveform.transpose(1, 0)
        audio = cut_or_pad(audio, len(video) * rate_ratio)
        
        video = video_transform(video)
        
        if not args.single_projector_avhubert:
            video = video[: video.size(0) // args.downsample_ratio_video * args.downsample_ratio_video]
        
        device = audio.device
        audio = logfbank(audio)
        audio = torch.tensor(stacker(audio, 4), dtype=torch.float32, device=device)
        with torch.no_grad():
            audio = F.layer_norm(audio, audio.shape[1:])
        
        batch_data["video"] = video
        batch_data["audio"] = audio
        batch_data["tokens"] = ""
    
    # Batch 형태로 변환
    batch = collate_LLM(batch_data, modelmodule.tokenizer, args.modality, is_trainval=False)
    
    # GPU로 이동
    device = next(modelmodule.model.parameters()).device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    # Inference 수행
    modelmodule.eval()
    with torch.no_grad():
        generated_ids = modelmodule.model(batch, is_trainval=False)
        generated_text = modelmodule.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    logging.info(f"생성된 텍스트: {generated_text}")
    print(f"\n{'='*50}")
    print(f"생성된 텍스트: {generated_text}")
    print(f"{'='*50}\n")
    
    return generated_text


def cli_main():
    args = parse_args()
    init_logger(args.debug)
    
    modelmodule = ModelModule_LLM(args)
    
    # 단일 파일 inference 모드
    if args.single_file:
        single_file_inference(args, modelmodule)
    else:
        # 기존 데이터셋 테스트 모드
        datamodule = DataModule_LLM(args, modelmodule.tokenizer, train_num_buckets=args.train_num_buckets)
        trainer = get_trainer(args)
        trainer.test(model=modelmodule, datamodule=datamodule)
    
    


if __name__ == "__main__":
    cli_main()
