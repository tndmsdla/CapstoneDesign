#!/usr/bin/env python3
import argparse
import os
from typing import Tuple

import torch
import torch.nn.functional as F

from datamodule.grid_dataset import GRIDDataset
from models.uadf_block import create_uadf_block


def downsample_audio_to_video(audio: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    audio: (audio_seq_len, audio_dim)
    return: (target_len, audio_dim)
    """
    if audio.size(0) == target_len:
        return audio

    audio = audio.transpose(0, 1).unsqueeze(0)  # (1, audio_dim, audio_seq_len)
    audio = F.interpolate(audio, size=target_len, mode="linear", align_corners=False)
    return audio.squeeze(0).transpose(0, 1)


def build_feature_projectors(video_channels: int, audio_dim: int, hidden_size: int, device: torch.device):
    video_proj = torch.nn.Linear(video_channels, hidden_size, bias=True).to(device)
    audio_proj = torch.nn.Linear(audio_dim, hidden_size, bias=True).to(device)
    video_proj.eval()
    audio_proj.eval()
    return video_proj, audio_proj


def prepare_features(
    sample: dict,
    video_proj: torch.nn.Linear,
    audio_proj: torch.nn.Linear,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    video = sample["video"].float() / 255.0  # (T, C, H, W)
    audio = sample["audio"].float()  # (audio_seq_len, audio_dim)

    frame_features = video.mean(dim=(2, 3))  # (T, C)
    frame_features = video_proj(frame_features.to(device))

    audio_downsampled = downsample_audio_to_video(audio, frame_features.size(0))
    audio_features = audio_proj(audio_downsampled.to(device))

    return audio_features.unsqueeze(0), frame_features.unsqueeze(0)


def run_test(args):
    dataset = GRIDDataset(
        root_dir=args.grid_root,
        modality="audiovisual",
        max_samples=args.num_samples,
        shuffle=False,
    )

    sample0 = dataset[0]
    video_channels = sample0["video"].size(1)
    audio_dim = sample0["audio"].size(1)

    device = torch.device(args.device)
    video_proj, audio_proj = build_feature_projectors(video_channels, audio_dim, args.hidden_size, device)
    uadf = create_uadf_block(hidden_size=args.hidden_size, fusion_method="uncertainty", temperature=args.temperature).to(device)
    uadf.eval()

    print(f"[설정] hidden_size={args.hidden_size}, temperature={args.temperature}, device={device}")
    print(f"[데이터] GRID 위치={args.grid_root}, 샘플 수={min(len(dataset), args.num_samples)}")

    with torch.no_grad():
        for idx in range(min(len(dataset), args.num_samples)):
            sample = dataset[idx]
            audio_feats, video_feats = prepare_features(sample, video_proj, audio_proj, device)
            fused = uadf(audio_feats, video_feats)

            uncertainty = uadf.compute_uncertainty(video_feats)
            weights = uadf.sigmoid(uncertainty) - 0.5

            delta = (fused - video_feats).abs().mean().item()
            audio_contrib = (weights.unsqueeze(-1) * audio_feats).abs().mean().item()

            print(f"[샘플 {idx}] 길이={video_feats.size(1)} | Δmean={delta:.6f} | audio_contrib={audio_contrib:.6f} | weight범위=({weights.min().item():.4f}, {weights.max().item():.4f})")


def parse_args():
    parser = argparse.ArgumentParser(description="UADF 블록을 GRID 일부 샘플로 검증하는 스크립트")
    default_root = os.environ.get("GRID_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GRID"))
    parser.add_argument("--grid-root", type=str, default=default_root, help="GRID 데이터셋 루트 경로")
    parser.add_argument("--num-samples", type=int, default=4, help="테스트할 샘플 수")
    parser.add_argument("--hidden-size", type=int, default=256, help="UADF 숨김 차원")
    parser.add_argument("--temperature", type=float, default=1.0, help="불확실성 온도 스케일")
    parser.add_argument("--device", type=str, default="cpu", help="실행 디바이스")
    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(42)
    args = parse_args()
    run_test(args)

