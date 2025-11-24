#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UADF (Unified Audio-Decoder Fusion) Block for Llama-AVSR
[Updated] BFloat16 Safety & Debugging Added
"""

import torch
from torch import nn
import torch.nn.functional as F


class UADFBlock(nn.Module):
    """
    UADF Block: Dynamic weighted fusion of audio and video features
    """
    
    def __init__(self, hidden_size, fusion_method='uncertainty', temperature=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.fusion_method = fusion_method
        self.temperature = temperature
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(hidden_size)
        
        if fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                batch_first=True
            )

    def compute_uncertainty(self, features):
        """
        Compute uncertainty (entropy) of features
        [Safety Fix] Performs calculation in Float32 to avoid NaN in BFloat16
        """
        # 1. 입력 타입을 기억해둠
        original_dtype = features.dtype
        
        # 2. 정밀한 계산(Entropy)을 위해 Float32로 변환 (BFloat16 NaN 방지)
        features = features.to(dtype=torch.float32)

        # Normalize features and compute probability distribution
        features_norm = F.normalize(features, p=2, dim=-1)      

        probs = F.softmax(features_norm / self.temperature, dim=-1)
        
        eps = 1e-10
        log_probs = torch.log(probs + eps)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # 3. 원래 타입(BFloat16 등)으로 복구
        return entropy.to(dtype=original_dtype)
    
    def align_sequences(self, audio_features, video_features):
        """
        Align audio and video sequences to the same length
        [Safety Fix] Interpolate handles BFloat16 poorly, casting to Float32 temporarily
        """
        batch_size = audio_features.shape[0]
        audio_len = audio_features.shape[1]
        video_len = video_features.shape[1]
        
        if audio_len == video_len:
            return audio_features, video_features

        # Interpolate를 위해 잠시 Float32로 변환
        original_dtype = audio_features.dtype
        audio_features = audio_features.to(dtype=torch.float32)
        video_features = video_features.to(dtype=torch.float32)

        if audio_len > video_len:
            # (Batch, Time, Dim) -> (Batch, Dim, Time) for interpolate
            audio_features = F.interpolate(
                audio_features.transpose(1, 2),
                size=video_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            video_features = F.interpolate(
                video_features.transpose(1, 2),
                size=audio_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        # 원래 타입(BFloat16)으로 복구
        return audio_features.to(dtype=original_dtype), video_features.to(dtype=original_dtype)
    
    def forward(self, audio_features, video_features):
        """
        Forward pass: Fuse audio and video features dynamically
        """
        # 입력 타입 강제 일치 (Safety Check)
        if audio_features.dtype != video_features.dtype:
            target_dtype = video_features.dtype
            audio_features = audio_features.to(dtype=target_dtype)

        # 1. 시퀀스 길이 맞추기 (Interpolate)
        audio_features, video_features = self.align_sequences(
            audio_features, video_features
        )
        
        if self.fusion_method == 'uncertainty':
            # 2. 불확실성(Uncertainty) 계산 (Float32 내부 처리)
            video_uncertainty = self.compute_uncertainty(video_features)
            
            # 3. 가중치 계산
            weight = self.sigmoid(video_uncertainty) - 0.5
            weight = weight.unsqueeze(-1) 
            
            # 4. 퓨전 (Fusion)
            # fused = Video + Weight * Audio
            fused_features = video_features + weight * audio_features
        
            # 5. 정규화 (LayerNorm)
            # Residual Connection 포함
            fused_features = self.norm(fused_features + video_features)
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused_features


def create_uadf_block(hidden_size, fusion_method='uncertainty', temperature=1.0):
    return UADFBlock(
        hidden_size=hidden_size,
        fusion_method=fusion_method,
        temperature=temperature
    )