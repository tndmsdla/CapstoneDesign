import os
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torchaudio
import torchvision


def cut_or_pad(data: torch.Tensor, size: int, dim: int = 0) -> torch.Tensor:
    """
    Pads or trims `data` along `dim` to match `size`.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        pad_dims: Tuple[int, ...] = (0, 0, 0, padding)
        data = torch.nn.functional.pad(data, pad_dims, "constant")
    elif data.size(dim) > size:
        data = data.narrow(dim, 0, size)
    return data


def load_video(path: str) -> torch.Tensor:
    """
    Returns video tensor with shape (T, C, H, W).
    """
    video, _, _ = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")
    return video.permute(0, 3, 1, 2)


def load_audio(path: str) -> Tuple[torch.Tensor, int]:
    """
    Returns audio tensor with shape (T, 1) and its sample rate.
    """
    waveform, sample_rate = torchaudio.load(path, normalize=True)
    return waveform.transpose(1, 0), sample_rate


class GRIDDataset(torch.utils.data.Dataset):
    """
    GRID dataset loader.

    Expected folder structure:
        root_dir/
            ├── s1/                     (video .mpg)
            ├── audio_25k/s1/           (audio .wav)
            └── alignments/s1/          (alignment .align)

    By default only speaker s1 is used. Additional speakers can be passed via `speakers`.
    """

    def __init__(
        self,
        root_dir: str,
        modality: str = "audiovisual",
        split: Optional[str] = None,
        split_ratio: Sequence[float] = (0.8, 0.1, 0.1),
        speakers: Optional[Iterable[str]] = None,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        video_transform: Optional[torch.nn.Module] = None,
        audio_transform: Optional[torch.nn.Module] = None,
        rate_ratio: int = 640,
    ) -> None:
        super().__init__()
        self.root_dir = os.path.abspath(root_dir)
        self.modality = modality
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.rate_ratio = rate_ratio

        self.speakers = list(speakers) if speakers else ["s1"]
        self.split = split
        self.split_ratio = split_ratio
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.seed = seed

        if split is not None:
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Unsupported split: {split}")
            if len(split_ratio) != 3 or abs(sum(split_ratio) - 1.0) > 1e-6:
                raise ValueError("split_ratio must contain three values summing to 1.0")

        self._video_dirs = {
            speaker: os.path.join(self.root_dir, speaker) for speaker in self.speakers
        }
        self._audio_dirs = {
            speaker: os.path.join(self.root_dir, "audio_25k", speaker)
            for speaker in self.speakers
        }
        self._align_dirs = {
            speaker: os.path.join(self.root_dir, "alignments", speaker)
            for speaker in self.speakers
        }

        self.samples: List[Tuple[str, str]] = self._collect_samples()

        if not self.samples:
            raise RuntimeError(
                f"No GRID samples found under {self.root_dir} for speakers {self.speakers}"
            )

    def _collect_samples(self) -> List[Tuple[str, str]]:
        all_samples: List[Tuple[str, str]] = []
        for speaker in self.speakers:
            video_dir = self._video_dirs[speaker]
            if not os.path.isdir(video_dir):
                continue

            ids = []
            for filename in os.listdir(video_dir):
                base, ext = os.path.splitext(filename)
                if ext.lower() in {".mpg", ".mp4"}:
                    ids.append(base)

            ids.sort()
            for sample_id in ids:
                all_samples.append((speaker, sample_id))

        if self.split is None:
            subset = all_samples
        else:
            n_total = len(all_samples)
            n_train = int(n_total * self.split_ratio[0])
            n_val = int(n_total * self.split_ratio[1])

            if self.split == "train":
                subset = all_samples[:n_train] or all_samples
            elif self.split == "val":
                start = n_train
                end = start + n_val
                subset = all_samples[start:end] or all_samples[start:start + 1]
            else:  # test split
                start = n_train + n_val
                subset = all_samples[start:] or all_samples[-1:]

        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(subset)

        if self.max_samples is not None:
            subset = subset[: self.max_samples]

        return subset

    def _alignment_path(self, speaker: str, sample_id: str) -> str:
        return os.path.join(self._align_dirs[speaker], f"{sample_id}.align")

    def _audio_path(self, speaker: str, sample_id: str) -> str:
        return os.path.join(self._audio_dirs[speaker], f"{sample_id}.wav")

    def _video_path(self, speaker: str, sample_id: str) -> str:
        # Prefer .mpg, fall back to .mp4 if needed.
        mpg_path = os.path.join(self._video_dirs[speaker], f"{sample_id}.mpg")
        if os.path.isfile(mpg_path):
            return mpg_path
        return os.path.join(self._video_dirs[speaker], f"{sample_id}.mp4")

    @staticmethod
    def _load_alignment(path: str) -> str:
        words: List[str] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) == 3:
                    _, _, word = parts
                    if word != "sil":
                        words.append(word)
        return " ".join(words)

    def __getitem__(self, index: int) -> dict:
        speaker, sample_id = self.samples[index]

        need_video = self.modality in {"video", "audiovisual", "audiovisual_avhubert"}
        need_audio = self.modality in {"audio", "audiovisual", "audiovisual_avhubert"}

        video = None
        if need_video:
            video = load_video(self._video_path(speaker, sample_id))
            if self.video_transform is not None:
                video = self.video_transform(video)

        audio = None
        if need_audio:
            audio, sample_rate = load_audio(self._audio_path(speaker, sample_id))
            if sample_rate != 25000:
                audio = torchaudio.functional.resample(audio.t(), sample_rate, 25000).t()

            if video is not None:
                expected_audio_frames = video.size(0) * self.rate_ratio
                audio = cut_or_pad(audio, expected_audio_frames, dim=0)

            if self.audio_transform is not None:
                audio = self.audio_transform(audio)

        tokens = self._load_alignment(self._alignment_path(speaker, sample_id))

        sample = {"tokens": tokens}
        if video is not None:
            sample["video"] = video
        if audio is not None:
            sample["audio"] = audio
        return sample

    def __len__(self) -> int:
        return len(self.samples)

