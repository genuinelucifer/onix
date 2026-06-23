"""
Data loading utilities for all Onix training modes.

Provides factory functions that return (train_loader, val_loader, data_info)
tuples for each training pipeline:

  - create_pretrain_dataloaders:      LLM pretraining (text file or .npy shards)
  - create_image_dataloaders:         VQ-VAE training (image directory)
  - create_multimodal_dataloaders:    Multi-modal LLM training (image+text pairs)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ===========================================================================
#  LLM Pretraining Datasets
# ===========================================================================

class ShardedTokenDataset(Dataset):
    """
    Dataset backed by pre-tokenized .npy shard files (uint16).

    Each shard is a flat array of token IDs produced by download_hf.py.
    Samples are non-overlapping chunks strided by seq_len. Each chunk is
    (seq_len + 1) tokens — the extra token provides the shifted target.
    """

    def __init__(self, shard_paths: list[Path], seq_len: int):
        from bisect import bisect_right
        self.seq_len = seq_len
        self.shards = []
        self.shard_offsets = []  # cumulative sample counts per shard

        total_samples = 0
        for path in sorted(shard_paths):
            data = np.load(path, mmap_mode="r")
            # Non-overlapping chunks: stride by seq_len, need seq_len+1 for target
            n_samples = max(0, (len(data) - 1) // seq_len)
            if n_samples > 0:
                self.shards.append(data)
                self.shard_offsets.append(total_samples)
                total_samples += n_samples

        self.total_samples = total_samples

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        from bisect import bisect_right
        # Binary search for the right shard
        shard_idx = bisect_right(self.shard_offsets, idx) - 1

        local_idx = idx - self.shard_offsets[shard_idx]
        data = self.shards[shard_idx]
        # Stride by seq_len for non-overlapping chunks
        start = local_idx * self.seq_len
        chunk = data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


class TextFileDataset(Dataset):
    """
    Simple dataset from a single text file, tokenized on-the-fly at init.

    Produces contiguous windows of (seq_len + 1) tokens — the extra token
    provides the shifted target.
    """

    def __init__(self, token_ids: list[int], seq_len: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def create_pretrain_dataloaders(
    data_source: str,
    seq_len: int,
    batch_size: int,
    tokenizer=None,
    max_shards: int | None = None,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Create train/val DataLoaders for LLM pretraining.

    Supports two modes:
      1. Sharded .npy directory (large datasets) — pass a directory path.
         The directory should contain shard_XXXXX.npy files (uint16 token IDs).
      2. Text file (small datasets) — pass a file path + tokenizer.

    Args:
        data_source:  Path to shard directory or text file.
        seq_len:      Context length (number of tokens per sample).
        batch_size:   Batch size.
        tokenizer:    tiktoken tokenizer (required for text file mode).
        max_shards:   Limit number of shards (for testing).
        val_ratio:    Fraction of data/shards reserved for validation.
        num_workers:  DataLoader workers.
        pin_memory:   Pin memory for GPU transfer.
        prefetch_factor: DataLoader prefetch factor.

    Returns:
        (train_loader, val_loader, data_info_dict)
    """
    source_path = Path(data_source)
    loader_kwargs = dict(
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    if source_path.is_dir():
        # ---- Sharded mode ----
        shard_files = sorted(source_path.glob("shard_*.npy"))
        if not shard_files:
            raise FileNotFoundError(
                f"No shard_*.npy files found in {source_path}"
            )
        if max_shards is not None:
            shard_files = shard_files[:max_shards]

        # Split shards into train / val
        n_val = max(1, int(len(shard_files) * val_ratio))
        val_shards = shard_files[-n_val:]
        train_shards = shard_files[:-n_val]

        if not train_shards:
            # If only 1 shard, use it for both
            train_shards = shard_files
            val_shards = shard_files

        train_ds = ShardedTokenDataset(train_shards, seq_len)
        val_ds = ShardedTokenDataset(val_shards, seq_len)

        # Try to load metadata for extra info
        meta = {}
        meta_path = source_path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        data_info = {
            "mode": "sharded",
            "total_shards": len(shard_files),
            "train_shards": len(train_shards),
            "val_shards": len(val_shards),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "seq_len": seq_len,
            "total_tokens": meta.get("total_tokens", "unknown"),
        }

        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    else:
        # ---- Text file mode ----
        if tokenizer is None:
            raise ValueError(
                "tokenizer is required when data_source is a text file"
            )

        text = source_path.read_text(encoding="utf-8")
        token_ids = tokenizer.encode(text, allowed_special="all")

        # Split into train/val
        split_idx = int(len(token_ids) * (1 - val_ratio))
        train_ids = token_ids[:split_idx]
        val_ids = token_ids[split_idx:]

        train_ds = TextFileDataset(train_ids, seq_len)
        val_ds = TextFileDataset(val_ids, seq_len)

        data_info = {
            "mode": "text_file",
            "file": str(source_path),
            "total_tokens": len(token_ids),
            "train_tokens": len(train_ids),
            "val_tokens": len(val_ids),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "seq_len": seq_len,
        }

        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, data_info


# ===========================================================================
#  VQ-VAE Image Dataset
# ===========================================================================

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


class ImageFolderDataset(Dataset):
    """
    Simple dataset that loads images from a flat directory.

    Images are resized to (image_size, image_size) and normalized to [-1, 1].
    """

    def __init__(
        self,
        image_paths: list[Path],
        image_size: int = 256,
        image_channels: int = 3,
    ):
        self.image_paths = image_paths
        self.image_size = image_size
        self.image_channels = image_channels
        self._transforms = None

    @property
    def transforms(self):
        """Lazy-init transforms (torchvision may not be installed)."""
        if self._transforms is None:
            from torchvision import transforms
            mode = "RGB" if self.image_channels == 3 else "RGBA"
            self._transforms = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),           # [0, 1]
                transforms.Normalize(            # [-1, 1]
                    mean=[0.5] * self.image_channels,
                    std=[0.5] * self.image_channels,
                ),
            ])
        return self._transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.image_paths[idx])
        mode = "RGB" if self.image_channels == 3 else "RGBA"
        img = img.convert(mode)
        return self.transforms(img)


def create_image_dataloaders(
    data_dir: str,
    image_size: int = 256,
    image_channels: int = 3,
    batch_size: int = 16,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Create train/val DataLoaders for VQ-VAE image training.

    Loads images from a flat directory (e.g. output of preprocess_image_dataset.py).

    Args:
        data_dir:       Path to directory containing image files.
        image_size:     Target square resolution.
        image_channels: Number of channels (3=RGB, 4=RGBA).
        batch_size:     Batch size.
        val_ratio:      Fraction reserved for validation.
        num_workers:    DataLoader workers.
        pin_memory:     Pin memory for GPU transfer.
        prefetch_factor: DataLoader prefetch factor.

    Returns:
        (train_loader, val_loader, data_info_dict)
    """
    data_path = Path(data_dir)
    image_paths = sorted([
        p for p in data_path.rglob("*")
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ])

    if not image_paths:
        raise FileNotFoundError(
            f"No image files found in {data_path}. "
            f"Supported formats: {SUPPORTED_IMAGE_EXTENSIONS}"
        )

    # Split into train / val
    n_val = max(1, int(len(image_paths) * val_ratio))
    val_paths = image_paths[-n_val:]
    train_paths = image_paths[:-n_val]

    if not train_paths:
        train_paths = image_paths
        val_paths = image_paths

    train_ds = ImageFolderDataset(train_paths, image_size, image_channels)
    val_ds = ImageFolderDataset(val_paths, image_size, image_channels)

    loader_kwargs = dict(
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    data_info = {
        "total_images": len(image_paths),
        "train_images": len(train_paths),
        "val_images": len(val_paths),
        "image_size": image_size,
        "image_channels": image_channels,
    }

    return train_loader, val_loader, data_info


# ===========================================================================
#  Multi-Modal Dataset (Text + Image tokens)
# ===========================================================================

class MultiModalDataset(Dataset):
    """
    Dataset for multi-modal autoregressive training.

    Each sample is a sequence:
        [text_tokens... | <IMG_START> | visual_tokens... | <IMG_END>]

    With a loss mask that is 1 only on visual token positions (and optionally
    <IMG_END>), so the model learns to predict image tokens given text.

    Supports two modes:
      1. Live encoding: loads .png + .txt pairs, encodes images via frozen VQ-VAE.
      2. Pre-encoded: loads .npy (pre-encoded visual token indices) + .txt pairs.
    """

    def __init__(
        self,
        data_dir: Path,
        tokenizer,
        vqvae_model,
        mm_config,
        device: torch.device = torch.device("cpu"),
        pre_encoded: bool = False,
    ):
        from PIL import Image
        from torchvision import transforms

        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.vqvae = vqvae_model
        self.mm_config = mm_config
        self.device = device
        self.pre_encoded = pre_encoded

        # Collect samples (matched .png + .txt pairs, or .npy + .txt)
        if pre_encoded:
            self.sample_stems = sorted([
                p.stem for p in self.data_dir.glob("*.npy")
            ])
        else:
            img_exts = {".png", ".jpg", ".jpeg", ".webp"}
            self.sample_stems = sorted([
                p.stem for p in self.data_dir.iterdir()
                if p.suffix.lower() in img_exts
                and (self.data_dir / (p.stem + ".txt")).exists()
            ])

        # Image transforms for live encoding
        if not pre_encoded and vqvae_model is not None:
            vqvae_cfg = mm_config.vqvae
            self._img_transform = transforms.Compose([
                transforms.Resize((vqvae_cfg.image_size, vqvae_cfg.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * vqvae_cfg.image_channels,
                                     [0.5] * vqvae_cfg.image_channels),
            ])
        else:
            self._img_transform = None

    def __len__(self):
        return len(self.sample_stems)

    def __getitem__(self, idx):
        stem = self.sample_stems[idx]
        mm = self.mm_config

        # --- Load text ---
        txt_path = self.data_dir / f"{stem}.txt"
        caption = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
        text_tokens = self.tokenizer.encode(caption, allowed_special="all")
        # Truncate / pad to max_text_tokens
        text_tokens = text_tokens[:mm.max_text_tokens]
        # Pad with EOT (50256) if shorter
        while len(text_tokens) < mm.max_text_tokens:
            text_tokens.append(50256)

        # --- Load visual tokens ---
        if self.pre_encoded:
            npy_path = self.data_dir / f"{stem}.npy"
            visual_tokens = np.load(npy_path).flatten().tolist()
        else:
            from PIL import Image
            # Find the image file
            img_path = None
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                candidate = self.data_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            img = Image.open(img_path).convert("RGB")
            img_tensor = self._img_transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                visual_tokens = self.vqvae.encode(img_tensor).squeeze(0).cpu().tolist()

        # Offset visual tokens into the joint vocab space
        visual_tokens = [t + mm.text_vocab_size for t in visual_tokens]

        # --- Build sequence ---
        # [text_tokens | <IMG_START> | visual_tokens | <IMG_END>]
        seq = text_tokens + [mm.img_start_id] + visual_tokens + [mm.img_end_id]

        # Input and target (shifted by 1)
        inp = torch.tensor(seq[:-1], dtype=torch.long)
        tgt = torch.tensor(seq[1:], dtype=torch.long)

        # Loss mask: 1 for visual positions (after <IMG_START> in the target)
        loss_mask = torch.zeros_like(tgt, dtype=torch.float)
        # The visual tokens in the target start at index max_text_tokens (after IMG_START)
        vis_start = mm.max_text_tokens  # IMG_START is at this position in inp, first visual token is here in tgt
        vis_end = vis_start + mm.num_visual_tokens + 1  # +1 for IMG_END
        loss_mask[vis_start:vis_end] = 1.0

        return inp, tgt, loss_mask


def create_multimodal_dataloaders(
    data_dir: str,
    tokenizer,
    vqvae_model,
    mm_config,
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
    pre_encoded: bool = False,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Create train/val DataLoaders for multi-modal autoregressive training.

    The data directory should contain paired files:
      - .png/.jpg + .txt  (live encoding via frozen VQ-VAE)
      - .npy + .txt       (pre-encoded visual tokens)

    Args:
        data_dir:       Path to image+text pairs directory.
        tokenizer:      tiktoken tokenizer for text encoding.
        vqvae_model:    Frozen VQ-VAE model (None if pre-encoded).
        mm_config:      MultiModalConfig instance.
        batch_size:     Batch size.
        device:         Device for VQ-VAE encoding.
        pre_encoded:    If True, load .npy files instead of encoding images.
        val_ratio:      Fraction reserved for validation.
        num_workers:    DataLoader workers.
        pin_memory:     Pin memory for GPU transfer.
        prefetch_factor: DataLoader prefetch factor.

    Returns:
        (train_loader, val_loader, data_info_dict)
    """
    full_ds = MultiModalDataset(
        data_dir, tokenizer, vqvae_model, mm_config,
        device=device, pre_encoded=pre_encoded,
    )

    if len(full_ds) == 0:
        raise FileNotFoundError(
            f"No paired samples found in {data_dir}. "
            f"Expected .png/.jpg + .txt pairs (or .npy + .txt if pre-encoded)."
        )

    # Split into train / val
    n_val = max(1, int(len(full_ds) * val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # For live encoding with VQ-VAE, use num_workers=0 to avoid CUDA in forked processes
    effective_workers = 0 if (not pre_encoded and vqvae_model is not None) else num_workers

    loader_kwargs = dict(
        batch_size=batch_size,
        drop_last=True,
        num_workers=effective_workers,
        pin_memory=pin_memory,
    )
    if effective_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    data_info = {
        "total_samples": len(full_ds),
        "train_samples": n_train,
        "val_samples": n_val,
        "pre_encoded": pre_encoded,
        "seq_length": mm_config.max_seq_length,
    }

    return train_loader, val_loader, data_info
