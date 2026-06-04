# YALLM Datasets Guide

YALLM utilizes tailored pipelines for different types of modalities and training processes to ensure maximum throughput and minimum memory usage.

## 1. Text Pretraining Datasets

We support an advanced sharded tokenization pipeline that memory-maps massive sets of data. 

### Download & Pre-tokenize
This pulls from HuggingFace, tokenizes the data locally, and saves it as pure binary arrays (`.npy`):

```bash
# General downloader usage (displays SFT, Vision, and Pretraining datasets)
python download_hf.py --list

# Download TinyStories for quick testing (pre-tokenized into shards)
python download_hf.py --dataset tiny-stories

# Download FineWeb-Edu 10B token sample
python download_hf.py --dataset fineweb-edu-10bt
```

## 2. Text Fine-tuning Data (SFT)

For instruction-following (SFT) tasks, we rely on standard JSON lines.

```bash
python download_hf.py --list

# Download standard JSON SFT datasets
python download_hf.py --dataset alpaca
python download_hf.py --dataset open-orca
```

## 3. Vision & Multimodal Datasets

Training our image generation capabilities involves two sequential datasets.

### A. Downloading the raw images
You can download standard Hugging Face datasets directly using our unified script:

```bash
# Downloads Coco, Tiny-Imagenet, or custom image datasets like diffusiondb-pixelart
python download_hf.py --dataset diffusiondb-pixelart --out-dir ./datasets/
```

### B. Preprocessing the images
Because datasets come in various raw formats (Zips, scripts, Parquet metadata), we require a unified preprocessing step. This outputs a heavily standardized flat directory of precisely cropped images with their matching `.txt` captions.

```bash
python preprocess_image_dataset.py \
    --input-dir ./datasets/diffusiondb-pixelart/ \
    --output-dir ./datasets/pixelart_processed/ \
    --image-size 256 \
    --aspect-ratio-tol 0.2
```

This single structured output (`pixelart_processed/`) works seamlessly for both **Phase 1 (VQ-VAE)** and **Phase 2 (Multimodal)** generation tasks.
