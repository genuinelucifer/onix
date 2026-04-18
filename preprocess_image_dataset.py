#!/usr/bin/env python3
"""
Preprocess a downloaded image dataset for VQ-VAE and multimodal training.

Takes a local input directory (downloaded via download_hf.py or any other
source), filters images by aspect ratio, resizes to square, and saves them
as individual image files + text caption files.

Supports two input formats:
  1. HuggingFace Arrow dataset (from download_hf.py's save_to_disk)
  2. Plain directory of image files (png/jpg/webp/...)

Output layout (compatible with both training pipelines):

    output_dir/
        00000.png      <- square image for VQ-VAE training (Phase 1)
        00000.txt      <- text caption for multimodal training (Phase 2)
        00001.png
        00001.txt
        ...

Usage:
    # From download_hf.py Arrow output (auto-detected)
    python preprocess_image_dataset.py \\
        --input-dir ./datasets/diffusiondb-pixelart \\
        --output-dir ./datasets/pixelart_processed

    # With custom size
    python preprocess_image_dataset.py \\
        --input-dir ./datasets/diffusiondb-pixelart \\
        --output-dir ./datasets/pixelart_128 \\
        --image-size 128

    # From a plain directory of images
    python preprocess_image_dataset.py \\
        --input-dir ./my_raw_images/ \\
        --output-dir ./datasets/processed

    # Keep all images regardless of aspect ratio
    python preprocess_image_dataset.py \\
        --input-dir ./datasets/diffusiondb-pixelart \\
        --output-dir ./datasets/pixelart_all \\
        --aspect-ratio-tol 999
"""

import argparse
import json
import sys
from pathlib import Path
import tarfile
import zipfile

from PIL import Image


# -- Defaults -----------------------------------------------------------------

DEFAULT_IMAGE_SIZE = 256
DEFAULT_ASPECT_TOL = 0.2   # |ratio - 1.0| <= tol -> "near square"
DEFAULT_IMAGE_FORMAT = "png"


# -- Filtering helpers --------------------------------------------------------

def is_near_square(width: int, height: int, tol: float) -> bool:
    """Return True if aspect ratio is within `tol` of 1.0."""
    if width == 0 or height == 0:
        return False
    ratio = max(width, height) / min(width, height)
    return (ratio - 1.0) <= tol


def resize_square(img: Image.Image, size: int) -> Image.Image:
    """Center-crop to square then resize to (size, size)."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)
    return img


# -- Input format detection ---------------------------------------------------

def is_arrow_dataset(input_dir: Path) -> bool:
    """Check if input_dir is a HuggingFace Arrow dataset (from save_to_disk)."""
    # save_to_disk creates dataset_info.json or state.json
    markers = ["dataset_info.json", "state.json"]
    for m in markers:
        if (input_dir / m).exists():
            return True
    # Also check for arrow files
    if list(input_dir.glob("*.arrow")):
        return True
    # DatasetDict: check subdirs (e.g. train/)
    for sub in input_dir.iterdir():
        if sub.is_dir():
            if (sub / "dataset_info.json").exists() or list(sub.glob("*.arrow")):
                return True
    return False


def detect_columns(dataset) -> tuple:
    """Auto-detect image and text/prompt column names from a HF dataset."""
    cols = dataset.column_names

    image_col = None
    for candidate in ["image", "img", "pixel_values", "input_image"]:
        if candidate in cols:
            image_col = candidate
            break

    text_col = None
    for candidate in ["prompt", "text", "caption", "description", "p", "label"]:
        if candidate in cols:
            text_col = candidate
            break

    return image_col, text_col


# -- Process Arrow dataset ----------------------------------------------------

def process_arrow_dataset(
    input_dir: Path,
    output_dir: Path,
    image_size: int,
    aspect_tol: float,
    image_format: str,
    image_col: str | None,
    text_col: str | None,
    min_size: int,
):
    """Process a HuggingFace Arrow dataset saved via save_to_disk."""
    try:
        from datasets import load_from_disk, concatenate_datasets
    except ImportError:
        print("Error: 'datasets' library required. Install: pip install datasets")
        sys.exit(1)

    print(f"Loading Arrow dataset from: {input_dir}")
    ds = load_from_disk(str(input_dir))

    # If DatasetDict, concatenate all splits
    from datasets import DatasetDict
    if isinstance(ds, DatasetDict):
        print(f"  Found splits: {list(ds.keys())}")
        all_splits = list(ds.values())
        if len(all_splits) == 1:
            ds = all_splits[0]
        else:
            ds = concatenate_datasets(all_splits)

    auto_img, auto_txt = detect_columns(ds)
    image_col = image_col or auto_img
    text_col = text_col or auto_txt

    if image_col is None:
        print(f"Error: Could not detect image column. Columns: {ds.column_names}")
        print("Specify --image-col explicitly.")
        sys.exit(1)

    print(f"Using columns: image='{image_col}', text='{text_col or '(none)'}'")
    print(f"Total samples: {len(ds)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped_aspect = 0
    skipped_size = 0
    skipped_error = 0

    for i, row in enumerate(ds):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(ds)} "
                  f"(kept={kept}, skipped_aspect={skipped_aspect}, "
                  f"skipped_size={skipped_size}, errors={skipped_error})")

        try:
            img = row[image_col]
            if img is None:
                skipped_error += 1
                continue

            if not isinstance(img, Image.Image):
                if isinstance(img, dict) and "bytes" in img:
                    import io
                    img = Image.open(io.BytesIO(img["bytes"]))
                else:
                    skipped_error += 1
                    continue

            w, h = img.size

            if min_size > 0 and (w < min_size or h < min_size):
                skipped_size += 1
                continue

            if not is_near_square(w, h, aspect_tol):
                skipped_aspect += 1
                continue

            img = img.convert("RGB")
            img = resize_square(img, image_size)

            fname = f"{kept:05d}"
            img.save(output_dir / f"{fname}.{image_format}")

            if text_col and text_col in row and row[text_col]:
                caption = str(row[text_col]).strip()
                if caption:
                    (output_dir / f"{fname}.txt").write_text(caption, encoding="utf-8")

            kept += 1

        except Exception as e:
            skipped_error += 1
            if skipped_error <= 5:
                print(f"  Warning: error on sample {i}: {e}")

    return kept, skipped_aspect, skipped_size, skipped_error


# -- Process plain image directory ---------------------------------------------

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def process_image_dir(
    input_dir: Path,
    output_dir: Path,
    image_size: int,
    aspect_tol: float,
    image_format: str,
    min_size: int,
    recursive: bool,
):
    """Process a plain directory of image files."""

    # Pre-extract any zip or tar.gz files found in the directory
    zip_paths = list(input_dir.rglob("*.zip"))
    if zip_paths:
        print(f"Found {len(zip_paths)} zip files, extracting...")
        for zp in zip_paths:
            with zipfile.ZipFile(zp, 'r') as zip_ref:
                zip_ref.extractall(zp.parent)

    tar_paths = list(input_dir.rglob("*.tar.gz"))
    if tar_paths:
        print(f"Found {len(tar_paths)} tar.gz files, extracting...")
        for tp in tar_paths:
            with tarfile.open(tp, 'r:gz') as tar_ref:
                tar_ref.extractall(tp.parent)

    if recursive:
        image_paths = sorted([
            p for p in input_dir.rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ])
    else:
        image_paths = sorted([
            p for p in input_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ])

    print(f"Found {len(image_paths)} images in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load metadata (check root and annotations/ folder)
    metadata_df = None
    meta_paths = [
        input_dir / "metadata.parquet",
        input_dir / "annotations" / "gz_decals_auto_posteriors.parquet"
    ]
    # Also look for any parquet in annotations/
    if (input_dir / "annotations").exists():
        meta_paths.extend(list((input_dir / "annotations").glob("*.parquet")))

    for meta_path in meta_paths:
        if meta_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(meta_path)
                print(f"Loaded metadata from {meta_path.name} with {len(df)} entries.")
                
                # Normalize index for lookup
                if "image_name" in df.columns:
                    df = df.set_index("image_name")
                elif "iauname" in df.columns:
                    # GZ Decals: file is often iauname.png
                    df = df.set_index("iauname")
                
                if metadata_df is None:
                    metadata_df = df
                else:
                    # Merge if multiple files (optional, depending on schema)
                    metadata_df = metadata_df.combine_first(df)
            except ImportError:
                print("Warning: pandas or pyarrow not installed, skipping metadata.")
                break
            except Exception as e:
                print(f"Warning: could not load {meta_path.name}: {e}")

    kept = 0
    skipped_aspect = 0
    skipped_size = 0
    skipped_error = 0

    for i, img_path in enumerate(image_paths):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(image_paths)} "
                  f"(kept={kept}, skipped_aspect={skipped_aspect}, "
                  f"skipped_size={skipped_size}, errors={skipped_error})")

        try:
            img = Image.open(img_path)
            w, h = img.size

            if min_size > 0 and (w < min_size or h < min_size):
                skipped_size += 1
                continue

            if not is_near_square(w, h, aspect_tol):
                skipped_aspect += 1
                continue

            img = img.convert("RGB")
            img = resize_square(img, image_size)

            fname = f"{kept:05d}"
            img.save(output_dir / f"{fname}.{image_format}")

            caption = ""
            # 1. Check for alongside .txt
            src_txt = img_path.with_suffix(".txt")
            if src_txt.exists():
                caption = src_txt.read_text(encoding="utf-8").strip()
            
            # 2. Check for metadata.parquet / GZ Decals
            if not caption and metadata_df is not None:
                # GZ Decals often has filenames like 'J094651.40-010228.5.png'
                lookup_key = img_path.stem 
                if lookup_key in metadata_df.index:
                    row = metadata_df.loc[lookup_key]
                    if isinstance(row, pd.DataFrame): # Handle non-unique index
                        row = row.iloc[0]
                    
                    # DiffusionDB
                    if "prompt" in row:
                        caption = str(row["prompt"]).strip()
                    elif "text" in row:
                        caption = str(row["text"]).strip()
                    # GZ Decals morphology mapping
                    elif "smooth-or-featured_smooth_fraction" in row:
                        parts = []
                        if row.get('smooth-or-featured_smooth_fraction', 0) > 0.5:
                            parts.append("a smooth galaxy")
                        elif row.get('smooth-or-featured_featured-or-disk_fraction', 0) > 0.5:
                            parts.append("a featured galaxy")
                        else:
                            parts.append("a galaxy")
                        
                        if row.get('disk-edge-on_yes_fraction', 0) > 0.5:
                            parts.append("seen edge-on")
                        
                        if row.get('has-spiral-arms_yes_fraction', 0) > 0.5:
                            parts.append("with spiral arms")
                        
                        if row.get('bar_strong_fraction', 0) > 0.5:
                            parts.append("and a strong bar")
                        
                        caption = " ".join(parts).capitalize() + "."
            
            # 3. Check for alongside .json (DiffusionDB structure part-xxxx.json)
            if not caption:
                json_path = img_path.parent / (img_path.parent.name + ".json")
                if json_path.exists():
                    try:
                        import json
                        with open(json_path) as jf:
                            data = json.load(jf)
                            # the structure is dict of filenames
                            if img_path.name in data:
                                caption = data[img_path.name].get("p", "").strip()
                    except Exception:
                        pass
                        
            if caption:
                (output_dir / f"{fname}.txt").write_text(caption, encoding="utf-8")

            kept += 1

        except Exception as e:
            skipped_error += 1
            if skipped_error <= 5:
                print(f"  Warning: error on {img_path}: {e}")

    return kept, skipped_aspect, skipped_size, skipped_error


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess image dataset for VQ-VAE / multimodal training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From download_hf.py Arrow output
  python preprocess_image_dataset.py \\
      --input-dir ./datasets/diffusiondb-pixelart \\
      --output-dir ./datasets/pixelart_processed

  # Different target size
  python preprocess_image_dataset.py \\
      --input-dir ./datasets/diffusiondb-pixelart \\
      --output-dir ./datasets/pixelart_128 --image-size 128

  # Plain directory of images
  python preprocess_image_dataset.py \\
      --input-dir ./raw_images/ \\
      --output-dir ./datasets/processed

  # Keep all images (no aspect ratio filtering)
  python preprocess_image_dataset.py \\
      --input-dir ./datasets/diffusiondb-pixelart \\
      --output-dir ./datasets/pixelart_all --aspect-ratio-tol 999
""",
    )

    parser.add_argument("--input-dir", type=str, required=True,
                        help="Input directory (Arrow dataset from download_hf.py, "
                             "or plain directory of images)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for processed images + captions")

    # Column overrides (Arrow datasets only)
    parser.add_argument("--image-col", type=str, default=None,
                        help="Image column name (auto-detected if not set)")
    parser.add_argument("--text-col", type=str, default=None,
                        help="Text/prompt column name (auto-detected if not set)")

    # Output format
    parser.add_argument("--image-format", type=str, default=DEFAULT_IMAGE_FORMAT,
                        choices=["png", "jpg", "webp"],
                        help=f"Output image format (default: {DEFAULT_IMAGE_FORMAT})")

    # Filtering & resizing
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE,
                        help=f"Target square resolution (default: {DEFAULT_IMAGE_SIZE})")
    parser.add_argument("--aspect-ratio-tol", type=float, default=DEFAULT_ASPECT_TOL,
                        help=f"Max aspect ratio deviation from 1:1 (default: {DEFAULT_ASPECT_TOL}). "
                             f"Set to 999 to keep all images.")
    parser.add_argument("--min-size", type=int, default=0,
                        help="Minimum image dimension in pixels (skip smaller)")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Don't search subdirectories (image dir mode only)")

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  Image Dataset Preprocessor")
    print("  Target: VQ-VAE (Phase 1) + Multimodal LLM (Phase 2)")
    print("=" * 60)
    print(f"  Input : {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Size  : {args.image_size}x{args.image_size}")
    print(f"  Aspect ratio tol: {args.aspect_ratio_tol}")
    print()

    # Auto-detect input format
    if is_arrow_dataset(input_dir):
        print("Detected: HuggingFace Arrow dataset (from download_hf.py)")
        kept, skip_ar, skip_sz, skip_err = process_arrow_dataset(
            input_dir, output_dir,
            image_size=args.image_size,
            aspect_tol=args.aspect_ratio_tol,
            image_format=args.image_format,
            image_col=args.image_col,
            text_col=args.text_col,
            min_size=args.min_size,
        )
    else:
        print("Detected: Plain image directory")
        kept, skip_ar, skip_sz, skip_err = process_image_dir(
            input_dir, output_dir,
            image_size=args.image_size,
            aspect_tol=args.aspect_ratio_tol,
            image_format=args.image_format,
            min_size=args.min_size,
            recursive=not args.no_recursive,
        )

    # -- Summary --
    ds_name = input_dir.name
    total = kept + skip_ar + skip_sz + skip_err
    print()
    print("=" * 60)
    print("  Preprocessing complete!")
    print("=" * 60)
    print(f"  Total processed : {total}")
    print(f"  Kept            : {kept}")
    print(f"  Skipped (aspect): {skip_ar}")
    print(f"  Skipped (size)  : {skip_sz}")
    print(f"  Skipped (error) : {skip_err}")
    print(f"  Output dir      : {output_dir}")
    print(f"  Image size      : {args.image_size}x{args.image_size}")
    print()

    txt_count = len(list(output_dir.glob("*.txt")))
    img_count = len(list(output_dir.glob(f"*.{args.image_format}")))
    print(f"  Output files: {img_count} images, {txt_count} captions")

    if txt_count > 0:
        print(f"\n  Ready for VQ-VAE training (Phase 1):")
        print(f"    ./run_train.sh {ds_name}-vqvae --mode vqvae "
              f"--config configs/vqvae_default.json --data-dir {output_dir}")
        print(f"\n  Ready for multimodal training (Phase 2):")
        print(f"    ./run_train.sh {ds_name}-multimodal --mode multimodal "
              f"--config configs/multimodal_pixelart.json --data-dir {output_dir}")
    else:
        print(f"\n  Ready for VQ-VAE training (Phase 1):")
        print(f"    ./run_train.sh {ds_name}-vqvae --mode vqvae "
              f"--config configs/vqvae_default.json --data-dir {output_dir}")
        print(f"\n  No text captions found -- multimodal training (Phase 2) "
              f"requires paired .png + .txt files.")

    # Save metadata
    meta = {
        "source": str(input_dir),
        "image_size": args.image_size,
        "aspect_ratio_tol": args.aspect_ratio_tol,
        "min_size": args.min_size,
        "image_format": args.image_format,
        "total_processed": total,
        "kept": kept,
        "skipped_aspect": skip_ar,
        "skipped_size": skip_sz,
        "skipped_error": skip_err,
        "num_images": img_count,
        "num_captions": txt_count,
    }
    meta_path = output_dir / "preprocessing_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
