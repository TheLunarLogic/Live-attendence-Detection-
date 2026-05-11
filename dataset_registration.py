"""
Dataset Registration Module for Face Recognition Attendance System.

Handles bulk student registration from a dataset folder or ZIP file.
Each image contains a passport-size student photo with text (name + roll no) below it.

Pipeline:
  1. Load images from folder/ZIP
  2. Detect face using MTCNN (top portion)
  3. Parse roll number and name from the filename
  4. Augment the single image to generate 5 samples
  5. Extract FaceNet embeddings and average them
  6. Store in SQLite via Database class
"""

import os
import re
import zipfile
import tempfile
import random
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import torch

from face_utils import normalize_embedding, extract_embedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Image augmentation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _augment_image(pil_img: Image.Image) -> List[Image.Image]:
    """
    Generate 4 augmented variants of a PIL image.

    Transformations applied (one dominant per variant):
      1. Random rotation  ±15°
      2. Horizontal flip
      3. Brightness / contrast jitter
      4. Gaussian blur + slight zoom crop

    Returns:
        List of 4 augmented PIL images (original NOT included).
    """
    augmented = []
    w, h = pil_img.size

    # --- Variant 1: rotation ---
    angle = random.uniform(-15, 15)
    aug = pil_img.rotate(angle, resample=Image.BILINEAR, expand=False)
    augmented.append(aug)

    # --- Variant 2: horizontal flip ---
    aug = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    augmented.append(aug)

    # --- Variant 3: brightness + contrast jitter ---
    aug = pil_img.copy()
    aug = ImageEnhance.Brightness(aug).enhance(random.uniform(0.7, 1.3))
    aug = ImageEnhance.Contrast(aug).enhance(random.uniform(0.8, 1.2))
    augmented.append(aug)

    # --- Variant 4: slight zoom crop + Gaussian noise ---
    crop_frac = random.uniform(0.05, 0.12)
    cx, cy = int(w * crop_frac), int(h * crop_frac)
    aug = pil_img.crop((cx, cy, w - cx, h - cy)).resize((w, h), Image.BILINEAR)
    # Add light Gaussian noise via numpy
    arr = np.array(aug).astype(np.float32)
    noise = np.random.normal(0, 8, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    aug = Image.fromarray(arr)
    augmented.append(aug)

    return augmented


# ──────────────────────────────────────────────────────────────────────────────
# Filename parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_filename(filename: str) -> Tuple[str, str]:
    """
    Parse roll number and student name from filename.
    Expected format: Name_RollNo.jpg or Name_Surname_RollNo.jpg
    """
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split("_")
    
    if len(parts) < 2:
        raise ValueError(f"Invalid filename format: {filename}")
        
    roll_no = parts[-1]
    student_name = " ".join(parts[:-1])
    
    return student_name, roll_no


# ──────────────────────────────────────────────────────────────────────────────
# Core registration pipeline
# ──────────────────────────────────────────────────────────────────────────────

def register_from_dataset(
    image_paths: List[str],
    class_id: int,
    mtcnn,
    resnet,
    device,
    db,
    progress_callback=None,
) -> Dict:
    """
    Register multiple students from passport-style images.

    Args:
        image_paths:       List of absolute image file paths.
        class_id:          Target class ID in the database.
        mtcnn:             MTCNN detector (from face_utils.load_face_models).
        resnet:            InceptionResnetV1 model.
        device:            torch.device.
        db:                Database instance.
        progress_callback: Optional callable(current, total, message) for UI updates.

    Returns:
        Dict with keys:
            total, registered, skipped, parsing_failures, errors
            (errors is a list of (filename, reason) tuples)
    """
    stats = {
        "total": len(image_paths),
        "registered": 0,
        "skipped": 0,
        "parsing_failures": 0,
        "errors": [],
    }

    for i, img_path in enumerate(image_paths):
        filename = Path(img_path).name

        if progress_callback:
            progress_callback(i + 1, len(image_paths), f"Processing: {filename}")

        # ── Load image ──
        try:
            pil_full = Image.open(img_path).convert("RGB")
            img_np = np.array(pil_full)
        except Exception as e:
            stats["skipped"] += 1
            stats["errors"].append((filename, f"Load error: {e}"))
            continue

        # ── Parse roll & name from filename ──
        try:
            name, roll_no = parse_filename(filename)
        except ValueError as e:
            stats["parsing_failures"] += 1
            stats["errors"].append((filename, str(e)))
            stats["skipped"] += 1
            continue

        # ── Check duplicate ──
        if db.check_duplicate_roll(class_id, roll_no):
            stats["skipped"] += 1
            stats["errors"].append((filename, f"Duplicate roll number: {roll_no}"))
            continue

        # ── Convert to PIL for augmentation ──
        face_pil = pil_full

        # ── Generate original + 4 augmented = 5 samples ──
        augmented_images = [face_pil] + _augment_image(face_pil)

        # ── Extract embeddings ──
        embeddings = []
        for aug_img in augmented_images:
            aug_np = np.array(aug_img)
            ok, emb, _ = extract_embedding(aug_np, mtcnn, resnet, device)
            if ok:
                embeddings.append(emb)

        if len(embeddings) == 0:
            stats["skipped"] += 1
            stats["errors"].append((filename, "No face detected in any sample"))
            continue

        # ── Average + normalize ──
        avg_embedding = normalize_embedding(np.mean(embeddings, axis=0))

        # ── Store in database ──
        ok, msg = db.add_student(class_id, roll_no, name, avg_embedding)
        if ok:
            stats["registered"] += 1
            logger.info(f"Registered: {name} ({roll_no}) from {filename}")
        else:
            stats["skipped"] += 1
            stats["errors"].append((filename, msg))

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# ZIP extraction helper
# ──────────────────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_zip_to_temp(zip_bytes: bytes) -> Tuple[str, List[str]]:
    """
    Extract a ZIP file (provided as bytes) to a system temp directory.

    Returns:
        (temp_dir_path, sorted_list_of_image_paths)
    """
    tmp_dir = tempfile.mkdtemp(prefix="face_dataset_")
    zip_path = os.path.join(tmp_dir, "upload.zip")

    with open(zip_path, "wb") as f:
        f.write(zip_bytes)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    # Collect all image files recursively
    image_paths = []
    for root, _, files in os.walk(tmp_dir):
        for fname in files:
            if Path(fname).suffix.lower() in ALLOWED_EXTENSIONS:
                image_paths.append(os.path.join(root, fname))

    return tmp_dir, sorted(image_paths)


def collect_images_from_folder(folder_path: str) -> List[str]:
    """
    Collect all image file paths from a folder (non-recursive).

    Args:
        folder_path: Path to the images folder.

    Returns:
        Sorted list of image absolute paths.
    """
    paths = []
    for fname in os.listdir(folder_path):
        if Path(fname).suffix.lower() in ALLOWED_EXTENSIONS:
            paths.append(os.path.join(folder_path, fname))
    return sorted(paths)
