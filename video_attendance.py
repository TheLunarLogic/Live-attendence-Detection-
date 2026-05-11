"""
Video Attendance Module — Optimized Production Pipeline
=========================================================

High-accuracy video attendance with intelligent performance optimizations:
  1. Async threaded frame reading for I/O overlap
  2. Motion-based frame pre-filtering (skip static frames)
  3. Detection scheduling — full detection every N frames, tracking between
  4. Adaptive grid detection (grid vs full-frame based on face count)
  5. Adaptive detection resolution (downscale based on face size)
  6. Face quality filtering (min size + cached blur rejection)
  7. Detection confidence filtering
  8. Centroid-based tracking with embedding cache + confirmed locking
  9. Batch FaceNet inference (N faces → 1 forward pass)
  10. Track-level embedding cache with smart refresh
  11. Student-level early exit (shrinking active embedding matrix)
  12. Vectorized cosine similarity against active students only
  13. Multi-frame aggregation — stores BEST similarity per student
  14. Temporal voting — requires MIN_VOTES confident frames before marking
  15. Final decision: votes >= MIN_VOTES AND best_similarity >= threshold
  16. Early stop when all students are confidently recognized
  17. Full pipeline profiling with cache statistics
"""

import os
import time
import tempfile
import logging
import threading
import queue
from typing import List, Dict, Set, Tuple, Optional, Callable
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from PIL import Image

from face_utils import normalize_embedding
from video_debug import DebugSession, DEBUG_MODE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE MODE — selects optimization aggressiveness
# ══════════════════════════════════════════════════════════════════════════════

PERFORMANCE_MODE = "balanced"  # "max_accuracy", "balanced", "fast"

_MODE_CONFIGS = {
    "max_accuracy": {
        "detection_interval": 1,
        "motion_threshold": 0.0,
        "blur_recheck_interval": 1,
        "embedding_cache_frames": 1,
        "embedding_cache_shift_px": 0,
    },
    "balanced": {
        "detection_interval": 2,      # RESTORED: detect every 2 frames for reliability
        "motion_threshold": 0.0,      # DISABLED: motion filter was skipping valid frames
        "blur_recheck_interval": 10,
        "embedding_cache_frames": 15,
        "embedding_cache_shift_px": 10,
    },
    "fast": {
        "detection_interval": 8,
        "motion_threshold": 5.0,
        "blur_recheck_interval": 15,
        "embedding_cache_frames": 20,
        "embedding_cache_shift_px": 15,
    },
}

def _get_mode_config() -> dict:
    return _MODE_CONFIGS.get(PERFORMANCE_MODE, _MODE_CONFIGS["balanced"])


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL CONFIGURATION — NOT exposed to UI
# ══════════════════════════════════════════════════════════════════════════════

FRAME_SKIP = 1                  # Process every frame for maximum recall
SIMILARITY_THRESHOLD = 0.62     # Final-calibrated for compressed classroom video
SIMILARITY_THRESHOLD_RELAXED = 0.61  # Relaxed further once student has several votes
VOTE_RELAXED_CUTOFF = 4         # Vote count at which threshold relaxes
MIN_VOTE_RATIO = 0.30           # Stricter ratio to prevent false positives

# ── New Constraints ──
MIN_FINAL_SMOOTHED_SIM = 0.60        # Hard global floor
MIN_CONSECUTIVE_MATCHES = 3          # Requires stability across nearby frames
UPSCALE_FACTOR = 1.5            # RESTORED: upscale for small-face detection
DETECTION_DOWNSCALE = 1.0      # RESTORED: no downscale — preserve facial detail
MIN_DETECTION_CONF = 0.4        # Ignore detections with probability below this
MIN_FACE_SIZE = 35              # Reduced: recover distant/small faces (was 40)
BLUR_THRESHOLD = 12.0           # Hard reject below this — completely unusable
MIN_VOTES = 3                   # Votes needed for standard confirmation
HIGH_CONF_THRESHOLD = 0.80     # Very strong match → confirm with fewer votes
HIGH_CONF_MIN_VOTES = 2        # Min votes when similarity > HIGH_CONF_THRESHOLD
COOLDOWN_FRAMES = 5            # Frames to skip before re-recognizing a tracked face
TRACK_DISTANCE_PX = 60         # Max centroid distance (px) to consider same track
SIM_HISTORY_TOP_K = 3          # Use average of top-K similarities for smoothed decision
SIM_HISTORY_MAX_LEN = 30       # Cap history length to prevent RAM growth on long videos
GRID_OVERLAP = 0.10            # ~10% overlap between grid quadrants

# ── Multi-level blur quality tiers ──
BLUR_HARD_REJECT = 8.0          # Below this → completely unusable, hard reject
BLUR_LOW_QUALITY = 15.0         # 8–15 → usable but low quality → reduced vote weight
VOTE_WEIGHT_HIGH = 1.0          # blur >= 15 → full vote weight
VOTE_WEIGHT_LOW = 0.5           # blur 8–15 → half vote weight

# ── Borderline candidate buffer ──
BORDERLINE_SIM_FLOOR = 0.52     # Track candidates with similarity >= this
BORDERLINE_MIN_APPEARANCES = 8  # Need this many consistent appearances
BORDERLINE_MIN_AVG_SIM = 0.48   # Average similarity must exceed this
BORDERLINE_CONSISTENCY = 0.6    # Fraction of appearances matching same student

# ── Adaptive detection resolution thresholds ──
DOWNSCALE_LARGE_FACE = 0.4     # avg face > 120px → aggressive downscale
DOWNSCALE_MEDIUM_FACE = 0.5    # avg face 80-120px → standard downscale
DOWNSCALE_SMALL_FACE = 0.7     # avg face < 80px → gentle downscale

# ── Embedding refresh thresholds ──
EMBEDDING_REFRESH_SHIFT_PX = 15   # re-extract if centroid moves > this
EMBEDDING_REFRESH_SIZE_PCT = 0.20 # re-extract if face size changes > 20%
EMBEDDING_REFRESH_INTERVAL = 20   # re-extract every N frames regardless

# ── Debug frame export ──
SAVE_DEBUG_FRAMES = True          # Export annotated debug frames to debug_frames/
DEBUG_FRAME_SAMPLE_RATE = 5       # Save every Nth processed frame (1 = all)

# ── Debug flag (imported from video_debug; toggle DEBUG_MODE there) ──
# Set DEBUG_MODE = True in video_debug.py to enable full diagnostics.
# All debug output goes to debug/run_<timestamp>/ + debug_report_<timestamp>.zip


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Profiler — timing instrumentation
# ══════════════════════════════════════════════════════════════════════════════

class PipelineProfiler:
    """Accumulates timing data per pipeline stage and prints periodic summaries."""

    def __init__(self, report_interval: int = 50):
        self._interval = report_interval
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._counters: Dict[str, int] = defaultdict(int)
        self._frame_count = 0

    def tick(self, stage: str, elapsed_ms: float):
        self._timings[stage].append(elapsed_ms)

    def count(self, counter: str, n: int = 1):
        self._counters[counter] += n

    def frame_done(self):
        self._frame_count += 1
        if self._frame_count % self._interval == 0:
            self._print_summary()

    def _print_summary(self):
        logger.info(f"[PERF] ── Frame {self._frame_count} averages ──")
        for stage, times in sorted(self._timings.items()):
            if times:
                avg = sum(times[-self._interval:]) / min(len(times), self._interval)
                logger.info(f"[PERF]   {stage:20s}: {avg:6.1f}ms")
        for name, val in sorted(self._counters.items()):
            logger.info(f"[PERF]   {name:20s}: {val}")
        # Cache hit ratio
        hits = self._counters.get("emb_cache_hits", 0)
        misses = self._counters.get("emb_cache_misses", 0)
        total = hits + misses
        if total > 0:
            logger.info(f"[PERF]   {'cache_hit_ratio':20s}: {hits/total*100:.1f}%")

    def final_summary(self) -> Dict:
        stats = {}
        for stage, times in self._timings.items():
            stats[f"avg_{stage}_ms"] = sum(times) / len(times) if times else 0
        stats.update(dict(self._counters))
        hits = self._counters.get("emb_cache_hits", 0)
        misses = self._counters.get("emb_cache_misses", 0)
        total = hits + misses
        stats["cache_hit_ratio"] = (hits / total * 100) if total > 0 else 0
        self._print_summary()
        return stats


# ══════════════════════════════════════════════════════════════════════════════
# Async Video Reader — threaded frame buffering
# ══════════════════════════════════════════════════════════════════════════════

class VideoCaptureThread:
    """Background thread that continuously reads frames into a bounded queue."""

    def __init__(self, video_path: str, queue_size: int = 32):
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stopped = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stopped:
            ret, frame = self._cap.read()
            if not ret:
                self._queue.put(None)  # sentinel
                break
            self._queue.put(frame)

    def read(self) -> Optional[np.ndarray]:
        """Get next frame (blocks until available). Returns None at end."""
        try:
            frame = self._queue.get(timeout=10.0)
            return frame
        except queue.Empty:
            return None

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 25.0

    @property
    def cap(self) -> cv2.VideoCapture:
        return self._cap

    def release(self):
        self._stopped = True
        self._thread.join(timeout=5.0)
        self._cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# Face quality helpers
# ══════════════════════════════════════════════════════════════════════════════

def _is_face_large_enough(box: np.ndarray) -> bool:
    """Check if bounding box meets minimum size requirement."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE


def _is_face_sharp(frame_rgb: np.ndarray, box: np.ndarray) -> bool:
    """Reject blurry face crops using Laplacian variance."""
    x1, y1, x2, y2 = [int(c) for c in box]
    h, w = frame_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return False
    crop = frame_rgb[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance >= BLUR_THRESHOLD


def _compute_avg_face_size(boxes: np.ndarray) -> float:
    """Compute average face bounding box size (mean of width and height)."""
    if boxes is None or len(boxes) == 0:
        return 0.0
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    return float(np.mean((widths + heights) / 2))


def _get_adaptive_downscale(avg_face_size: float) -> float:
    """Select detection downscale factor based on average detected face size."""
    if avg_face_size > 120:
        return DOWNSCALE_LARGE_FACE
    elif avg_face_size > 80:
        return DOWNSCALE_MEDIUM_FACE
    else:
        return DOWNSCALE_SMALL_FACE


# ══════════════════════════════════════════════════════════════════════════════
# Hybrid detection precheck — lightweight Haar cascade gate
# ══════════════════════════════════════════════════════════════════════════════

_HAAR_CASCADE = None

def _get_haar_cascade():
    """Lazy-load the Haar cascade classifier."""
    global _HAAR_CASCADE
    if _HAAR_CASCADE is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _HAAR_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _HAAR_CASCADE


def _haar_precheck(frame_rgb: np.ndarray, min_size: int = 30) -> bool:
    """
    Fast Haar cascade check — returns True if ANY face candidate is found.
    Used as a cheap gate before expensive MTCNN inference.
    Runs on a small grayscale thumbnail for speed (~1-2ms).
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (320, 240))
    cascade = _get_haar_cascade()
    faces = cascade.detectMultiScale(small, scaleFactor=1.3, minNeighbors=2,
                                     minSize=(min_size, min_size))
    return len(faces) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Hardware-adaptive configuration
# ══════════════════════════════════════════════════════════════════════════════

def _detect_hardware_config() -> dict:
    """
    Auto-detect hardware and return optimized settings.
    Adjusts batch size, detection interval, and thread count based on
    whether a CUDA GPU is available and its memory.
    """
    has_gpu = torch.cuda.is_available()
    config = {
        "batch_size": 1,
        "detection_interval_multiplier": 1.0,
        "prep_threads": 2,
        "device_type": "cpu",
    }
    if has_gpu:
        config["device_type"] = "cuda"
        try:
            mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        except Exception:
            mem_gb = 4.0
        if mem_gb >= 8:
            config["batch_size"] = 16
            config["prep_threads"] = 4
        elif mem_gb >= 4:
            config["batch_size"] = 8
            config["prep_threads"] = 4
        else:
            config["batch_size"] = 4
            config["prep_threads"] = 2
    else:
        # CPU mode — larger detection intervals, smaller batches
        config["batch_size"] = 2
        config["detection_interval_multiplier"] = 1.5
        config["prep_threads"] = 2
    return config


# ══════════════════════════════════════════════════════════════════════════════
# Centroid tracker — enhanced with embedding cache and confirmed locking
# ══════════════════════════════════════════════════════════════════════════════

class CentroidTracker:
    """
    Lightweight centroid-based tracker with embedding caching.

    Each tracked object has:
      - centroid (cx, cy)
      - student_id (None until recognized)
      - last_seen frame index
      - cooldown counter
      - confirmed (bool) — True when student fully confirmed, stops re-embedding
      - cached_embedding — last computed embedding for this track
      - last_embedding_frame — frame at which embedding was last computed
      - last_box_size — (w, h) of last bounding box for size-change detection
    """

    def __init__(self, max_distance: float = TRACK_DISTANCE_PX, cooldown: int = COOLDOWN_FRAMES):
        self._next_id = 0
        self._tracks: Dict[int, dict] = {}
        self._max_dist = max_distance
        self._cooldown = cooldown

    def update(self, boxes: np.ndarray, frame_idx: int) -> List[Tuple[int, np.ndarray, bool]]:
        """
        Update tracker with new detections.

        Returns:
            List of (track_id, box, needs_recognition) tuples.
        """
        if len(boxes) == 0:
            return []

        new_centroids = np.array([
            [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes
        ])

        results: List[Tuple[int, np.ndarray, bool]] = []
        used_track_ids: Set[int] = set()
        used_det_ids: Set[int] = set()

        if self._tracks:
            track_ids = list(self._tracks.keys())
            track_centroids = np.array([self._tracks[tid]["centroid"] for tid in track_ids])

            dists = np.linalg.norm(
                track_centroids[:, None, :] - new_centroids[None, :, :], axis=2
            )

            for _ in range(min(len(track_ids), len(boxes))):
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                t_idx, d_idx = int(min_idx[0]), int(min_idx[1])
                if dists[t_idx, d_idx] > self._max_dist:
                    break
                tid = track_ids[t_idx]
                track = self._tracks[tid]
                track["centroid"] = new_centroids[d_idx].tolist()
                track["last_seen"] = frame_idx

                # Confirmed tracks NEVER need recognition again
                needs_recog = True
                if track.get("confirmed", False):
                    needs_recog = False
                elif track["student_id"] is not None:
                    if track["cooldown_remaining"] > 0:
                        track["cooldown_remaining"] -= 1
                        needs_recog = False
                    else:
                        track["cooldown_remaining"] = self._cooldown
                        needs_recog = True

                results.append((tid, boxes[d_idx], needs_recog))
                used_track_ids.add(tid)
                used_det_ids.add(d_idx)

                dists[t_idx, :] = float("inf")
                dists[:, d_idx] = float("inf")

        for d_idx in range(len(boxes)):
            if d_idx not in used_det_ids:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "centroid": new_centroids[d_idx].tolist(),
                    "student_id": None,
                    "last_seen": frame_idx,
                    "cooldown_remaining": 0,
                    "confirmed": False,
                    "cached_embedding": None,
                    "last_embedding_frame": -999,
                    "last_box_size": None,
                }
                results.append((tid, boxes[d_idx], True))

        stale = [tid for tid, t in self._tracks.items() if frame_idx - t["last_seen"] > 30]
        for tid in stale:
            del self._tracks[tid]

        return results

    def assign_student(self, track_id: int, student_id: int):
        """Link a track to a recognized student and start cooldown."""
        if track_id in self._tracks:
            self._tracks[track_id]["student_id"] = student_id
            self._tracks[track_id]["cooldown_remaining"] = self._cooldown

    def confirm_track(self, track_id: int):
        """Mark track as permanently confirmed — no more embedding extraction."""
        if track_id in self._tracks:
            self._tracks[track_id]["confirmed"] = True

    def get_track(self, track_id: int) -> Optional[dict]:
        return self._tracks.get(track_id)

    def cache_embedding(self, track_id: int, emb: np.ndarray, frame_idx: int, box: np.ndarray):
        """Store embedding and metadata for cache-hit checks."""
        if track_id in self._tracks:
            t = self._tracks[track_id]
            t["cached_embedding"] = emb
            t["last_embedding_frame"] = frame_idx
            w, h = float(box[2] - box[0]), float(box[3] - box[1])
            t["last_box_size"] = (w, h)


# ══════════════════════════════════════════════════════════════════════════════
# Grid-based face detection with coordinate remapping
# ══════════════════════════════════════════════════════════════════════════════

def _detect_faces_grid(
    frame_rgb: np.ndarray,
    mtcnn,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run MTCNN on the full frame AND 4 quadrants.
    Remap quadrant bounding boxes back to full-frame coordinates.
    Deduplicate overlapping detections via IoU filtering.

    Returns:
        (boxes, probs) or (None, None) if no faces found.
        boxes: (N, 4) ndarray   probs: (N,) ndarray
    """
    h, w = frame_rgb.shape[:2]

    # Compute overlap margins (~10% of each half-dimension)
    ov_h = int(h * GRID_OVERLAP / 2)  # vertical overlap per edge
    ov_w = int(w * GRID_OVERLAP / 2)  # horizontal overlap per edge

    mid_h = h // 2
    mid_w = w // 2

    # Regions: (sub_image, x_offset, y_offset)
    # Quadrants include ~10% overlap so faces on boundaries are not missed
    regions = [
        (frame_rgb, 0, 0),                                                        # full frame
        (frame_rgb[0:mid_h + ov_h, 0:mid_w + ov_w],           0, 0),              # top-left
        (frame_rgb[0:mid_h + ov_h, mid_w - ov_w:w],           mid_w - ov_w, 0),   # top-right
        (frame_rgb[mid_h - ov_h:h, 0:mid_w + ov_w],           0, mid_h - ov_h),   # bottom-left
        (frame_rgb[mid_h - ov_h:h, mid_w - ov_w:w],           mid_w - ov_w, mid_h - ov_h),  # bottom-right
    ]

    all_boxes = []
    all_probs = []

    for region_img, x_off, y_off in regions:
        if region_img.size == 0:
            continue
        try:
            pil_img = Image.fromarray(region_img)
            boxes, probs = mtcnn.detect(pil_img)
            if boxes is None or probs is None:
                continue
            for box, prob in zip(boxes, probs):
                if prob is not None and prob < MIN_DETECTION_CONF:
                    continue
                # Remap to full-frame coordinates
                remapped = box.copy()
                remapped[0] += x_off
                remapped[1] += y_off
                remapped[2] += x_off
                remapped[3] += y_off
                all_boxes.append(remapped)
                all_probs.append(prob if prob is not None else 0.0)
        except Exception:
            continue

    if not all_boxes:
        return None, None

    all_boxes = np.array(all_boxes)
    all_probs = np.array(all_probs)

    # Deduplicate via greedy NMS (IoU > 0.5 → keep higher-confidence one)
    keep = _nms(all_boxes, all_probs, iou_threshold=0.5)
    return all_boxes[keep], all_probs[keep]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
    """Non-maximum suppression."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


# ══════════════════════════════════════════════════════════════════════════════
# Full-frame-only detection (used when adaptive grid switches off quadrants)
# ══════════════════════════════════════════════════════════════════════════════

def _detect_faces_fullframe(
    frame_rgb: np.ndarray,
    mtcnn,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Run MTCNN on full frame only (no quadrants). Much faster than grid."""
    try:
        pil_img = Image.fromarray(frame_rgb)
        boxes, probs = mtcnn.detect(pil_img)
        if boxes is None or probs is None:
            return None, None
        mask = [i for i, p in enumerate(probs) if p is not None and p >= MIN_DETECTION_CONF]
        if not mask:
            return None, None
        return boxes[mask], probs[mask]
    except Exception:
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# Embedding extraction — single + batch modes
# ══════════════════════════════════════════════════════════════════════════════

_FACENET_INPUT_SIZE = 160  # InceptionResnetV1 expected input


def _prepare_face_tensor(frame_rgb: np.ndarray, box: np.ndarray) -> Optional[torch.Tensor]:
    """Crop, resize and normalize a face to a ready-to-infer tensor (no batch dim)."""
    try:
        x1, y1, x2, y2 = [int(c) for c in box]
        h, w = frame_rgb.shape[:2]
        margin = int(max(x2 - x1, y2 - y1) * 0.15)
        x1m, y1m = max(0, x1 - margin), max(0, y1 - margin)
        x2m, y2m = min(w, x2 + margin), min(h, y2 + margin)
        crop = frame_rgb[y1m:y2m, x1m:x2m]
        if crop.size == 0:
            return None
        face_resized = cv2.resize(crop, (_FACENET_INPUT_SIZE, _FACENET_INPUT_SIZE))
        face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float()
        face_tensor = (face_tensor - 127.5) / 128.0
        return face_tensor
    except Exception:
        return None


def _batch_extract_embeddings(
    frame_rgb: np.ndarray,
    boxes: List[np.ndarray],
    resnet,
    device,
    max_threads: int = 4,
) -> List[Optional[np.ndarray]]:
    """
    Extract embeddings for multiple faces in a single batched FaceNet forward pass.
    Falls back to individual extraction if batch fails.
    """
    if not boxes:
        return []

    # Prepare tensors in parallel threads (CPU-bound crop/resize)
    with ThreadPoolExecutor(max_workers=min(max_threads, len(boxes))) as pool:
        futures = [pool.submit(_prepare_face_tensor, frame_rgb, box) for box in boxes]
        tensors = [f.result() for f in futures]

    # Separate valid from failed
    valid_indices = [i for i, t in enumerate(tensors) if t is not None]
    if not valid_indices:
        return [None] * len(boxes)

    valid_tensors = [tensors[i] for i in valid_indices]

    try:
        batch = torch.stack(valid_tensors).to(device, non_blocking=True)
        with torch.no_grad():
            batch_embs = resnet(batch).cpu().numpy()

        # Scatter results back
        results: List[Optional[np.ndarray]] = [None] * len(boxes)
        for j, orig_idx in enumerate(valid_indices):
            results[orig_idx] = normalize_embedding(batch_embs[j])
        return results
    except Exception as e:
        logger.debug(f"Batch embedding failed, falling back to individual: {e}")
        results = [None] * len(boxes)
        for idx in valid_indices:
            results[idx] = _extract_embedding_for_box(frame_rgb, boxes[idx], resnet, device)
        return results


def _extract_embedding_for_box(
    frame_rgb: np.ndarray,
    box: np.ndarray,
    resnet,
    device,
) -> Optional[np.ndarray]:
    """Single-face embedding extraction (fallback for batch failures)."""
    try:
        t = _prepare_face_tensor(frame_rgb, box)
        if t is None:
            return None
        t = t.unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad():
            emb = resnet(t).cpu().numpy()[0]
        return normalize_embedding(emb)
    except Exception as e:
        logger.debug(f"Embedding extraction error: {e}")
        return None


def _should_refresh_embedding(
    track: dict, box: np.ndarray, frame_idx: int, mode_cfg: dict
) -> bool:
    """
    Decide whether a tracked face needs a fresh embedding extraction,
    or can reuse its cached embedding.
    """
    if track.get("cached_embedding") is None:
        return True  # no cache yet

    # Check frame gap
    frame_gap = frame_idx - track.get("last_embedding_frame", -999)
    if frame_gap >= EMBEDDING_REFRESH_INTERVAL:
        return True

    # Check centroid movement
    cache_shift = mode_cfg.get("embedding_cache_shift_px", 10)
    if cache_shift > 0:
        cx_new = (box[0] + box[2]) / 2
        cy_new = (box[1] + box[3]) / 2
        cx_old, cy_old = track.get("centroid", [cx_new, cy_new])
        shift = ((cx_new - cx_old)**2 + (cy_new - cy_old)**2) ** 0.5
        if shift > EMBEDDING_REFRESH_SHIFT_PX:
            return True

    # Check size change
    prev_size = track.get("last_box_size")
    if prev_size is not None:
        w_new = float(box[2] - box[0])
        h_new = float(box[3] - box[1])
        w_old, h_old = prev_size
        if w_old > 0 and h_old > 0:
            size_change = abs(w_new - w_old) / w_old + abs(h_new - h_old) / h_old
            if size_change > EMBEDDING_REFRESH_SIZE_PCT:
                return True

    # Cache is still valid
    cache_frames = mode_cfg.get("embedding_cache_frames", 15)
    return frame_gap >= cache_frames


# ══════════════════════════════════════════════════════════════════════════════
# Debug frame export — visual diagnostics for detection pipeline
# ══════════════════════════════════════════════════════════════════════════════

_DEBUG_FRAMES_DIR = "debug_frames"

def _save_debug_frame(
    frame_rgb: np.ndarray,
    frame_idx: int,
    boxes: np.ndarray,
    quality_mask: list,
    blur_scores: list,
    face_sizes: list,
    probs,
):
    """
    Export an annotated frame showing every detected box with diagnostics:
      - GREEN box: passed quality filter
      - RED box: rejected (blur or size)
      - Text overlay: blur score, face size, confidence
    """
    os.makedirs(_DEBUG_FRAMES_DIR, exist_ok=True)
    vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    blur_idx = 0  # blur_scores only includes faces that passed size check
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(c) for c in box]
        passed = i in quality_mask
        color = (0, 200, 0) if passed else (0, 0, 220)  # green / red in BGR
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Build annotation text
        sz = face_sizes[i] if i < len(face_sizes) else 0
        conf = float(probs[i]) if probs is not None and i < len(probs) else 0.0
        label_parts = [f"sz={sz:.0f}", f"c={conf:.2f}"]

        # Blur score only exists for faces that passed size check
        if sz >= MIN_FACE_SIZE and blur_idx < len(blur_scores):
            label_parts.append(f"blur={blur_scores[blur_idx]:.1f}")
            blur_idx += 1

        label = " ".join(label_parts)
        status = "OK" if passed else "REJ"
        cv2.putText(vis, f"{status} {label}", (x1, max(y1 - 6, 12)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(os.path.join(_DEBUG_FRAMES_DIR, f"frame_{frame_idx:05d}.jpg"), vis)


# ══════════════════════════════════════════════════════════════════════════════
# Main video processing function
# ══════════════════════════════════════════════════════════════════════════════

def process_video_for_attendance(
    video_path: str,
    students: List[Dict],
    mtcnn,
    resnet,
    device,
    progress_callback: Optional[Callable[[int, int, str, int], None]] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Process a video file and return matched students for attendance.
    Optimized pipeline with detection scheduling, batch inference, embedding
    caching, motion filtering, and adaptive resolution — all accuracy logic
    (temporal voting, adaptive thresholds, smoothing) is preserved exactly.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # ── GPU optimization ──
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # ── Async video reader ──
    vid_reader = VideoCaptureThread(video_path)
    total_frames = vid_reader.total_frames
    fps = vid_reader.fps

    # ── Debug + profiler ──
    dbg = DebugSession(enabled=DEBUG_MODE)
    dbg.log_video_info(vid_reader.cap, video_path)
    prof = PipelineProfiler(report_interval=50)
    mode_cfg = _get_mode_config()
    hw_cfg = _detect_hardware_config()
    logger.info(f"[HW] Device: {hw_cfg['device_type']} | batch_size={hw_cfg['batch_size']} | threads={hw_cfg['prep_threads']}")

    video_stats = {
        "total_frames": total_frames, "processed_frames": 0,
        "fps": fps, "faces_detected_total": 0, "recognition_attempts": 0,
    }

    # ── Filter 'unk' students ──
    valid_students = [s for s in students if s["name"].strip().lower() != "unk"]
    n_excluded = len(students) - len(valid_students)
    if n_excluded:
        logger.info(f"[FILTER] Excluded {n_excluded} 'unk' student(s) from matching matrix")

    # ── Student embedding matrix (active = not yet confirmed) ──
    student_ids = [s["id"] for s in valid_students]
    student_embeddings = np.array([s["embedding"] for s in valid_students])
    student_lookup = {s["id"]: s for s in valid_students}
    # Active subset for early-exit optimization
    active_ids = list(student_ids)
    active_embeddings = student_embeddings.copy()
    _active_dirty = False  # flag to rebuild active matrix

    logger.info(f"[INFO] Matching against {len(valid_students)} students (excluded {n_excluded} unk)")
    for s in valid_students:
        logger.debug(f"[DEBUG]   id={s['id']}  name={s['name']}  "
                     f"embedding_norm={float(np.linalg.norm(s['embedding'])):.4f}")

    # ── Accumulators (core accuracy logic preserved) ──
    student_best_similarity: Dict[int, float] = {}
    student_votes: Dict[int, int] = {}
    student_weighted_votes: Dict[int, float] = {}  # quality-weighted vote accumulator
    student_frame_counts: Dict[int, int] = {}
    student_sim_history: Dict[int, List[float]] = {}
    confirmed_students: Set[int] = set()

    # ── Borderline candidate buffer ──
    # Tracks sub-threshold but potentially valid candidates for temporal promotion
    # {student_id: {"appearances": int, "sims": [float], "student_counts": {sid: int}}}
    borderline_buffer: Dict[int, dict] = {}
    borderline_promoted: int = 0  # counter for diagnostics

    # ── Strict Identity Tracking ──
    student_last_seen_frame: Dict[int, int] = {}
    student_consecutive_matches: Dict[int, int] = {}
    consecutive_match_failures: int = 0
    confirmations_below_0_60: int = 0
    confirmations_below_ratio_threshold: int = 0

    tracker = CentroidTracker()

    frame_idx = 0
    processed = 0
    current_frame_skip = FRAME_SKIP

    # ── Optimization state ──
    prev_gray = None                    # for motion detection
    last_detected_boxes = None          # for detection scheduling
    expected_face_count = 0             # for adaptive grid
    use_grid_detection = True           # adaptive grid toggle
    detection_interval = max(1, int(mode_cfg["detection_interval"] * hw_cfg["detection_interval_multiplier"]))
    motion_threshold = mode_cfg["motion_threshold"]
    blur_recheck_interval = mode_cfg["blur_recheck_interval"]
    prep_threads = hw_cfg["prep_threads"]

    logger.info(f"Starting video processing: {total_frames} frames @ {fps:.1f} FPS")
    logger.info(f"Mode: {PERFORMANCE_MODE} | detection_interval={detection_interval}")
    logger.info(f"DETECTION_DOWNSCALE={DETECTION_DOWNSCALE} | UPSCALE_FACTOR={UPSCALE_FACTOR}")
    logger.info(f"BLUR_THRESHOLD={BLUR_THRESHOLD} | MIN_FACE_SIZE={MIN_FACE_SIZE}")
    logger.info(f"motion_threshold={motion_threshold} | SAVE_DEBUG_FRAMES={SAVE_DEBUG_FRAMES}")
    logger.info(f"Registered students: {len(students)} | Threshold: {SIMILARITY_THRESHOLD} | Min votes: {MIN_VOTES}")

    while True:
        t_frame_start = time.perf_counter()
        frame_bgr = vid_reader.read()
        if frame_bgr is None:
            break

        frame_idx += 1

        # Dynamic frame skip — ramp up as more students get confirmed
        n_valid = len(valid_students)
        n_confirmed = len(confirmed_students)
        if n_valid > 0:
            if n_confirmed >= int(n_valid * 0.75):
                current_frame_skip = 3
            elif n_confirmed >= n_valid // 2:
                current_frame_skip = 2
        if current_frame_skip > 1 and (frame_idx % current_frame_skip) != 0:
            continue

        processed += 1
        video_stats["processed_frames"] = processed
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        del frame_bgr

        # ── Motion pre-filter: skip near-static frames ──
        if motion_threshold > 0:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            gray_small = cv2.resize(gray, (160, 120))
            if prev_gray is not None:
                diff = cv2.absdiff(gray_small, prev_gray)
                motion_score = float(np.mean(diff))
                if motion_score < motion_threshold and last_detected_boxes is not None:
                    prev_gray = gray_small
                    prof.count("frames_skipped_motion")
                    continue
            prev_gray = gray_small

        dbg.save_raw_frame(frame_rgb, frame_idx)

        # ── Build detection frame (upscale then optionally downscale) ──
        # Upscale improves small-face detectability without degrading embedding crops
        h_orig, w_orig = frame_rgb.shape[:2]
        if UPSCALE_FACTOR != 1.0:
            detect_base = cv2.resize(
                frame_rgb, None,
                fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            detect_base = frame_rgb

        ds = DETECTION_DOWNSCALE
        if ds != 1.0:
            detect_frame = cv2.resize(detect_base, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
        else:
            detect_frame = detect_base

        # ── Detection scheduling: full detect every N frames, reuse between ──
        run_full_detection = (frame_idx % detection_interval == 1) or last_detected_boxes is None
        t_det = time.perf_counter()

        if run_full_detection:
            # MTCNN is the primary detector — NO precheck gate
            if use_grid_detection:
                boxes, probs = _detect_faces_grid(detect_frame, mtcnn)
            else:
                boxes, probs = _detect_faces_fullframe(detect_frame, mtcnn)

            # Remap boxes back to original scale (accounting for upscale + downscale)
            scale_factor = UPSCALE_FACTOR * ds
            if boxes is not None and scale_factor != 1.0:
                boxes = boxes / scale_factor

            if boxes is not None and len(boxes) > 0:
                last_detected_boxes = boxes
                avg_sz = _compute_avg_face_size(boxes)
                expected_face_count = len(boxes)
                use_grid_detection = expected_face_count >= 3
                prof.count("full_detections")
            else:
                last_detected_boxes = None
        else:
            boxes = last_detected_boxes
            probs = None
            prof.count("detection_reuses")

        prof.tick("detection", (time.perf_counter() - t_det) * 1000)

        if boxes is None or len(boxes) == 0:
            dbg.log_detection(frame_idx, None, None, [], 0, 0)
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, total_frames,
                    f"Processing frame {frame_idx}/{total_frames}…", len(confirmed_students))
            prof.frame_done()
            continue

        # ── Multi-level quality filtering (replaces binary blur rejection) ──
        quality_mask = []       # indices of faces that pass (high + low quality)
        face_quality = {}       # idx → {"tier": "high"|"low", "weight": float, "blur": float}
        blur_rej = 0
        size_rej = 0
        blur_scores = []        # for diagnostics
        face_sizes = []         # for diagnostics
        for i, box in enumerate(boxes):
            w_face = box[2] - box[0]
            h_face = box[3] - box[1]
            face_sizes.append(float(min(w_face, h_face)))
            if not _is_face_large_enough(box):
                size_rej += 1
                prof.count("skipped_by_size")
                continue
            # Compute blur score
            x1, y1, x2, y2 = [int(c) for c in box]
            fh, fw = frame_rgb.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(fw, x2), min(fh, y2)
            if x2c > x1c and y2c > y1c:
                crop = frame_rgb[y1c:y2c, x1c:x2c]
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                blur_var = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
            else:
                blur_var = 0.0
            blur_scores.append(blur_var)

            # Multi-level blur decision
            if blur_var < BLUR_HARD_REJECT:
                # Completely unusable
                blur_rej += 1
                prof.count("skipped_by_blur")
                logger.debug(f"  Face {i}: HARD REJECT (blur={blur_var:.1f} < {BLUR_HARD_REJECT})")
                continue
            elif blur_var < BLUR_LOW_QUALITY:
                # Low quality — allow but with reduced vote weight
                face_quality[i] = {"tier": "low", "weight": VOTE_WEIGHT_LOW, "blur": blur_var}
                prof.count("faces_low_quality")
            else:
                # High quality
                face_quality[i] = {"tier": "high", "weight": VOTE_WEIGHT_HIGH, "blur": blur_var}
                prof.count("faces_high_quality")
            quality_mask.append(i)

        # Track detection statistics
        prof.count("total_faces_raw", len(boxes))
        prof.count("total_faces_passed", len(quality_mask))
        prof.count("total_blur_rejected", blur_rej)
        prof.count("total_size_rejected", size_rej)

        good_boxes_for_log = boxes[quality_mask] if quality_mask else []
        dbg.log_detection(frame_idx, boxes, probs, good_boxes_for_log, blur_rej, size_rej)
        if run_full_detection:
            dbg.save_detection_frame(detect_frame if ds == 1.0 else frame_rgb, frame_idx, boxes, probs)

        # ── Debug frame export with annotations ──
        if SAVE_DEBUG_FRAMES and processed % DEBUG_FRAME_SAMPLE_RATE == 0:
            _save_debug_frame(frame_rgb, frame_idx, boxes, quality_mask,
                              blur_scores, face_sizes, probs)

        if not quality_mask:
            prof.count("frames_no_quality_faces")
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, total_frames,
                    f"Processing frame {frame_idx}/{total_frames}…", len(confirmed_students))
            prof.frame_done()
            continue

        good_boxes = boxes[quality_mask]
        # Map each good_box to its quality weight (from multi-level blur scoring)
        box_quality_weights = [face_quality[qi]["weight"] for qi in quality_mask]
        video_stats["faces_detected_total"] += len(good_boxes)

        for fi, box in enumerate(good_boxes):
            dbg.save_crop(frame_rgb, box, frame_idx, fi)

        logger.debug(f"Frame {frame_idx}: {len(good_boxes)} quality faces detected")

        # ── Tracking ──
        tracked = tracker.update(good_boxes, frame_idx)

        # Map tracked boxes to quality weights
        # tracked[i] corresponds to good_boxes ordering, build box→weight lookup
        track_quality_weights = {}
        for t_idx, (track_id, box, _) in enumerate(tracked):
            # Find which good_box this track matched to (by centroid proximity)
            best_match_idx = 0
            if len(good_boxes) > 1:
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                dists = [abs((gb[0]+gb[2])/2-cx) + abs((gb[1]+gb[3])/2-cy) for gb in good_boxes]
                best_match_idx = int(np.argmin(dists))
            track_quality_weights[track_id] = box_quality_weights[min(best_match_idx, len(box_quality_weights)-1)]

        # ── Batch embedding: collect faces that need extraction ──
        faces_to_embed = []  # (track_id, box)
        cached_results = []  # (track_id, box, embedding, weight)
        for track_id, box, needs_recognition in tracked:
            dbg.log_track(frame_idx, track_id, needs_recognition,
                          reason="cooldown" if not needs_recognition else "")
            if not needs_recognition:
                continue
            weight = track_quality_weights.get(track_id, VOTE_WEIGHT_HIGH)
            track = tracker.get_track(track_id)
            if track and not _should_refresh_embedding(track, box, frame_idx, mode_cfg):
                cached_results.append((track_id, box, track["cached_embedding"], weight))
                prof.count("emb_cache_hits")
            else:
                faces_to_embed.append((track_id, box, weight))
                prof.count("emb_cache_misses")

        # Batch extract new embeddings
        t_emb = time.perf_counter()
        new_embeddings = []
        if faces_to_embed:
            embed_boxes = [item[1] for item in faces_to_embed]
            new_embeddings = _batch_extract_embeddings(frame_rgb, embed_boxes, resnet, device,
                                                       max_threads=prep_threads)
        prof.tick("embedding", (time.perf_counter() - t_emb) * 1000)

        # Combine cached + fresh embeddings into recognition list
        # Each entry: (track_id, box, embedding, vote_weight)
        recognition_list = []
        for (tid, bx, wt), emb in zip(faces_to_embed, new_embeddings):
            if emb is not None:
                tracker.cache_embedding(tid, emb, frame_idx, bx)
                recognition_list.append((tid, bx, emb, wt))
        recognition_list.extend(cached_results)

        # Rebuild active embedding matrix if needed
        if _active_dirty and confirmed_students:
            active_ids = [sid for sid in student_ids if sid not in confirmed_students]
            if active_ids:
                idxs = [student_ids.index(sid) for sid in active_ids]
                active_embeddings = student_embeddings[idxs]
            _active_dirty = False

        # ── Fully vectorized similarity: all faces × all active students ──
        if not recognition_list:
            prof.frame_done()
            continue

        # Build detected embedding matrix (F×512) and compute (F×S) similarity
        det_embs = np.array([emb for _, _, emb, _ in recognition_list])  # (F, 512)
        if len(active_ids) > 0:
            sim_matrix = det_embs @ active_embeddings.T  # (F, S_active)
            best_indices = np.argmax(sim_matrix, axis=1)
            best_sims = sim_matrix[np.arange(len(recognition_list)), best_indices]
            best_sids = [active_ids[i] for i in best_indices]
            use_active = True
        else:
            sim_matrix = det_embs @ student_embeddings.T  # (F, S_all)
            best_indices = np.argmax(sim_matrix, axis=1)
            best_sims = sim_matrix[np.arange(len(recognition_list)), best_indices]
            best_sids = [student_ids[i] for i in best_indices]
            use_active = False

        for face_i, (track_id, box, emb, vote_weight) in enumerate(recognition_list):
            video_stats["recognition_attempts"] += 1
            dbg.log_embedding(frame_idx, track_id, emb)

            best_similarity = float(best_sims[face_i])
            best_student_id = best_sids[face_i]

            logger.debug(
                f"  Track {track_id}: best match = student {best_student_id} "
                f"(sim={best_similarity:.3f}, weight={vote_weight:.1f})"
            )

            # Skip students already fully confirmed
            if best_student_id in confirmed_students:
                tracker.assign_student(track_id, best_student_id)
                continue

            # Track frame exposure for this student (regardless of threshold)
            student_frame_counts[best_student_id] = student_frame_counts.get(best_student_id, 0) + 1

            # Check consecutive stability
            last_seen = student_last_seen_frame.get(best_student_id, -1)
            # If seen recently (allow small gaps for detection interval)
            if last_seen > 0 and frame_idx - last_seen <= detection_interval * 3:
                student_consecutive_matches[best_student_id] = student_consecutive_matches.get(best_student_id, 0) + 1
            else:
                student_consecutive_matches[best_student_id] = 1
            student_last_seen_frame[best_student_id] = frame_idx

            consecutive_matches = student_consecutive_matches[best_student_id]

            # Record similarity in history (for temporal smoothing)
            if best_student_id not in student_sim_history:
                student_sim_history[best_student_id] = []
            hist = student_sim_history[best_student_id]
            hist.append(best_similarity)
            if len(hist) > SIM_HISTORY_MAX_LEN:
                del hist[0]

            # Adaptive threshold: relax if student has many votes already
            current_votes = student_votes.get(best_student_id, 0)
            current_weighted = student_weighted_votes.get(best_student_id, 0.0)
            effective_threshold = (
                SIMILARITY_THRESHOLD_RELAXED
                if current_votes >= VOTE_RELAXED_CUTOFF
                else SIMILARITY_THRESHOLD
            )

            accepted_by_threshold = best_similarity >= effective_threshold
            # ── Debug: log similarity ranking + threshold decision ──
            face_sims = sim_matrix[face_i]
            face_best_idx = int(best_indices[face_i])
            ids_for_log = active_ids if use_active else student_ids
            dbg.log_similarity(
                frame_idx, track_id, face_sims, ids_for_log, student_lookup,
                face_best_idx, best_similarity, effective_threshold, accepted_by_threshold
            )

            # ── Borderline candidate buffer (sub-threshold tracking) ──
            if not accepted_by_threshold and best_similarity >= BORDERLINE_SIM_FLOOR:
                prof.count("borderline_observations")
                if best_student_id not in borderline_buffer:
                    borderline_buffer[best_student_id] = {
                        "appearances": 0, "sims": [], "student_counts": {}
                    }
                buf = borderline_buffer[best_student_id]
                buf["appearances"] += 1
                buf["sims"].append(best_similarity)
                # Track which student ID this face matched to
                buf["student_counts"][best_student_id] = buf["student_counts"].get(best_student_id, 0) + 1

                # Check borderline promotion criteria
                if (buf["appearances"] >= BORDERLINE_MIN_APPEARANCES
                    and len(buf["sims"]) >= BORDERLINE_MIN_APPEARANCES):
                    avg_sim = sum(buf["sims"]) / len(buf["sims"])
                    # Consistency: what fraction of appearances matched this student
                    top_count = max(buf["student_counts"].values())
                    consistency = top_count / buf["appearances"]
                    if (avg_sim >= BORDERLINE_MIN_AVG_SIM
                        and consistency >= BORDERLINE_CONSISTENCY
                        and consecutive_matches >= MIN_CONSECUTIVE_MATCHES):
                        # Promote: treat as accepted
                        accepted_by_threshold = True
                        borderline_promoted += 1
                        prof.count("borderline_promoted")
                        logger.info(
                            f"  ↑ BORDERLINE PROMOTED student {best_student_id} "
                            f"({student_lookup[best_student_id]['name']}) — "
                            f"appearances={buf['appearances']}, avg_sim={avg_sim:.3f}, "
                            f"consistency={consistency:.2f}"
                        )

            # Update best similarity and cast weighted vote
            if accepted_by_threshold:
                if best_student_id not in student_best_similarity or best_similarity > student_best_similarity[best_student_id]:
                    student_best_similarity[best_student_id] = best_similarity

                # Increment both integer vote and weighted vote
                student_votes[best_student_id] = current_votes + 1
                student_weighted_votes[best_student_id] = current_weighted + vote_weight
                tracker.assign_student(track_id, best_student_id)

                votes = student_votes[best_student_id]
                weighted_votes = student_weighted_votes[best_student_id]
                frames_seen = student_frame_counts[best_student_id]
                vote_ratio = votes / frames_seen if frames_seen > 0 else 0.0

                logger.debug(
                    f"  Student {best_student_id}: vote {votes}/{MIN_VOTES} "
                    f"(weighted={weighted_votes:.1f}, ratio={vote_ratio:.2f}, "
                    f"best={student_best_similarity[best_student_id]:.3f}, "
                    f"threshold={effective_threshold:.2f})"
                )

                # Compute smoothed similarity (average of top-K)
                sim_hist = student_sim_history[best_student_id]
                top_k = sorted(sim_hist, reverse=True)[:SIM_HISTORY_TOP_K]
                smoothed_sim = sum(top_k) / len(top_k)

                # High-confidence fast path
                is_high_conf = (
                    best_similarity >= HIGH_CONF_THRESHOLD
                    and smoothed_sim >= HIGH_CONF_THRESHOLD - 0.05
                )
                votes_required = HIGH_CONF_MIN_VOTES if is_high_conf else MIN_VOTES

                # Confirmation: strict enforcement of rules
                would_confirm = True
                reject_reason = ""
                if votes < votes_required:
                    would_confirm = False
                    reject_reason = f"need {votes_required} votes got {votes}"
                elif vote_ratio < MIN_VOTE_RATIO:
                    would_confirm = False
                    reject_reason = f"ratio {vote_ratio:.2f} < {MIN_VOTE_RATIO}"
                    confirmations_below_ratio_threshold += 1
                elif smoothed_sim < effective_threshold:
                    would_confirm = False
                    reject_reason = f"smoothed {smoothed_sim:.4f} < {effective_threshold:.4f}"
                elif smoothed_sim < MIN_FINAL_SMOOTHED_SIM:
                    would_confirm = False
                    reject_reason = f"smoothed {smoothed_sim:.4f} < floor {MIN_FINAL_SMOOTHED_SIM:.2f}"
                    confirmations_below_0_60 += 1
                elif consecutive_matches < MIN_CONSECUTIVE_MATCHES:
                    would_confirm = False
                    reject_reason = f"consecutive {consecutive_matches} < {MIN_CONSECUTIVE_MATCHES}"
                    consecutive_match_failures += 1
                dbg.log_vote(
                    best_student_id, student_lookup[best_student_id]["name"],
                    votes, frames_seen, student_best_similarity.get(best_student_id, 0.0),
                    smoothed_sim, effective_threshold, would_confirm, reject_reason
                )

                # Check confirmation
                if would_confirm:
                    confirmed_students.add(best_student_id)
                    tracker.confirm_track(track_id)
                    _active_dirty = True
                    logger.info(
                        f"  ✓ CONFIRMED student {best_student_id} "
                        f"({student_lookup[best_student_id]['name']}) — "
                        f"votes={votes}, weighted={weighted_votes:.1f}, "
                        f"ratio={vote_ratio:.2f}, smoothed_sim={smoothed_sim:.3f}"
                    )

        prof.tick("frame_total", (time.perf_counter() - t_frame_start) * 1000)
        prof.frame_done()

        # Progress callback
        if progress_callback and (frame_idx % 10 == 0 or frame_idx == total_frames):
            progress_callback(
                frame_idx, total_frames,
                f"Processing frame {frame_idx}/{total_frames} — "
                f"{len(confirmed_students)} student(s) confirmed",
                len(confirmed_students),
            )

        # Early stop: all valid students confirmed
        if len(confirmed_students) == len(valid_students):
            logger.info("All students confirmed — stopping early.")
            break

    vid_reader.release()
    perf_stats = prof.final_summary()

    # ── Debug: write final report + export zip ──
    dbg.write_final_report(
        n_registered=len(valid_students),
        n_confirmed=len(confirmed_students),
        n_processed_frames=processed,
        total_frames=total_frames,
        student_votes=student_votes,
        student_best_sim=student_best_similarity,
        student_frame_counts=student_frame_counts,
        student_lookup=student_lookup,
        confirmed_students=confirmed_students,
    )
    dbg.close()

    # ══════════════════════════════════════════════════════════════════════
    # Final decision — only students with BOTH enough votes AND high
    # similarity are marked present.
    # ══════════════════════════════════════════════════════════════════════
    matched_students = []
    for sid in confirmed_students:
        s = student_lookup[sid]
        matched_students.append({
            "id": sid,
            "roll_no": s["roll_no"],
            "name": s["name"],
            "similarity": student_best_similarity.get(sid, 0.0),
        })

    # Log summary
    logger.info("═" * 60)
    logger.info("VIDEO PROCESSING COMPLETE")
    logger.info(f"  Frames processed : {processed}/{total_frames}")
    logger.info(f"  Faces detected   : {video_stats['faces_detected_total']}")
    logger.info(f"  Recognition runs : {video_stats['recognition_attempts']}")
    logger.info(f"  Students confirmed: {len(confirmed_students)}/{len(valid_students)} (excl. {n_excluded} unk)")
    if student_votes:
        logger.info("  Vote breakdown:")
        for sid, votes in sorted(student_votes.items(), key=lambda x: -x[1]):
            name = student_lookup[sid]["name"]
            sim = student_best_similarity.get(sid, 0.0)
            wv = student_weighted_votes.get(sid, 0.0)
            frames = student_frame_counts.get(sid, 0)
            ratio = votes / frames if frames > 0 else 0.0
            hist = student_sim_history.get(sid, [])
            top_k = sorted(hist, reverse=True)[:SIM_HISTORY_TOP_K]
            smoothed = sum(top_k) / len(top_k) if top_k else 0.0
            status = "✓" if sid in confirmed_students else "✗"
            logger.info(
                f"    {status} {name}: votes={votes}, weighted={wv:.1f}, frames={frames}, "
                f"ratio={ratio:.2f}, best={sim:.3f}, smoothed={smoothed:.3f}"
            )
    logger.info("═" * 60)

    # ── Detection & Quality Pipeline Diagnostics ──
    logger.info("═" * 60)
    logger.info("DETECTION PIPELINE DIAGNOSTICS")
    logger.info(f"  Total frames         : {total_frames}")
    logger.info(f"  Frames processed     : {processed}")
    logger.info(f"  Frames skipped (sched): {total_frames - processed}")
    c = perf_stats
    logger.info(f"  Frames skipped motion: {c.get('frames_skipped_motion', 0)}")
    logger.info(f"  Full MTCNN detections: {c.get('full_detections', 0)}")
    logger.info(f"  Detection reuses     : {c.get('detection_reuses', 0)}")
    logger.info(f"  Frames w/o quality   : {c.get('frames_no_quality_faces', 0)}")
    total_raw = c.get('total_faces_raw', 0)
    total_pass = c.get('total_faces_passed', 0)
    total_blur = c.get('total_blur_rejected', 0)
    total_size = c.get('total_size_rejected', 0)
    hi_q = c.get('faces_high_quality', 0)
    lo_q = c.get('faces_low_quality', 0)
    logger.info(f"  Total faces raw      : {total_raw}")
    logger.info(f"  Total faces passed   : {total_pass}")
    logger.info(f"    ├ High quality      : {hi_q}")
    logger.info(f"    └ Low quality       : {lo_q}")
    logger.info(f"  Blur rejected (hard) : {total_blur} ({total_blur/max(total_raw,1)*100:.1f}%)")
    logger.info(f"  Size rejected        : {total_size} ({total_size/max(total_raw,1)*100:.1f}%)")
    if total_pass > 0 and c.get('full_detections', 0) > 0:
        logger.info(f"  Avg faces/detection  : {total_raw / c['full_detections']:.1f}")
    # Borderline buffer stats
    logger.info(f"  Borderline observed  : {c.get('borderline_observations', 0)}")
    logger.info(f"  Borderline promoted  : {borderline_promoted}")
    if borderline_buffer:
        logger.info(f"  Borderline candidates: {len(borderline_buffer)}")
        logger.info(f"  Borderline rejected  : {len(borderline_buffer) - borderline_promoted}")
        for sid, buf in sorted(borderline_buffer.items(), key=lambda x: -x[1]['appearances']):
            avg_s = sum(buf["sims"])/len(buf["sims"]) if buf["sims"] else 0
            name = student_lookup.get(sid, {}).get("name", f"id={sid}")
            promoted = "PROMOTED" if sid in confirmed_students else "not promoted"
            logger.info(f"    {name}: appearances={buf['appearances']}, avg_sim={avg_s:.3f} ({promoted})")
    # Similarity distribution
    all_sims = []
    for hist in student_sim_history.values():
        all_sims.extend(hist)
    if all_sims:
        arr = np.array(all_sims)
        logger.info(f"  Similarity dist      : min={arr.min():.3f} median={np.median(arr):.3f} "
                     f"max={arr.max():.3f} mean={arr.mean():.3f}")
    logger.info(f"  Embedding cache hits : {c.get('emb_cache_hits', 0)}")
    logger.info(f"  Embedding cache miss : {c.get('emb_cache_misses', 0)}")
    logger.info(f"  Cache hit ratio      : {c.get('cache_hit_ratio', 0):.1f}%")
    # Weighted vote summary
    if student_weighted_votes:
        logger.info("  Weighted votes per student:")
        for sid, wv in sorted(student_weighted_votes.items(), key=lambda x: -x[1]):
            name = student_lookup.get(sid, {}).get("name", f"id={sid}")
            int_v = student_votes.get(sid, 0)
            status = "✓" if sid in confirmed_students else "✗"
            logger.info(f"    {status} {name}: integer={int_v}, weighted={wv:.1f}")
    
    # ── Final Precision Diagnostics ──
    logger.info("═" * 60)
    logger.info("FINAL PRECISION DIAGNOSTICS")
    logger.info(f"  Promoted candidates count         : {borderline_promoted}")
    logger.info(f"  Rejected borderline candidates    : {len(borderline_buffer) - borderline_promoted}")
    logger.info(f"  Confirmations blocked (<0.60)     : {confirmations_below_0_60}")
    logger.info(f"  Confirmations blocked (<ratio)    : {confirmations_below_ratio_threshold}")
    logger.info(f"  Consecutive match failures        : {consecutive_match_failures}")
    logger.info("═" * 60)

    video_stats["confirmed_students"] = len(confirmed_students)
    video_stats["performance"] = perf_stats

    return matched_students, video_stats


# ══════════════════════════════════════════════════════════════════════════════
# Temp file helper for Streamlit UploadedFile
# ══════════════════════════════════════════════════════════════════════════════

def save_uploaded_video(uploaded_file) -> str:
    """
    Save a Streamlit UploadedFile to a temporary file and return its path.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Absolute path to the saved temp file.
    """
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="attendance_video_") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name
