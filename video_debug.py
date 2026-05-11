"""
video_debug.py — Deep Diagnostics for Video Attendance Pipeline
===============================================================
Import this module in video_attendance.py to gain full visibility into
every stage of the pipeline without touching the core logic.

Usage inside video_attendance.py:
    from video_debug import DebugSession
    dbg = DebugSession(enabled=DEBUG_MODE)
"""

import os
import json
import shutil
import logging
import zipfile
import textwrap
from datetime import datetime
from typing import Optional, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration (edit here or override after import)
# ──────────────────────────────────────────────────────────────────────────────
DEBUG_MODE = True            # Master switch
DEBUG_SAVE_FRAMES = True     # Save raw frames to disk
DEBUG_MAX_SAVED_FRAMES = 20  # Max number of raw / crop frames saved
DEBUG_DIR = "debug"          # Root output folder
SIMILARITY_THRESHOLD_DEBUG = 0.60   # Relaxed threshold used ONLY for logging; actual decisions still use original


class DebugSession:
    """
    One DebugSession instance per video-processing run.
    All state is isolated — safe for concurrent Streamlit runs.
    """

    def __init__(self, enabled: bool = DEBUG_MODE, run_tag: str = ""):
        self.enabled = enabled
        if not enabled:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tag = run_tag or ts
        self.root = os.path.join(DEBUG_DIR, f"run_{self.tag}")

        # Sub-folders
        self.dir_frames   = os.path.join(self.root, "video_frames")
        self.dir_detected = os.path.join(self.root, "detected_faces")
        self.dir_crops    = os.path.join(self.root, "face_crops")
        self.sim_log_path = os.path.join(self.root, "similarity_logs.txt")
        self.report_path  = os.path.join(self.root, "final_report.txt")
        self.zip_path     = os.path.join(DEBUG_DIR, f"debug_report_{self.tag}.zip")

        for d in (self.dir_frames, self.dir_detected, self.dir_crops):
            os.makedirs(d, exist_ok=True)

        self._sim_log_fh = open(self.sim_log_path, "w", encoding="utf-8")

        # Counters / accumulators
        self.saved_raw_frames   = 0
        self.saved_crops        = 0
        self.total_faces_det    = 0
        self.total_emb_ok       = 0
        self.total_emb_fail     = 0
        self.reject_threshold   = 0
        self.reject_votes       = 0
        self.reject_blur        = 0
        self.reject_size        = 0
        self.reject_no_det      = 0
        self.all_similarities: List[float] = []

        # Per-student voting log  {student_name: {votes, best_sim, decisions:[]}}
        self._student_log = {}

        logger.info("=" * 70)
        logger.info(f"[DEBUG] Session started → {self.root}")
        logger.info("=" * 70)

    # ── 1. Video input validation ──────────────────────────────────────────
    def log_video_info(self, cap: cv2.VideoCapture, path: str):
        if not self.enabled:
            return
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc  = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec   = "".join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)])
        dur_sec = total / fps if fps > 0 else 0

        logger.info("[DEBUG] ── VIDEO INPUT ──────────────────────────────")
        logger.info(f"[DEBUG] File       : {path}")
        logger.info(f"[DEBUG] Resolution : {width}x{height}")
        logger.info(f"[DEBUG] FPS        : {fps:.2f}")
        logger.info(f"[DEBUG] Frames     : {total}")
        logger.info(f"[DEBUG] Duration   : {dur_sec:.1f} sec")
        logger.info(f"[DEBUG] Codec      : {codec.strip()}")
        logger.info("[DEBUG] Video loaded successfully ✓")

    # ── 2. Raw frame save ──────────────────────────────────────────────────
    def save_raw_frame(self, frame_rgb: np.ndarray, frame_idx: int):
        if not self.enabled or not DEBUG_SAVE_FRAMES:
            return
        if self.saved_raw_frames >= DEBUG_MAX_SAVED_FRAMES:
            return
        path = os.path.join(self.dir_frames, f"frame_{frame_idx:04d}.jpg")
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
        self.saved_raw_frames += 1

    # ── 3. Face detection log ──────────────────────────────────────────────
    def log_detection(self, frame_idx: int, boxes, probs,
                       good_boxes, blur_rejects: int, size_rejects: int):
        if not self.enabled:
            return

        n_raw  = len(boxes) if boxes is not None else 0
        n_good = len(good_boxes) if good_boxes is not None else 0

        self.reject_blur += blur_rejects
        self.reject_size += size_rejects
        self.total_faces_det += n_good

        if n_raw == 0:
            self.reject_no_det += 1
            logger.info(f"[DEBUG] Frame {frame_idx:4d}: NO FACES detected")
        else:
            conf_str = ""
            if probs is not None:
                conf_str = ", ".join(f"{p:.2f}" for p in probs[:5])
            logger.info(
                f"[DEBUG] Frame {frame_idx:4d}: raw={n_raw}  quality_pass={n_good}  "
                f"blur_rej={blur_rejects}  size_rej={size_rejects}  "
                f"confs=[{conf_str}]"
            )

    # ── 3b. Save annotated detection frame ────────────────────────────────
    def save_detection_frame(self, frame_rgb: np.ndarray, frame_idx: int,
                              boxes, probs):
        if not self.enabled or not DEBUG_SAVE_FRAMES:
            return
        if self.saved_raw_frames > DEBUG_MAX_SAVED_FRAMES * 2:
            return
        if boxes is None or len(boxes) == 0:
            return
        vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(c) for c in box]
            conf = f"{probs[i]:.2f}" if probs is not None else "?"
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, conf, (x1, max(y1-5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        path = os.path.join(self.dir_detected, f"det_{frame_idx:04d}.jpg")
        cv2.imwrite(path, vis)

    # ── 4. Crop save ───────────────────────────────────────────────────────
    def save_crop(self, frame_rgb: np.ndarray, box: np.ndarray,
                  frame_idx: int, face_idx: int):
        if not self.enabled or not DEBUG_SAVE_FRAMES:
            return
        if self.saved_crops >= DEBUG_MAX_SAVED_FRAMES:
            return
        x1, y1, x2, y2 = [int(c) for c in box]
        h, w = frame_rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return
        path = os.path.join(self.dir_crops,
                             f"frame{frame_idx:04d}_face{face_idx}.jpg")
        cv2.imwrite(path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        self.saved_crops += 1

    # ── 5. Embedding log ───────────────────────────────────────────────────
    def log_embedding(self, frame_idx: int, face_idx: int,
                       emb: Optional[np.ndarray], error: str = ""):
        if not self.enabled:
            return
        if emb is not None:
            norm = float(np.linalg.norm(emb))
            self.total_emb_ok += 1
            logger.debug(
                f"[DEBUG] Frame {frame_idx} face {face_idx}: "
                f"embedding OK  norm={norm:.4f}"
            )
        else:
            self.total_emb_fail += 1
            logger.warning(
                f"[DEBUG] Frame {frame_idx} face {face_idx}: "
                f"EMBEDDING FAILED  reason={error or 'unknown'}"
            )

    # ── 6 + 7. Similarity + threshold decision log ─────────────────────────
    def log_similarity(self, frame_idx: int, track_id: int,
                        similarities: np.ndarray,
                        student_ids: list, student_lookup: dict,
                        best_idx: int, best_sim: float,
                        threshold: float, accepted: bool):
        if not self.enabled:
            return

        self.all_similarities.append(best_sim)

        # Top-5 matches
        top5_idx = np.argsort(similarities)[::-1][:5]
        lines = [
            f"\n[DEBUG] ── Similarity | Frame {frame_idx} Track {track_id} ──"
        ]
        for rank, idx in enumerate(top5_idx, 1):
            sid   = student_ids[idx]
            name  = student_lookup[sid]["name"]
            sim   = float(similarities[idx])
            marker = "★" if idx == best_idx else " "
            lines.append(f"  {marker} {rank}. {name:<20s} → {sim:.4f}")

        decision = "ACCEPTED ✓" if accepted else "REJECTED ✗"
        reason   = "" if accepted else (
            f"sim {best_sim:.4f} < threshold {threshold:.4f}"
        )
        lines.append(
            f"  Threshold={threshold:.4f}  Best={best_sim:.4f}  "
            f"Decision={decision}  {reason}"
        )

        msg = "\n".join(lines)
        logger.info(msg)
        self._sim_log_fh.write(msg + "\n")
        self._sim_log_fh.flush()

        # Also log at relaxed threshold for diagnostics
        if not accepted and best_sim >= SIMILARITY_THRESHOLD_DEBUG:
            logger.warning(
                f"[DEBUG] ⚠ Would ACCEPT at relaxed threshold "
                f"{SIMILARITY_THRESHOLD_DEBUG} (sim={best_sim:.4f}) → "
                f"threshold may be too strict!"
            )
            self._sim_log_fh.write(
                f"  ⚠ Would accept at relaxed={SIMILARITY_THRESHOLD_DEBUG}\n"
            )

        if not accepted:
            self.reject_threshold += 1

    # ── 8. Tracking log ────────────────────────────────────────────────────
    def log_track(self, frame_idx: int, track_id: int,
                   needs_recognition: bool, reason: str = ""):
        if not self.enabled:
            return
        status = "NEEDS_RECOG" if needs_recognition else f"COOLDOWN ({reason})"
        logger.debug(
            f"[DEBUG] Frame {frame_idx} Track {track_id}: {status}"
        )

    # ── 9. Voting log ──────────────────────────────────────────────────────
    def log_vote(self, student_id: int, student_name: str,
                  votes: int, frames_seen: int, best_sim: float,
                  smoothed_sim: float, threshold: float,
                  confirmed: bool, reject_reason: str = ""):
        if not self.enabled:
            return

        vote_ratio = votes / frames_seen if frames_seen > 0 else 0.0
        decision   = "ACCEPTED ✓" if confirmed else f"REJECTED ✗ ({reject_reason})"

        logger.info(
            f"[DEBUG] Voting | {student_name}: "
            f"votes={votes}  frames={frames_seen}  ratio={vote_ratio:.2f}  "
            f"best_sim={best_sim:.4f}  smoothed={smoothed_sim:.4f}  "
            f"threshold={threshold:.4f}  → {decision}"
        )

        if not confirmed:
            self.reject_votes += 1

        # Update per-student log
        entry = self._student_log.setdefault(student_name, {
            "votes": 0, "best_sim": 0.0, "confirmed": False
        })
        entry["votes"]     = votes
        entry["best_sim"]  = max(entry["best_sim"], best_sim)
        entry["confirmed"] = confirmed

    # ── 10. Final report ───────────────────────────────────────────────────
    def write_final_report(self, n_registered: int, n_confirmed: int,
                            n_processed_frames: int, total_frames: int,
                            student_votes: dict, student_best_sim: dict,
                            student_frame_counts: dict, student_lookup: dict,
                            confirmed_students: set):
        if not self.enabled:
            return

        self._sim_log_fh.close()

        avg_sim = (sum(self.all_similarities) / len(self.all_similarities)
                   if self.all_similarities else 0.0)

        lines = [
            "=" * 70,
            "VIDEO ATTENDANCE DIAGNOSTICS — FINAL REPORT",
            f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
            "── VIDEO STATS ──────────────────────────────────────────────────",
            f"  Frames total     : {total_frames}",
            f"  Frames processed : {n_processed_frames}",
            "",
            "── DETECTION STATS ──────────────────────────────────────────────",
            f"  Frames with NO detection : {self.reject_no_det}",
            f"  Faces passed quality     : {self.total_faces_det}",
            f"  Rejected (too small)     : {self.reject_size}",
            f"  Rejected (blur)          : {self.reject_blur}",
            "",
            "── EMBEDDING STATS ──────────────────────────────────────────────",
            f"  Embeddings OK    : {self.total_emb_ok}",
            f"  Embeddings FAIL  : {self.total_emb_fail}",
            "",
            "── SIMILARITY STATS ─────────────────────────────────────────────",
            f"  Average similarity       : {avg_sim:.4f}",
            f"  Rejected (below thresh)  : {self.reject_threshold}",
            f"  Relaxed threshold used   : {SIMILARITY_THRESHOLD_DEBUG}",
            "",
            "── STUDENT RESULTS ──────────────────────────────────────────────",
            f"  Registered : {n_registered}",
            f"  Confirmed  : {n_confirmed}",
            "",
        ]

        for sid, s in student_lookup.items():
            name    = s["name"]
            votes   = student_votes.get(sid, 0)
            frames  = student_frame_counts.get(sid, 0)
            bsim    = student_best_sim.get(sid, 0.0)
            status  = "PRESENT ✓" if sid in confirmed_students else "ABSENT  ✗"
            ratio   = votes / frames if frames > 0 else 0.0
            lines.append(
                f"  {status} | {name:<25s} | votes={votes:3d}  "
                f"frames={frames:4d}  ratio={ratio:.2f}  best_sim={bsim:.4f}"
            )

        lines += [
            "",
            "── FAILURE ANALYSIS ─────────────────────────────────────────────",
        ]
        if self.reject_no_det > n_processed_frames * 0.5:
            lines.append("  ⚠ >50% frames had NO faces detected → "
                         "check video quality / MTCNN min_face_size")
        if avg_sim < 0.65 and self.total_faces_det > 0:
            lines.append("  ⚠ Average similarity very low (<0.65) → "
                         "possible wrong crop / bad embedding pipeline")
        if self.reject_threshold > self.total_emb_ok * 0.5:
            lines.append("  ⚠ >50% embeddings rejected by threshold → "
                         "threshold may be too strict for this video")
        if self.total_emb_fail > self.total_emb_ok:
            lines.append("  ⚠ More embedding failures than successes → "
                         "check _extract_embedding_for_box()")
        if n_confirmed == 0 and n_registered > 0:
            lines.append("  ✗ ZERO students confirmed — pipeline is broken at one of the stages above")

        lines.append("=" * 70)

        report_text = "\n".join(lines)
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # Print to logger too
        for line in lines:
            logger.info(line)

    # ── 11. ZIP export ─────────────────────────────────────────────────────
    def export_zip(self):
        if not self.enabled:
            return
        try:
            with zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for dirpath, _, filenames in os.walk(self.root):
                    for fname in filenames:
                        full = os.path.join(dirpath, fname)
                        arc  = os.path.relpath(full, os.path.dirname(self.root))
                        zf.write(full, arc)
            logger.info(f"[DEBUG] ZIP exported → {self.zip_path}")
        except Exception as e:
            logger.warning(f"[DEBUG] ZIP export failed: {e}")

    def close(self):
        """Call at end of pipeline run."""
        if not self.enabled:
            return
        try:
            if not self._sim_log_fh.closed:
                self._sim_log_fh.close()
        except Exception:
            pass
        self.export_zip()
        logger.info(f"[DEBUG] Session closed. Debug files at: {self.root}")
        logger.info(f"[DEBUG] ZIP report  at: {self.zip_path}")
