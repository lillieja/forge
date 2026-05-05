#!/usr/bin/env python3
"""Compare MediaPipe Tasks ``gesture_recognizer.task`` vs TensorRT triple engines on one image.

Uses the same preprocessing and TRT wiring as ``swiper_opt.py`` (imported at runtime).

Example (repo root, ``.tests`` venv active):

  python scripts/tests/compare_gesture_task_vs_trt.py \\
    --image data/gestures/open/open_0.jpg
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Repo root: scripts/tests -> scripts -> forge
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS_TESTS = Path(__file__).resolve().parent
if str(_SCRIPTS_TESTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_TESTS))


def _load_swiper_opt():
    """Load ``swiper_opt`` as a module (not a package import)."""
    path = Path(__file__).resolve().parent / "swiper_opt.py"
    spec = importlib.util.spec_from_file_location("_swiper_opt_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64) - float(np.max(logits))
    e = np.exp(z)
    return (e / np.sum(e)).astype(np.float32)


def _landmarks_to_numpy(hand_lm) -> np.ndarray:
    """MediaPipe ``hand_landmarks[0]`` list -> (21, 3) float32."""
    out = np.zeros((21, 3), dtype=np.float32)
    for i, lm in enumerate(hand_lm):
        out[i, 0] = float(lm.x)
        out[i, 1] = float(lm.y)
        out[i, 2] = float(lm.z)
    return out


def _run_mediapipe_task(
    rgb_uint8: np.ndarray, task_path: str
) -> tuple[list[tuple[str, float]], np.ndarray | None]:
    """Returns (top gestures per hand as (name, score), first-hand (21,3) normalized landmarks)."""
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_uint8)
        result = recognizer.recognize(mp_image)
    finally:
        recognizer.close()

    summaries: list[tuple[str, float]] = []
    if result.gestures:
        for hand_idx, categories in enumerate(result.gestures):
            if not categories:
                summaries.append((f"<hand{hand_idx}_none>", 0.0))
                continue
            top = max(categories, key=lambda c: float(c.score))
            name = getattr(top, "category_name", None) or getattr(top, "display_name", "") or "?"
            summaries.append((str(name), float(top.score)))

    lms_np: np.ndarray | None = None
    if result.hand_landmarks:
        lms_np = _landmarks_to_numpy(result.hand_landmarks[0])

    return summaries, lms_np


def _collect_trt_63_outputs(lm_outs: list[np.ndarray]) -> list[tuple[int, np.ndarray]]:
    out = []
    for i, arr in enumerate(lm_outs):
        if arr.size == 63:
            out.append((i, arr.reshape(-1).astype(np.float32)))
    return out


def _run_trt_chain(
    so,
    frame_bgr: np.ndarray,
    *,
    palm_engine: Any | None = None,
    rgb_full: np.ndarray | None = None,
):
    """Mirror ``swiper_opt`` main-loop inference; returns a dict of arrays and scores."""
    use_det = palm_engine is not None and rgb_full is not None
    if use_det:
        img_in, had_roi = so._landmark_input_from_frame(
            frame_bgr,
            use_detector=True,
            palm_engine=palm_engine,
            rgb_full=rgb_full,
        )
        if not had_roi or img_in is None:
            return {
                "img_in_layout": "none",
                "error": "no_hand_roi",
                "gesture_id": -1,
                "top_prob": 0.0,
                "probs": np.zeros(8, dtype=np.float32),
                "logits": np.zeros(8, dtype=np.float32),
                "lm_outputs_63": [],
                "lms_used_flat": np.zeros(63, dtype=np.float32),
                "embedding": np.zeros(128, dtype=np.float32),
                "landmark_engine": os.path.basename(so.LANDMARK_ENGINE_PATH),
                "embedder_engine": os.path.basename(so.EMBEDDER_ENGINE_PATH),
                "classifier_engine": os.path.basename(so.CLASSIFIER_ENGINE_PATH),
            }
    elif so.LEGACY_LANDMARK_PREPROCESS:
        img_in = cv2.resize(frame_bgr, so.LM_INPUT_SHAPE).astype(np.float32) / 255.0
        img_in = np.transpose(img_in, (2, 0, 1))
    else:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_in = cv2.resize(rgb, so.LM_INPUT_SHAPE).astype(np.float32) / 255.0
        img_in = np.expand_dims(img_in, axis=0)

    lm = so.TRTEngine(so.LANDMARK_ENGINE_PATH)
    emb = so.TRTEngine(so.EMBEDDER_ENGINE_PATH)
    cls = so.TRTEngine(so.CLASSIFIER_ENGINE_PATH)

    lm_outs = lm.run(img_in)
    cand_63 = _collect_trt_63_outputs(lm_outs)
    _, lms_image, lms_world_flat = so._landmark_image_world_flat(lm_outs)

    lms_norm = np.clip(lms_image.astype(np.float32) / 224.0, 0.0, 1.0)
    hand = lms_norm.reshape(1, 21, 3)

    if so.LEGACY_EMBEDDER_RAW_LMS and len(emb.inputs) == 1:
        emb_outs = emb.run(np.asarray(lms_image, dtype=np.float32))
    elif so.LEGACY_EMBEDDER_RAW_LMS and len(emb.inputs) > 1:
        lms_f = np.asarray(lms_image, dtype=np.float32).reshape(1, 21, 3)
        world_f = (
            np.asarray(lms_world_flat, dtype=np.float32).reshape(1, 21, 3)
            if lms_world_flat is not None
            else lms_f.copy()
        )
        handedness = np.ones((1, 1), dtype=np.float32)
        feed = {}
        for tin in emb.inputs:
            nl = tin["name"].lower()
            if "handedness" in nl:
                feed[tin["name"]] = handedness
            elif "world" in nl:
                feed[tin["name"]] = world_f
            else:
                feed[tin["name"]] = lms_f
        emb_outs = emb.run_feed_dict(feed)
    elif len(emb.inputs) == 1:
        emb_outs = emb.run(lms_norm.reshape(1, 21, 3))
    else:
        handedness = np.ones((1, 1), dtype=np.float32)
        if lms_world_flat is not None:
            world_hand = lms_world_flat.reshape(1, 21, 3)
        else:
            world_hand = hand.copy()
        feed = {}
        for tin in emb.inputs:
            nl = tin["name"].lower()
            if "handedness" in nl:
                feed[tin["name"]] = handedness
            elif "world" in nl:
                feed[tin["name"]] = world_hand
            else:
                feed[tin["name"]] = hand
        emb_outs = emb.run_feed_dict(feed)

    embedding = emb_outs[0].ravel()
    cls_outs = cls.run(embedding.reshape(1, -1))
    logits = cls_outs[0].ravel().astype(np.float32)
    probs = _softmax(logits)
    gid = int(np.argmax(probs))

    return {
        "img_in_layout": "CHW_BGR_legacy" if so.LEGACY_LANDMARK_PREPROCESS else "NHWC_RGB",
        "landmark_engine": os.path.basename(so.LANDMARK_ENGINE_PATH),
        "embedder_engine": os.path.basename(so.EMBEDDER_ENGINE_PATH),
        "classifier_engine": os.path.basename(so.CLASSIFIER_ENGINE_PATH),
        "lm_outputs_63": cand_63,
        "lms_used_flat": lms_image,
        "embedding": embedding,
        "logits": logits,
        "probs": probs,
        "gesture_id": gid,
        "top_prob": float(probs[gid]),
    }


def _rmse_xy(mp_norm: np.ndarray, trt_flat: np.ndarray) -> float:
    """MP landmarks normalized [0,1] vs TRT 63-vector in ~pixel space (first pass uses first head)."""
    trt = trt_flat.reshape(21, 3)
    mp_px = mp_norm[:, :2] * np.array([224.0, 224.0], dtype=np.float32)
    trt_xy = trt[:, :2].astype(np.float32)
    d = mp_px - trt_xy
    return float(np.sqrt(np.mean(d * d)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--image",
        default=str(_REPO_ROOT / "data/gestures/open/open_0.jpg"),
        help="BGR image path (default: data/gestures/open/open_0.jpg)",
    )
    ap.add_argument(
        "--task",
        default=str(_REPO_ROOT / "models/weights/gesture_recognizer.task"),
        help="MediaPipe bundled .task (default: models/weights/gesture_recognizer.task)",
    )
    ap.add_argument(
        "--mirror",
        action="store_true",
        help="Apply cv2.flip(..., 1) like the live webcam path in swiper_opt.",
    )
    ap.add_argument(
        "--with-detector",
        action="store_true",
        help="Use models/weights/hand_detector.engine for palm ROI (like swiper_opt).",
    )
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        print(f"Image not found: {img_path}", file=sys.stderr)
        sys.exit(1)
    task_path = Path(args.task)
    if not task_path.is_file():
        print(f"Task not found: {task_path}", file=sys.stderr)
        sys.exit(1)

    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"Failed to read image: {img_path}", file=sys.stderr)
        sys.exit(1)
    if args.mirror:
        frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print("=== MediaPipe Tasks (gesture_recognizer.task) ===")
    try:
        mp_gestures, mp_lms = _run_mediapipe_task(rgb, str(task_path))
    except ImportError as exc:
        print(
            "mediapipe not installed. Install with:\n"
            "  uv pip install mediapipe --python .tests/bin/python3",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    if mp_gestures:
        for name, sc in mp_gestures:
            print(f"  gesture: {name}  score={sc:.4f}")
    else:
        print("  (no gestures returned — hand may be undetected in Tasks pipeline)")
    if mp_lms is not None:
        print(
            f"  wrist (normalized): x={mp_lms[0,0]:.4f} y={mp_lms[0,1]:.4f} z={mp_lms[0,2]:.6f}"
        )
    else:
        print("  (no hand_landmarks)")

    print()
    print("=== TensorRT (same preprocess as swiper_opt) ===")
    so = _load_swiper_opt()
    palm_engine = None
    rgb_full = rgb.copy()
    det_path = _REPO_ROOT / "models/weights" / "hand_detector.engine"
    if args.with_detector:
        if det_path.is_file():
            palm_engine = so.TRTEngine(str(det_path))
            print(f"  palm detector: {det_path.name}")
        else:
            print(f"  (--with-detector but missing {det_path}; using full-frame TRT)")
    print(
        f"  LEGACY_LANDMARK_PREPROCESS={so.LEGACY_LANDMARK_PREPROCESS} "
        f"LEGACY_EMBEDDER_RAW_LMS={so.LEGACY_EMBEDDER_RAW_LMS}"
    )
    trt = _run_trt_chain(
        so,
        frame,
        palm_engine=palm_engine,
        rgb_full=rgb_full if palm_engine is not None else None,
    )
    if trt.get("error") == "no_hand_roi":
        print("  TRT: no palm ROI (detector found no hand above threshold).")
    print(f"  engines: {trt['landmark_engine']}, {trt['embedder_engine']}, {trt['classifier_engine']}")
    print(f"  landmarker input layout: {trt['img_in_layout']}")
    print(f"  landmarker (1,63) outputs: {len(trt['lm_outputs_63'])} candidate(s)")
    for idx, vec in trt["lm_outputs_63"]:
        tag = "first_match" if np.allclose(vec, trt["lms_used_flat"]) else "alternate"
        print(
            f"    [{idx}] vec[0:3]={vec[:3]}  min={vec.min():.4f} max={vec.max():.4f}  ({tag})"
        )
    print(
        f"  classifier: argmax class id={trt['gesture_id']}  "
        f"prob={trt['top_prob']:.4f}  (Open Palm id in swiper_opt={so.OPEN_PALM_ID})"
    )
    top3 = np.argsort(-trt["probs"])[: min(3, trt["probs"].size)]
    print("  top-3 class ids:", [int(i) for i in top3], "probs:", [float(trt["probs"][i]) for i in top3])

    print()
    print("=== Landmark agreement (heuristic) ===")
    if mp_lms is not None and trt["lm_outputs_63"] and trt.get("error") != "no_hand_roi":
        for idx, vec in trt["lm_outputs_63"]:
            r = _rmse_xy(mp_lms, vec)
            print(f"  RMSE(MP_xy*224 vs TRT output[{idx}] xy): {r:.2f} px")
        print(
            "  (Low RMSE on one head suggests that head matches Tasks image landmarks; "
            "large RMSE often means ROI/preprocess mismatch.)"
        )

    print()
    print("=== Interpretation ===")
    print(
        "  If Tasks gesture looks correct but TRT argmax is wrong, inspect embedder inputs "
        "(world vs image landmarks) and whether swiper_opt should use the other (1,63) tensor."
    )


if __name__ == "__main__":
    main()
