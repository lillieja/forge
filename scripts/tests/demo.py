"""
Ultimate demo pipeline:
- Open_Palm => desktop swipe right
- Closed_Fist => desktop swipe left
- Victory => servo press pulse (0.15s then rest)

This keeps the TensorRT GPU pipeline (detector/landmarks/embedder/classifier) and
optionally uses MediaPipe GestureRecognizer as the canonical gesture label source.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import mp_palm


def _setup_import_paths() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))

    jetson_sites: list[str] = []
    for site in (
        "/usr/lib/python3.10/dist-packages",
        "/usr/lib/python3/dist-packages",
    ):
        if os.path.isdir(site):
            jetson_sites.append(site)

    repo_venv_sites: list[str] = []
    for env_dir in (".tests", ".tasks"):
        pattern = os.path.join(repo, env_dir, "lib", "python*", "site-packages")
        for site in sorted(glob.glob(pattern)):
            if os.path.isdir(site):
                repo_venv_sites.append(site)

    prefix_order = repo_venv_sites + jetson_sites
    for p in prefix_order:
        while p in sys.path:
            sys.path.remove(p)
    for p in reversed(prefix_order):
        sys.path.insert(0, p)

    for site in glob.glob(os.path.expanduser("~/.local/lib/python*/site-packages")):
        if os.path.isdir(site) and site not in sys.path:
            sys.path.append(site)


_setup_import_paths()

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt
except (ImportError, SystemError, AttributeError, TypeError) as exc:
    raise ImportError(
        "Could not load PyCUDA/TensorRT. Activate `.tests` and keep numpy<2."
    ) from exc

import cv2
import numpy as np
import pyautogui
import serial

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_WEIGHTS = os.path.join(_PROJECT_ROOT, "models/weights")


def _pick_engine(preferred: str, fallback: str) -> str:
    p = os.path.join(_WEIGHTS, preferred)
    if os.path.isfile(p):
        return p
    f = os.path.join(_WEIGHTS, fallback)
    if os.path.isfile(f):
        return f
    raise FileNotFoundError(
        f"Missing TensorRT engine: need {preferred} or {fallback} under {_WEIGHTS}"
    )


LANDMARK_ENGINE_PATH = _pick_engine("landmarks.engine", "default_landmarker.engine")
EMBEDDER_ENGINE_PATH = _pick_engine("embedder.engine", "default_gestures.engine")
CLASSIFIER_ENGINE_PATH = _pick_engine("classifier.engine", "default_classifier.engine")
HAND_DETECTOR_ENGINE_PATH = os.path.join(_WEIGHTS, "hand_detector.engine")

LEGACY_LANDMARK_PREPROCESS = "default_landmarker" in os.path.basename(LANDMARK_ENGINE_PATH)
LEGACY_EMBEDDER_RAW_LMS = "default_gestures" in os.path.basename(EMBEDDER_ENGINE_PATH)

LM_INPUT_SHAPE = (224, 224)
GESTURE_SCORE_MIN = 0.60
COOLDOWN_SEC = 2.0
MP_INFER_EVERY_N_FRAMES = 4
SERVO_PRESS_DURATION_SEC = 0.15
GEMINI_URL = "https://gemini.google.com"

NONE_ID = 0
CLOSE_ID = 1
ILOVEYOU_ID = 2
LIGHT_ID = 3
OPEN_ID = 4
GESTURE_LABELS = (
    "none",
    "closed",
    "iloveyou",
    "light",
    "open",
)
LABEL_TO_ID = {name.lower(): idx for idx, name in enumerate(GESTURE_LABELS)}


class TRTEngine:
    """TensorRT 10.x async execution with pycuda streams."""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs: list[dict] = []
        self.outputs: list[dict] = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            vol = int(np.prod([s if s > 0 else 1 for s in shape]))
            host_mem = cuda.pagelocked_empty(vol, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            info = {"name": name, "shape": shape, "host": host_mem, "device": device_mem}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(info)
            else:
                self.outputs.append(info)

    def _execute_bound(self) -> list[np.ndarray]:
        for t in self.inputs + self.outputs:
            self.context.set_tensor_address(t["name"], int(t["device"]))
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()
        out_views: list[np.ndarray] = []
        for out in self.outputs:
            shape = out["shape"]
            vol = int(np.prod([s if s > 0 else 1 for s in shape]))
            out_views.append(np.copy(out["host"][:vol].reshape(shape)))
        return out_views

    def run(self, *input_arrays: np.ndarray) -> list[np.ndarray]:
        if len(input_arrays) != len(self.inputs):
            raise ValueError(f"Expected {len(self.inputs)} inputs, got {len(input_arrays)}")
        for arr, tin in zip(input_arrays, self.inputs):
            flat = np.asarray(arr, dtype=np.float32).ravel()
            if flat.size != tin["host"].size:
                raise ValueError(
                    f"Input {tin['name']}: got {flat.size} elements, engine expects {tin['host'].size}"
                )
            np.copyto(tin["host"], flat)
            cuda.memcpy_htod_async(tin["device"], tin["host"], self.stream)
        return self._execute_bound()

    def run_feed_dict(self, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        for tin in self.inputs:
            if tin["name"] not in feed:
                raise KeyError(f"Missing input tensor {tin['name']!r}")
            flat = np.asarray(feed[tin["name"]], dtype=np.float32).ravel()
            if flat.size != tin["host"].size:
                raise ValueError(
                    f"Input {tin['name']}: got {flat.size} elements, engine expects {tin['host'].size}"
                )
            np.copyto(tin["host"], flat)
            cuda.memcpy_htod_async(tin["device"], tin["host"], self.stream)
        return self._execute_bound()


def _landmark_image_world_flat(lm_outs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray | None]:
    idxs = [i for i, o in enumerate(lm_outs) if o.size == 63]
    if not idxs:
        raise RuntimeError("Landmarker produced no flat size-63 output.")
    img = lm_outs[idxs[0]].ravel().astype(np.float32, copy=False)
    world = lm_outs[idxs[1]].ravel().astype(np.float32, copy=False) if len(idxs) >= 2 else None
    return img, world


def _landmark_input_from_frame(
    frame_bgr: np.ndarray,
    *,
    use_detector: bool,
    palm_engine: TRTEngine | None,
    rgb_full: np.ndarray,
) -> tuple[np.ndarray | None, bool]:
    if not use_detector or palm_engine is None:
        if LEGACY_LANDMARK_PREPROCESS:
            img_in = cv2.resize(frame_bgr, LM_INPUT_SHAPE).astype(np.float32) / 255.0
            img_in = np.transpose(img_in, (2, 0, 1))
        else:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_in = cv2.resize(rgb, LM_INPUT_SHAPE).astype(np.float32) / 255.0
            img_in = np.expand_dims(img_in, axis=0)
        return img_in, True

    det_tensor, pad = mp_palm.preprocess_detector_input(rgb_full)
    det_outs = palm_engine.run(det_tensor)
    rect = mp_palm.palm_roi_from_detector_outputs(det_outs, pad)
    if rect is None:
        return None, False

    patch = mp_palm.warp_normalized_roi_to_landmark_input(rgb_full, rect)
    if LEGACY_LANDMARK_PREPROCESS:
        u8 = (patch[0] * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        img_in = np.transpose(bgr.astype(np.float32) / 255.0, (2, 0, 1))
    else:
        img_in = patch
    return img_in, True


def _gesture_name(gesture_id: int) -> str:
    if 0 <= gesture_id < len(GESTURE_LABELS):
        return GESTURE_LABELS[gesture_id]
    return f"Unknown({gesture_id})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified demo: gestures + desktop swipe + servo toggle.")
    ap.add_argument("--no-detector", action="store_true", help="Use full-frame landmarks.")
    ap.add_argument("--serial-port", default="/dev/ttyCH341USB0", help="Servo serial port.")
    ap.add_argument("--baud-rate", type=int, default=9600, help="Servo serial baud.")
    ap.add_argument("--servo-rest", type=int, default=90, help="Servo rest angle.")
    ap.add_argument("--servo-active", type=int, default=50, help="Servo active angle.")
    args = ap.parse_args()

    use_detector = not args.no_detector
    if use_detector and not os.path.isfile(HAND_DETECTOR_ENGINE_PATH):
        raise FileNotFoundError(
            f"Missing {HAND_DETECTOR_ENGINE_PATH}. Build engines first or run with --no-detector."
        )

    servo = None
    servo_enabled = False
    try:
        servo = serial.Serial(args.serial_port, args.baud_rate, timeout=1)
        time.sleep(2.0)
        servo.write(f"{args.servo_rest}\n".encode())
        servo_enabled = True
        print(f"Servo connected on {args.serial_port}.")
    except Exception as exc:
        print(f"Servo disabled ({exc}). Continuing desktop-only demo.")

    palm = TRTEngine(HAND_DETECTOR_ENGINE_PATH) if use_detector else None
    lm = TRTEngine(LANDMARK_ENGINE_PATH)
    emb = TRTEngine(EMBEDDER_ENGINE_PATH)
    cls = TRTEngine(CLASSIFIER_ENGINE_PATH)

    mp_recognizer = None
    try:
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks import python as mp_python  # type: ignore
        from mediapipe.tasks.python import vision as mp_vision  # type: ignore

        mp_options = mp_vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=os.path.join(_WEIGHTS, "gesture_recognizer.task"))
        )
        mp_recognizer = mp_vision.GestureRecognizer.create_from_options(mp_options)
    except Exception:
        mp = None

    cap = cv2.VideoCapture(0)
    last_action_time = 0.0
    prev_time = 0.0
    frame_idx = 0
    mp_cached_name: str | None = None
    mp_cached_score: float | None = None
    mp_cached_has_hand = False
    mode = "detector+TRT pipeline" if use_detector else "full-frame TRT pipeline"
    print(f"demo.py active ({mode}). Press q to quit.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()
        frame_idx += 1
        h, w, _ = frame.shape

        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_in, had_roi = _landmark_input_from_frame(
            frame, use_detector=use_detector, palm_engine=palm, rgb_full=rgb_full
        )
        if use_detector and not had_roi:
            fps = int(1.0 / (now - prev_time)) if prev_time > 0 else 0
            prev_time = now
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
            cv2.putText(frame, f"No hand | FPS: {fps}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 255), 2)
            cv2.imshow("demo.py", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        assert img_in is not None
        lm_outs = lm.run(img_in)
        lms_image, lms_world_flat = _landmark_image_world_flat(lm_outs)
        lms_norm = np.clip((lms_image.astype(np.float32).reshape(21, 3) / 224.0), 0.0, 1.0)
        hand = lms_norm.reshape(1, 21, 3)

        if LEGACY_EMBEDDER_RAW_LMS and len(emb.inputs) == 1:
            emb_outs = emb.run(np.asarray(lms_image, dtype=np.float32))
        elif LEGACY_EMBEDDER_RAW_LMS and len(emb.inputs) > 1:
            lms_f = np.asarray(lms_image, dtype=np.float32).reshape(1, 21, 3)
            world_f = np.asarray(lms_world_flat, dtype=np.float32).reshape(1, 21, 3) if lms_world_flat is not None else lms_f.copy()
            feed = {}
            for tin in emb.inputs:
                nl = tin["name"].lower()
                if "handedness" in nl:
                    feed[tin["name"]] = np.ones((1, 1), dtype=np.float32)
                elif "world" in nl:
                    feed[tin["name"]] = world_f
                else:
                    feed[tin["name"]] = lms_f
            emb_outs = emb.run_feed_dict(feed)
        elif len(emb.inputs) == 1:
            emb_outs = emb.run(lms_norm.reshape(1, 21, 3))
        else:
            world_hand = np.asarray(lms_world_flat, dtype=np.float32).reshape(1, 21, 3) if lms_world_flat is not None else hand.copy()
            feed = {}
            for tin in emb.inputs:
                nl = tin["name"].lower()
                if "handedness" in nl:
                    feed[tin["name"]] = np.ones((1, 1), dtype=np.float32)
                elif "world" in nl:
                    feed[tin["name"]] = world_hand
                else:
                    feed[tin["name"]] = hand
            emb_outs = emb.run_feed_dict(feed)

        embedding = emb_outs[0].ravel()
        logits = cls.run(embedding.reshape(1, -1))[0].ravel()
        gid = int(np.argmax(logits))
        score = float(logits[gid])
        gname = _gesture_name(gid)
        source = "trt"

        mp_name_live = mp_cached_name
        mp_score_live = mp_cached_score
        mp_has_hand_live = mp_cached_has_hand
        if mp_recognizer is not None and frame_idx % MP_INFER_EVERY_N_FRAMES == 0:
            try:
                mp_image_live = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_full)
                mp_result_live = mp_recognizer.recognize(mp_image_live)
                mp_name_live = None
                mp_score_live = None
                mp_has_hand_live = False
                if mp_result_live.gestures and mp_result_live.gestures[0]:
                    top_live = max(mp_result_live.gestures[0], key=lambda c: float(c.score))
                    mp_name_live = getattr(top_live, "category_name", None) or getattr(top_live, "display_name", "") or "?"
                    mp_score_live = float(top_live.score)
                    mp_has_hand_live = True
                elif mp_result_live.hand_landmarks:
                    mp_has_hand_live = True
                mp_cached_name, mp_cached_score, mp_cached_has_hand = mp_name_live, mp_score_live, mp_has_hand_live
            except Exception:
                pass

        if mp_name_live is not None and mp_score_live is not None and mp_name_live.lower() in LABEL_TO_ID:
            gid = LABEL_TO_ID[mp_name_live.lower()]
            score = float(mp_score_live)
            gname = _gesture_name(gid)
            source = "mediapipe"

        open_ok = gid == OPEN_ID and score >= GESTURE_SCORE_MIN
        close_ok = gid == CLOSE_ID and score >= GESTURE_SCORE_MIN
        light_ok = gid == LIGHT_ID and score >= GESTURE_SCORE_MIN
        ily_ok = gid == ILOVEYOU_ID and score >= GESTURE_SCORE_MIN
        in_cooldown = (now - last_action_time) < COOLDOWN_SEC
        action = "none"

        if not in_cooldown:
            if open_ok:
                pyautogui.hotkey("ctrl", "alt", "right")
                action = "desktop_right"
                last_action_time = now
            elif close_ok:
                pyautogui.hotkey("ctrl", "alt", "left")
                action = "desktop_left"
                last_action_time = now
            elif light_ok and servo_enabled and servo is not None:
                servo.write(f"{args.servo_active}\n".encode())
                time.sleep(SERVO_PRESS_DURATION_SEC)
                servo.write(f"{args.servo_rest}\n".encode())
                action = f"servo_pulse->{args.servo_active}->{args.servo_rest}"
                last_action_time = now
            elif ily_ok:
                try:
                    subprocess.Popen(["firefox", GEMINI_URL])
                    action = "open_firefox_gemini"
                except OSError:
                    action = "firefox_not_found"
                last_action_time = now

        fps = int(1.0 / (now - prev_time)) if prev_time > 0 else 0
        prev_time = now
        color = (0, 255, 0) if action != "none" else (0, 0, 255)
        servo_state = "enabled" if servo_enabled else "disabled"
        status = f"{gname}:{score:.2f} | {source} | {action} | servo:{servo_state} | FPS:{fps}"
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.imshow("demo.py", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if mp_recognizer is not None:
        mp_recognizer.close()
    if servo is not None:
        try:
            servo.write(f"{args.servo_rest}\n".encode())
        except Exception:
            pass
        servo.close()


if __name__ == "__main__":
    main()
