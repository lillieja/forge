from __future__ import annotations
import os
import sys
import time
import cv2
import serial
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# --- Configuration ---
LANDMARK_ENGINE = "landmarks.engine"
GESTURE_ENGINE = "gestures.engine"

SERIAL_PORT = "/dev/ttyCH341USB0"
BAUD_RATE = 9600
ARDUINO_BOOT_DELAY_SEC = 2.0

# Servo/Logic Config
ANGLE_REST, ANGLE_PRESS = 90, 0
PRESS_DURATION = 0.15
GESTURE_SCORE_MIN = 0.65
STABLE_FRAMES = 10
COMMAND_COOLDOWN = 2.0
TRIGGER_INDEX = 0 

# Model Input Specs (Update these based on your training)
LM_INPUT_SHAPE = (224, 224) 

class TRTEngine:
    """Generic wrapper for any TensorRT engine."""
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def run(self, input_data):
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host']

def main():
    # 1. Serial Setup
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(ARDUINO_BOOT_DELAY_SEC)
        ser.write(f"{ANGLE_REST}\n".encode())
    except Exception as e:
        print(f"Serial Error: {e}"); sys.exit(1)

    # 2. Load Both Engines
    lm_detector = TRTEngine(LANDMARK_ENGINE)
    gs_classifier = TRTEngine(GESTURE_ENGINE)

    cap = cv2.VideoCapture(0)
    last_command_time = 0.0
    streak = 0

    print("Double-Engine Pipeline Active. 'q' to quit.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)

        # STAGE 1: Extract Landmarks from Pixels
        # Preprocess image for the landmark engine
        img_input = cv2.resize(frame, LM_INPUT_SHAPE).astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1)) # HWC to CHW
        landmarks = lm_detector.run(img_input)

        # STAGE 2: Classify Gesture from Landmarks
        # The output of Stage 1 becomes the input for Stage 2
        gesture_probs = gs_classifier.run(landmarks)
        
        gesture_id = np.argmax(gesture_probs)
        score = gesture_probs[gesture_id]
        triggered = (gesture_id == TRIGGER_INDEX and score >= GESTURE_SCORE_MIN)

        # Logic for stable detection
        now = time.time()
        if triggered:
            streak += 1
        else:
            streak = 0

        if streak == STABLE_FRAMES and (now - last_command_time) > COMMAND_COOLDOWN:
            print(f"MATCH: {gesture_id} with score {score:.2f}")
            ser.write(f"{ANGLE_PRESS}\n".encode())
            time.sleep(PRESS_DURATION)
            ser.write(f"{ANGLE_REST}\n".encode())
            last_command_time = time.time()

        # UI Overlay
        hud = f"ID: {gesture_id} | Score: {score:.2f} | Streak: {streak}"
        cv2.putText(frame, hud, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if triggered else (0, 0, 255), 2)
        cv2.imshow("Full TRT Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    ser.close()

if __name__ == "__main__":
    main()
