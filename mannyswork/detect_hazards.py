import cv2
import torch
import numpy as np
from transformers import pipeline
from PIL import Image


# Load only the depth-estimation model.
depth_estimator = pipeline(
    task="depth-estimation",
    # model="depth-anything/Depth-Anything-V2-Small-hf",
    model="depth-anything/Depth-Anything-V2-Base-hf",
    device=0 if torch.cuda.is_available() else -1,
)

# Open the external USB camera.
USB_CAMERA_INDEX = 1
cap = cv2.VideoCapture(USB_CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap.release()
    raise RuntimeError(
        f"Could not open USB camera at index {USB_CAMERA_INDEX}. "
        "Check that it is connected and not being used by another application."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    waist_top = int(h * 0.35)
    waist_bottom = int(h * 0.65)

    zones = {
        "MOTOR 1 (Far Left)": (0, int(w * 0.20)),
        "MOTOR 2 (Near Left)": (int(w * 0.20), int(w * 0.40)),
        "MOTOR 3 (Near Right)": (int(w * 0.60), int(w * 0.80)),
        "MOTOR 4 (Far Right)": (int(w * 0.80), w),
    }

    # Smaller input reduces depth-estimation latency.
    inference_height = 320
    inference_width = int(w * inference_height / h)
    inference_frame = cv2.resize(
        frame,
        (inference_width, inference_height),
        interpolation=cv2.INTER_AREA,
    )
    rgb_frame = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
    depth_output = depth_estimator(Image.fromarray(rgb_frame))
    depth_map = np.array(depth_output["depth"])
    depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map = depth_map.astype(np.uint8)

    cv2.rectangle(frame, (0, waist_top), (w, waist_bottom), (255, 255, 0), 1)
    waist_depth = depth_map[waist_top:waist_bottom, :]

    for motor_name, (zx1, zx2) in zones.items():
        closeness_score = np.median(waist_depth[:, zx1:zx2])
        if closeness_score > 160:
            cv2.rectangle(
                frame,
                (zx1, waist_top),
                (zx2, waist_bottom),
                (0, 0, 255),
                3,
            )
            cv2.putText(
                frame,
                motor_name,
                (zx1, waist_bottom - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

    depth_display = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
    cv2.rectangle(
        depth_display,
        (0, waist_top),
        (w, waist_bottom),
        (255, 255, 255),
        1,
    )
    combined = np.hstack((frame, depth_display))

    cv2.imshow("Depth Estimation", combined)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
