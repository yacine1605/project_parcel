"""
Simple webcam demonstration for the frozen parcel-inspection pipeline.

OpenCV camera frames use BGR channel order. PIL, MobileNet preprocessing, and
the pipeline use RGB. The explicit conversions prevent swapped red/blue colors.
Press `q` to quit.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
from PIL import Image

try:
    from .inspection_pipeline import inspect_parcel_image, load_prototype_models
except ImportError:
    from inspection_pipeline import inspect_parcel_image, load_prototype_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Live four-stage parcel inspection")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index, usually 0")
    args = parser.parse_args()

    models = load_prototype_models()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Webcam {args.camera} is unavailable. Check permissions and camera index.")

    print("Webcam started. Hold a parcel in view and press 'q' to quit.")
    try:
        while True:
            success, bgr_frame = camera.read()
            if not success:
                raise RuntimeError("The webcam opened but a frame could not be captured.")

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            _, annotated_rgb = inspect_parcel_image(pil_image, models)
            # Convert the annotated PIL RGB image into a NumPy array, then back
            # to BGR because OpenCV display functions expect BGR channel order.
            display_bgr = cv2.cvtColor(np.array(annotated_rgb), cv2.COLOR_RGB2BGR)
            cv2.imshow("Parcel inspection — press q to quit", display_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
