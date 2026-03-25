import cv2

from kitting_cv.segmentation import BinSegmenter
from kitting_cv.tracking import HandTracker


def run_camera_pipeline(camera_index: int = 0) -> None:
    hand_tracker = HandTracker()
    bin_segmenter = BinSegmenter()

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Could not open camera.")

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        hand_detections = hand_tracker.detect(frame)
        bin_mask = bin_segmenter.segment(frame)

        cv2.putText(
            frame,
            f"Hands detected: {len(hand_detections)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        cv2.imshow("kitting-camera", frame)
        cv2.imshow("bin-mask", bin_mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()
