import cv2
import numpy as np
from yolov7_package import Yolov7Detector

from constants import V_BUFFER
from utils import get_closest_box, get_center, move_box


def stabilize_frame(frame, base, current) -> np.ndarray:
    """
    Stabilize the frame so that the object stays in the same position.

    :param frame: The current video frame.
    :param base: The base position of the object ((x1, y1), (x2, y2)).
    :param current: The current position of the object ((x1, y1), (x2, y2)).
    :return: Stabilized frame.
    """

    # Calculate center points of base and param boxes
    base_center = get_center(base)
    param_center = get_center(current)

    # Calculate translation
    dx = base_center[0] - param_center[0]
    dy = base_center[1] - param_center[1]

    # Define translation matrix
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    # Apply affine translation to the frame
    stabilized_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    return stabilized_frame


def stabilize_by_class_id(video: V_BUFFER, cid: int):
    det = Yolov7Detector(traced=False)

    base = None
    last = ((0, 0), (0, 0))
    v_last = (0, 0)

    for frames in video:
        classes, boxes, scores = det.detect(frames)

        for i, box in enumerate(boxes):
            if cid not in classes[i]:
                last = move_box(last, v_last)
                yield stabilize_frame(frames[i], base or last, last)
                continue

            last_center = get_center(last)
            last = get_closest_box(last, box, classes[i], cid)
            current_center = get_center(last)

            v_last = (current_center[0] - last_center[0], current_center[1] - last_center[1])

            if base is None:
                base = last
                yield frames[i]
                continue

            yield stabilize_frame(frames[i], base, last)
