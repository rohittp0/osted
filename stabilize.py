from typing import Generator

import cv2
import numpy as np
from yolov7_package import Yolov7Detector


def stabilize_frame(frame, base, current) -> np.ndarray:
    """
    Stabilize the frame so that the object stays in the same position.

    :param frame: The current video frame.
    :param base: The base position of the object ((x1, y1), (x2, y2)).
    :param current: The current position of the object ((x1, y1), (x2, y2)).
    :return: Stabilized frame.
    """

    # Calculate center points of base and param boxes
    base_center = ((base[0][0] + base[1][0]) // 2, (base[0][1] + base[1][1]) // 2)
    param_center = ((current[0][0] + current[1][0]) // 2, (current[0][1] + current[1][1]) // 2)

    # Calculate translation
    dx = base_center[0] - param_center[0]
    dy = base_center[1] - param_center[1]

    # Define translation matrix
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])

    c1, c2 = (int(current[0][0]), int(current[0][1])), (int(current[1][0]), int(current[1][1]))
    b1, b2 = (int(base[0][0]), int(base[0][1])), (int(base[1][0]), int(base[1][1]))
    frame = cv2.rectangle(frame, c1, c2, (255, 0, 0), thickness=1)
    frame = cv2.rectangle(frame, b1, b2, (0, 255, 0), thickness=1)

    # Apply affine translation to the frame
    stabilized_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    return stabilized_frame


def stabilize_by_class_id(video: Generator[np.ndarray, None, None], cid: int, shape=(640, 640)):
    det = Yolov7Detector(traced=False, img_size=list(shape))

    base = None
    last = None

    for frames in video:
        classes, boxes, scores = det.detect(frames)

        for i, box in enumerate(boxes):
            if cid not in classes[i]:
                yield stabilize_frame(frames[i], base, last) if last is not None else frames[i]
                continue

            x1, y1, x2, y2 = box[classes[i].index(cid)]
            last = (x1, y1), (x2, y2)

            if base is None:
                base = last
                yield frames[i]
                continue

            yield stabilize_frame(frames[i], base, last)



