from pathlib import Path
from typing import Generator

import numpy as np
import cv2
from tqdm import tqdm

from stabilize import stabilize_by_class_id


def get_video_props(path):
    cap = cv2.VideoCapture(path)

    return {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }


def read_video_as_frames(path, buffer_size=16) -> Generator[np.ndarray, None, None]:
    cap = cv2.VideoCapture(path)
    buffer = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        buffer.append(frame)

        if len(buffer) == buffer_size:
            yield buffer
            buffer = []

    if len(buffer):
        yield buffer


def main(video):
    props = get_video_props(video)
    frames = read_video_as_frames(video)

    print(props)
    props["fps"] = float(input("New fps ( blank to use default ): ") or "0") or props["fps"]

    width, height = props["width"], props["height"]
    out_path = str(Path(video).parent.joinpath("out.mp4"))

    stabilizer = stabilize_by_class_id(frames, 39, (width, height))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, props["fps"], (width, height))

    for frame in tqdm(stabilizer, "Processing Video", props["frames"]):
        out.write(frame)

    out.release()


if __name__ == '__main__':
    main("video.mp4")
