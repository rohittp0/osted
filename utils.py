from typing import List

from constants import RECTANGLE, BOXES


def get_center(box: RECTANGLE) -> tuple[float, float]:
    """
    Get the center of a box.
    :param box:
    :return:
    """

    if box is None:
        return 0, 0

    return (box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2


def move_box(box: RECTANGLE, v: tuple[float, float]) -> RECTANGLE:
    """
    Move a box by a velocity.
    :param box:
    :param v:
    :return:
    """

    return (box[0][0] + v[0], box[0][1] + v[1]), (box[1][0] + v[0], box[1][1] + v[1])


def box_to_tuple(box: List[int]) -> RECTANGLE:
    """
    Convert a box to a tuple.
    :param box:
    :return:
    """
    return (box[0], box[1]), (box[2], box[3])


def get_closest_box(last: RECTANGLE, box: BOXES, classes: List[int], cid: int):
    """
    Get the closest box to the last box.
    :param last:
    :param box:
    :param classes:
    :param cid:
    :return:
    """
    if last is None or classes.count(cid) == 1:
        return box_to_tuple(box[classes.index(cid)])

    last_center = get_center(last)
    min_dist = float("inf")
    min_box = None

    for i, b in enumerate(box):
        if classes[i] != cid:
            continue

        center = get_center(box_to_tuple(b))
        dist = abs(last_center[0] - center[0]) + abs(last_center[1] - center[1])

        if dist < min_dist:
            min_dist = dist
            min_box = b

    return box_to_tuple(min_box)
