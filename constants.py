from typing import List, Generator

import numpy as np

RECTANGLE = tuple[tuple[float, float], tuple[float, float]]
V_BUFFER = Generator[List[np.ndarray], None, None]
BOXES = List[List[int]]

BUFFER_SIZE = 16
