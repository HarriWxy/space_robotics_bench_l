from typing import Tuple

from pydantic import BaseModel


class Transform(BaseModel):
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
