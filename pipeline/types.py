from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Subject bounding box in full-frame pixel coordinates."""
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass
class CropRect:
    """Output crop rectangle in full-frame pixel coordinates (top-left origin)."""
    x1: int
    y1: int
    w: int
    h: int

    @property
    def center_x(self) -> float:
        return self.x1 + self.w / 2.0

    @property
    def center_y(self) -> float:
        return self.y1 + self.h / 2.0

    @property
    def x2(self) -> int:
        return self.x1 + self.w

    @property
    def y2(self) -> int:
        return self.y1 + self.h
