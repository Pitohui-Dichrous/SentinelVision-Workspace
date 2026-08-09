"""Pure spatial policy applied before semantic conflict resolution."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, TypeVar


DetectionT = TypeVar("DetectionT")
Point = Tuple[float, float]
Polygon = Sequence[Sequence[float]]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Return whether a point is inside a polygon using an even-odd ray test."""

    x, y = point
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
        ):
            inside = not inside
    return inside


def filter_monitoring_domain(
    detections: Iterable[DetectionT],
    rois: Iterable[Polygon] = (),
    masks: Iterable[Polygon] = (),
) -> Tuple[DetectionT, ...]:
    """Filter raw detections by ROI/mask before boxes can be merged.

    ROI and mask polygons define the monitored physical domain. Applying this
    policy to raw boxes prevents an out-of-domain conflict box from moving a
    valid in-domain detection's fused center across a boundary.
    """

    roi_items = tuple(rois)
    mask_items = tuple(masks)
    output = []
    for detection in detections:
        x1, y1, x2, y2 = getattr(detection, "box")
        center = ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)
        if any(point_in_polygon(center, mask) for mask in mask_items):
            continue
        if roi_items and not any(point_in_polygon(center, roi) for roi in roi_items):
            continue
        output.append(detection)
    return tuple(output)
