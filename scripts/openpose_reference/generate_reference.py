# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Draw reference OpenPose skeletons for Klartraum's skeleton renderer tests.

The drawing follows the body-pose convention of ControlNet's OpenPose annotator
(``draw_bodypose`` in lllyasviel/ControlNet and controlnet_aux), which is what
ControlNet-OpenPose models were trained on: COCO-18 keypoints, 17 limbs drawn
as filled ellipses in 60 % of their colour, then every keypoint as a filled
circle in full colour, over a black canvas. Each person is drawn completely
(limbs, then joints) before the next one.

Writes ``data/openpose/reference_keypoints.txt`` and
``data/openpose/reference_256.ppm``.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

SIZE = 256
STICK_WIDTH = 4
JOINT_RADIUS = 4

# Pairs of 0-based COCO-18 indices, in drawing order.
LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]

COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
    (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
]

# Normalized (x, y) per COCO-18 joint; None marks a joint that is not visible.
# nose, neck, r-shoulder, r-elbow, r-wrist, l-shoulder, l-elbow, l-wrist,
# r-hip, r-knee, r-ankle, l-hip, l-knee, l-ankle, r-eye, l-eye, r-ear, l-ear
PERSONS = [
    [
        (0.270, 0.150), (0.270, 0.250), (0.190, 0.255), (0.150, 0.390), (0.120, 0.520),
        (0.350, 0.255), (0.400, 0.380), (0.455, 0.490), (0.215, 0.520), (0.205, 0.700),
        (0.195, 0.880), (0.320, 0.520), (0.345, 0.700), (0.370, 0.875), (0.255, 0.130),
        (0.288, 0.131), (0.235, 0.143), (0.307, 0.145),
    ],
    [
        (0.740, 0.205), (0.715, 0.290), (0.665, 0.295), (0.610, 0.400), None,
        (0.770, 0.300), (0.840, 0.250), (0.880, 0.160), (0.680, 0.560), (0.700, 0.730),
        (0.640, 0.890), (0.760, 0.555), (0.820, 0.700), (0.850, 0.850), (0.728, 0.188),
        None, (0.700, 0.200), None,
    ],
]


def draw_bodypose(canvas: np.ndarray, keypoints: list[tuple[float, float] | None]) -> None:
    height, width = canvas.shape[:2]
    for (first, second), color in zip(LIMBS, COLORS):
        if keypoints[first] is None or keypoints[second] is None:
            continue
        xs = np.array([keypoints[first][0], keypoints[second][0]]) * float(width)
        ys = np.array([keypoints[first][1], keypoints[second][1]]) * float(height)
        mean_x = np.mean(xs)
        mean_y = np.mean(ys)
        length = ((ys[0] - ys[1]) ** 2 + (xs[0] - xs[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(ys[0] - ys[1], xs[0] - xs[1]))
        polygon = cv2.ellipse2Poly((int(mean_x), int(mean_y)), (int(length / 2), STICK_WIDTH), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])
    for keypoint, color in zip(keypoints, COLORS):
        if keypoint is None:
            continue
        x = int(keypoint[0] * width)
        y = int(keypoint[1] * height)
        cv2.circle(canvas, (x, y), JOINT_RADIUS, color, thickness=-1)


def main() -> None:
    output = Path(__file__).resolve().parents[2] / "data" / "openpose"
    output.mkdir(parents=True, exist_ok=True)

    canvas = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for person in PERSONS:
        draw_bodypose(canvas, person)

    # The canvas holds RGB values (the colour table is RGB), as a PPM expects.
    with open(output / "reference_256.ppm", "wb") as image:
        image.write(f"P6\n{SIZE} {SIZE}\n255\n".encode("ascii"))
        image.write(canvas.tobytes())

    with open(output / "reference_keypoints.txt", "w", encoding="ascii") as keypoints:
        keypoints.write("# size stick_width joint_radius\n")
        keypoints.write(f"{SIZE} {STICK_WIDTH} {JOINT_RADIUS}\n")
        keypoints.write("# person joint x y visible\n")
        for person_index, person in enumerate(PERSONS):
            for joint_index, keypoint in enumerate(person):
                x, y = keypoint if keypoint is not None else (0.0, 0.0)
                keypoints.write(f"{person_index} {joint_index} {x:.6f} {y:.6f} {int(keypoint is not None)}\n")
    print(f"Wrote {output / 'reference_256.ppm'} and {output / 'reference_keypoints.txt'}")


if __name__ == "__main__":
    main()
