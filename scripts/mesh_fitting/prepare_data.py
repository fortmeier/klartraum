# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "opencv-python-headless>=4.10"]
# ///
# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Prepares photographs, masks and poses of one scene for mesh_texture_fitting_example --scene.

Input: a scene directory with images/, masks/ and splat_poses.json (OpenGL camera-to-world poses, +X right,
+Y up, -Z forward, with per-image pinhole intrinsics and OpenCV distortion, as exported from COLMAP/Nerfstudio).

Per frame, the image and the mask are undistorted, the mask shrunk by --erode target pixels (its edge often holds
background), both downsampled by area averaging (which turns the mask's hard edge
into fractional coverage) and the colour is premultiplied by the mask. A start sphere around the object comes from
carving the masks on a voxel grid; views whose mask disagrees with the carved hull are excluded. Output, in the output directory:

  targets.bin  float32, views x height x width x 4: premultiplied rgb, mask
  cameras.bin  float32, views x 16: Vulkan clip-from-world matrices (y down, depth 0 to 1), column-major
  scene.txt    "key values..." lines: width, height, views, train and validation view indices, centre, radius,
               excluded views (faulty masks), point count and spacing
  points.bin   float32, points x 3: the surface voxel centres of the visual hull (start points for Gaussian
               splatting)
  preview.png  every view's target over gray, for a look

Run from the repository root:
  uv run scripts/mesh_fitting/prepare_data.py SCENE_DIR build/TestingOutput/real_data/SCENE [--width 256]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

NEAR, FAR = 0.02, 20.0


def clip_from_camera(fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> np.ndarray:
    """The Vulkan projection of an OpenGL camera (looking down -Z, +Y up) with pinhole intrinsics.

    fx, fy, cx, cy are in pixels of a width x height image, with the principal point measured from the image's
    corner (pixel x covers [x, x + 1)). Screen x = fx X / -Z + cx and screen y (down) = fy Y / Z + cy map to
    normalized device coordinates 2 screen / size - 1; depth is 0 at NEAR and 1 at FAR.
    """
    return np.array(
        [
            [2 * fx / width, 0, -(2 * cx / width - 1), 0],
            [0, -2 * fy / height, -(2 * cy / height - 1), 0],
            [0, 0, FAR / (NEAR - FAR), -FAR * NEAR / (FAR - NEAR)],
            [0, 0, -1, 0],
        ]
    )


def project(clip_from_world: np.ndarray, points: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Screen positions (pixels, y down) of world points and whether each lies in front of the camera."""
    clip = np.c_[points, np.ones(len(points))] @ clip_from_world.T
    w = clip[:, 3]
    screen = (clip[:, :2] / w[:, None] * 0.5 + 0.5) * [width, height]
    return screen, w > NEAR


def sample(mask: np.ndarray, screen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The mask at screen positions (nearest pixel) and whether they lie inside the image."""
    height, width = mask.shape
    x, y = np.floor(screen[:, 0]).astype(int), np.floor(screen[:, 1]).astype(int)
    inside = (x >= 0) & (y >= 0) & (x < width) & (y < height)
    values = np.zeros(len(screen))
    values[inside] = mask[y[inside], x[inside]]
    return values, inside


def carve(cameras: list[np.ndarray], masks: list[np.ndarray], centre: np.ndarray, half: float, resolution: int):
    """The voxel centres of a cube around centre that project into the masks of the views that see them, and those
    of them on the hull's surface (a face neighbour carved away).

    A few faulty masks must not cut away the object, so a voxel survives up to max(2, 10 %) views placing it
    outside their mask.
    """
    axis = np.linspace(-half, half, resolution)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3) + centre
    seen = np.zeros(len(grid), int)
    misses = np.zeros(len(grid), int)
    for camera, mask in zip(cameras, masks):
        height, width = mask.shape
        screen, front = project(camera, grid, width, height)
        values, inside = sample(mask, screen)
        visible = front & inside
        seen += visible
        misses += visible & (values <= 0.5)
    keep = misses <= np.maximum(2, np.ceil(0.1 * seen))
    occupied = keep.reshape(resolution, resolution, resolution)
    padded = np.pad(occupied, 1)
    interior = occupied.copy()
    for axis_index in range(3):
        for shift in (-1, 1):
            interior &= np.roll(padded, shift, axis_index)[1:-1, 1:-1, 1:-1]
    return grid[keep], grid[(occupied & ~interior).reshape(-1)]


def hull_iou(camera: np.ndarray, mask: np.ndarray, hull: np.ndarray, spacing: float) -> float:
    """The IoU of a mask with the projection of the hull's voxels (each drawn as a disc of its projected size)."""
    height, width = mask.shape
    screen, front = project(camera, hull, width, height)
    clip_w = (np.c_[hull, np.ones(len(hull))] @ camera.T)[:, 3]
    radius = np.maximum(1, np.ceil(0.5 * spacing * abs(camera[0, 0]) * width / 2 / np.maximum(clip_w, 1e-6)))
    drawn = np.zeros((height, width), np.uint8)
    for (x, y), r, ok in zip(screen, radius, front):
        if ok and -r <= x < width + r and -r <= y < height + r:
            cv2.circle(drawn, (int(x), int(y)), int(r), 1, -1)
    covered = mask > 0.5
    union = (covered | (drawn > 0)).sum()
    return float((covered & (drawn > 0)).sum() / max(union, 1))


def ray_centre(c2w: list[np.ndarray], fx: float, fy: float, centres: list[np.ndarray]) -> np.ndarray:
    """The point closest (least squares) to the rays through the mask centroids."""
    a, b = np.zeros((3, 3)), np.zeros(3)
    for pose, (cx_offset, cy_offset) in zip(c2w, centres):
        direction = pose[:3, :3] @ np.array([cx_offset / fx, -cy_offset / fy, -1.0])
        direction /= np.linalg.norm(direction)
        projector = np.eye(3) - np.outer(direction, direction)
        a += projector
        b += projector @ pose[:3, 3]
    return np.linalg.solve(a, b)


def sphere_coverage(camera: np.ndarray, centre: np.ndarray, radius: float, width: int, height: int) -> np.ndarray:
    """Per pixel centre, whether the sphere covers it (exact: the ray hits the sphere)."""
    inverse = np.linalg.inv(camera)
    ys, xs = np.mgrid[0:height, 0:width] + 0.5
    ndc = np.stack([2 * xs / width - 1, 2 * ys / height - 1], -1).reshape(-1, 2)
    near = np.c_[ndc, np.zeros(len(ndc)), np.ones(len(ndc))] @ inverse.T
    far = np.c_[ndc, np.ones(len(ndc)), np.ones(len(ndc))] @ inverse.T
    origin = near[:, :3] / near[:, 3:]
    direction = far[:, :3] / far[:, 3:] - origin
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    offset = origin - centre
    b = np.sum(offset * direction, 1)
    discriminant = b * b - (np.sum(offset * offset, 1) - radius * radius)
    return (discriminant > 0).reshape(height, width)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", type=Path, help="scene directory with images/, masks/ and splat_poses.json")
    parser.add_argument("output", type=Path, help="output directory")
    parser.add_argument("--width", type=int, default=256, help="target image width in pixels (default 256)")
    parser.add_argument(
        "--erode",
        type=float,
        default=1.0,
        help="shrink the masks by this many target pixels before downsampling, to drop background at their edge "
        "(default 1)",
    )
    arguments = parser.parse_args()

    poses = json.loads((arguments.scene / "splat_poses.json").read_text(encoding="utf-8"))
    frames = poses["frames"]
    arguments.output.mkdir(parents=True, exist_ok=True)
    targets, cameras, masks, c2ws, mask_centres, splits = [], [], [], [], [], []
    for frame in frames:
        image = cv2.imread(str(arguments.scene / frame["file_path"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(arguments.scene / frame["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise SystemExit(f"cannot read {frame['file_path']} or {frame['mask_path']}")
        full_width, full_height = frame["w"], frame["h"]
        if image.shape[:2] != (full_height, full_width) or mask.shape != (full_height, full_width):
            raise SystemExit(f"{frame['file_path']}: size differs from the poses")
        k = np.array([[frame["fl_x"], 0, frame["cx"]], [0, frame["fl_y"], frame["cy"]], [0, 0, 1]])
        distortion = np.array([frame["k1"], frame["k2"], frame["p1"], frame["p2"], frame.get("k3", 0.0)])
        image = cv2.undistort(image, k, distortion)
        mask = cv2.undistort(mask, k, distortion)
        width = arguments.width
        height = round(width * full_height / full_width)
        erosion = round(arguments.erode * full_width / width)
        if erosion > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion + 1, 2 * erosion + 1))
            mask = cv2.erode(mask, kernel)
        scale_x, scale_y = width / full_width, height / full_height
        rgb = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255
        alpha = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        targets.append(np.dstack([rgb * alpha[:, :, None], alpha]))
        masks.append(alpha)
        # OpenCV puts pixel centres at integer coordinates; here pixel x covers [x, x + 1).
        fx, fy = frame["fl_x"] * scale_x, frame["fl_y"] * scale_y
        cx, cy = (frame["cx"] + 0.5) * scale_x, (frame["cy"] + 0.5) * scale_y
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        cameras.append(clip_from_camera(fx, fy, cx, cy, width, height) @ np.linalg.inv(c2w))
        c2ws.append(c2w)
        ys, xs = np.nonzero(alpha > 0.5)
        mask_centres.append((xs.mean() + 0.5 - cx, ys.mean() + 0.5 - cy))
        splits.append(frame["source_split"])
        print(f"{frame['file_path']}: {splits[-1]}, coverage {alpha.mean():.3f}")

    # The start sphere: carve a coarse grid around the rays' meeting point, then a fine one around the hull.
    distances = [np.linalg.norm(pose[:3, 3]) for pose in c2ws]
    centre = ray_centre(c2ws, fx, fy, mask_centres)
    half = 0.5 * float(np.median(distances))
    for resolution in (48, 96):
        hull, surface = carve(cameras, masks, centre, half, resolution)
        if len(hull) == 0:
            raise SystemExit("carving removed every voxel: the masks and poses do not agree")
        centre = 0.5 * (hull.min(0) + hull.max(0))
        half = 0.6 * float((hull.max(0) - hull.min(0)).max()) + 2 * half / resolution
    radius = float(np.linalg.norm(hull - centre, axis=1).max())
    spacing = 2 * half / (96 - 1)
    print(f"visual hull: {len(hull)} voxels, centre {np.round(centre, 4)}, bounding radius {radius:.4f}")

    # Views whose mask disagrees with the hull (a faulty mask) are excluded from training and validation.
    excluded = []
    for index, (camera, mask) in enumerate(zip(cameras, masks)):
        iou = hull_iou(camera, mask, hull, spacing)
        if iou < 0.6:
            excluded.append(index)
            print(f"view {index} ({frames[index]['file_path']}): mask vs. hull IoU {iou:.2f}, excluded")

    columns = 10
    tiles = [np.clip(t[:, :, :3] + 0.5 * (1 - t[:, :, 3:]), 0, 1) for t in targets]
    tiles += [np.full_like(tiles[0], 0.5)] * (-len(tiles) % columns)
    rows = [np.hstack(tiles[r : r + columns]) for r in range(0, len(tiles), columns)]
    cv2.imwrite(str(arguments.output / "preview.png"), (np.vstack(rows)[:, :, ::-1] * 255).astype(np.uint8))

    # Checks: the centre projects into the mask of every view that shows it (close-ups may cut it off), and the
    # start sphere covers the masks.
    ious = []
    for index, (camera, mask) in enumerate(zip(cameras, masks)):
        if index in excluded:
            continue
        screen, front = project(camera, centre[None], width, height)
        value, inside = sample(mask, screen)
        if not front[0] or (inside[0] and value[0] <= 0.5):
            raise SystemExit(f"view {index}: the hull centre does not project into the mask")
        sphere = sphere_coverage(camera, centre, radius, width, height)
        covered = mask > 0.5
        ious.append((sphere & covered).sum() / (sphere | covered).sum())
        if (covered & ~sphere).sum() > 0.01 * covered.sum():
            raise SystemExit(f"view {index}: the start sphere misses more than 1 % of the mask")
    print(f"start sphere vs. masks: IoU mean {np.mean(ious):.3f}, min {np.min(ious):.3f}")

    np.stack(targets).astype(np.float32).tofile(arguments.output / "targets.bin")
    np.stack([camera.T.reshape(16) for camera in cameras]).astype(np.float32).tofile(arguments.output / "cameras.bin")
    train = [str(i) for i, split in enumerate(splits) if split == "train" and i not in excluded]
    validation = [str(i) for i, split in enumerate(splits) if split != "train" and i not in excluded]
    surface.astype(np.float32).tofile(arguments.output / "points.bin")
    (arguments.output / "scene.txt").write_text(
        f"width {width}\nheight {height}\nviews {len(frames)}\ntrain {' '.join(train)}\n"
        f"validation {' '.join(validation)}\ncentre {centre[0]:.6f} {centre[1]:.6f} {centre[2]:.6f}\n"
        f"radius {radius:.6f}\nexcluded {' '.join(str(i) for i in excluded)}\npoints {len(surface)}\n"
        f"spacing {spacing:.6f}\n",
        encoding="utf-8",
    )
    print(f"wrote {arguments.output}: {len(frames)} views ({len(train)} train, {len(validation)} validation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
