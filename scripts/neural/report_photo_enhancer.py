# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Hourly report of a photo enhancer training (train_photo_enhancer.py): the loss curves of log.txt (training losses
smoothed over 10 entries, validation L1 and VGG distance), the newest validation grid, and with --flight an mp4 of a
fly-in rendered with the current checkpoint (enhance_flight.py). Writes into <run>/hourly/<label>/.

    uv run --project scripts/qwen_image python scripts/neural/report_photo_enhancer.py build/TestingOutput/gan_runs/v1 \\
        --flight build/TestingOutput/qwen_flyin
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--flight", type=Path)
    parser.add_argument("--every", type=int, default=3, help="every n-th fly-in frame for the mp4")
    parser.add_argument("--uv-flight", type=Path, help="the fly-in's coordinate G-buffers (textured checkpoints)")
    parser.add_argument("--label", default=time.strftime("%m%d_%H%M"))
    args = parser.parse_args()
    out = args.run / "hourly" / args.label
    out.mkdir(parents=True, exist_ok=True)

    train, validation = {}, []
    for line in (args.run / "log.txt").read_text().splitlines():
        if m := re.match(r"step (\d+) (.*) \(", line):
            values = dict(re.findall(r"(\w+) ([-\d.]+)", m.group(2)))
            train[int(m.group(1))] = {k: float(v) for k, v in values.items()}  # a resumed run overwrites its steps
        elif m := re.match(r"validation step (\d+): L1 ([\d.]+), VGG ([\d.]+)", line):
            validation.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    steps = sorted(train)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for key, axis in (("l1", 0), ("vgg", 0), ("adv", 1), ("d", 1), ("fm", 2)):
        points = [(s, train[s][key]) for s in steps if key in train[s]]
        if len(points) < 2:
            continue
        x, y = np.array(points).T
        kernel = min(10, len(y))
        smooth = np.convolve(y, np.ones(kernel) / kernel, mode="valid")
        axes[axis].plot(x[kernel - 1:], smooth, label=key)
    if validation:
        v = np.array(validation)
        axes[0].plot(v[:, 0], v[:, 1], "o-", label="validation L1")
        axes[0].plot(v[:, 0], v[:, 2], "o-", label="validation VGG")
    for axis, title in zip(axes, ("reconstruction", "adversarial (G, D)", "feature matching")):
        axis.set_title(title)
        axis.set_xlabel("step")
        axis.legend()
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out / "losses.png", dpi=90)
    grids = sorted(args.run.glob("validation_*.png"))
    if grids:
        shutil.copy(grids[-1], out / grids[-1].name)
    print(f"step {steps[-1] if steps else 0}; validation {validation[-3:]}; report in {out}")

    if args.flight:
        checkpoint = out / "checkpoint.pt"
        shutil.copy(args.run / "checkpoint.pt", checkpoint)
        subprocess.run([sys.executable, str(Path(__file__).with_name("enhance_flight.py")), str(args.flight),
                        str(checkpoint), str(out / "flight"), "--every", str(args.every)]
                       + (["--uv-flight", str(args.uv_flight)] if args.uv_flight else []), check=True)
        checkpoint.unlink()


if __name__ == "__main__":
    main()
