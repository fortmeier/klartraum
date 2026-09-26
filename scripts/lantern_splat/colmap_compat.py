"""Forward Nerfstudio's COLMAP 3.x arguments to the local COLMAP 4.x build."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


FLAG_RENAMES = {
    "--SiftExtraction.use_gpu": "--FeatureExtraction.use_gpu",
    "--SiftMatching.use_gpu": "--FeatureMatching.use_gpu",
}

launcher = Path(os.environ["KLARTRAUM_COLMAP"])
arguments = [FLAG_RENAMES.get(argument, argument) for argument in sys.argv[1:]]
environment = os.environ.copy()

if launcher.suffix.lower() in (".bat", ".cmd"):
    colmap_root = launcher.parent
    executable = colmap_root / "bin" / "colmap.exe"
    environment["PATH"] = str(executable.parent) + os.pathsep + environment.get("PATH", "")
    environment["QT_PLUGIN_PATH"] = str(colmap_root / "plugins")
else:
    executable = launcher

raise SystemExit(subprocess.call([str(executable), *arguments], env=environment))
