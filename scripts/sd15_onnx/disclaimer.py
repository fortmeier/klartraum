"""Usage notice shown by every Stable Diffusion script in this directory."""

from __future__ import annotations

import sys


STABLE_DIFFUSION_NOTICE = """\
WARNING: Stable Diffusion is not part of Klartraum.
  These scripts download and/or process third-party Stable Diffusion 1.5
  model weights, which are distributed under their own license (CreativeML
  Open RAIL-M) by their respective authors. Klartraum neither ships nor
  endorses these models. You are solely responsible for complying with the
  model license and for how you use the models and anything they generate.
"""


def print_stable_diffusion_notice() -> None:
    print(STABLE_DIFFUSION_NOTICE, file=sys.stderr, flush=True)
