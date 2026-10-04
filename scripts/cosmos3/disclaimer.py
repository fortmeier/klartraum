# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Usage notice shown by every Cosmos 3 script in this directory."""

from __future__ import annotations

import sys


COSMOS3_NOTICE = """\
WARNING: NVIDIA Cosmos 3 is not part of Klartraum.
  These scripts download and/or process third-party NVIDIA Cosmos3-Edge
  model weights, which are distributed under their own license (OpenMDW 1.1)
  by NVIDIA. Klartraum neither ships nor endorses these models. You are
  solely responsible for complying with the model license and for how you use
  the models and anything they generate.
"""


def print_cosmos3_notice() -> None:
    print(COSMOS3_NOTICE, file=sys.stderr, flush=True)
