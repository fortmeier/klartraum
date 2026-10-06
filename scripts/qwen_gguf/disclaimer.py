# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Usage notice shown by every Qwen script in this directory."""

from __future__ import annotations

import sys


QWEN_NOTICE = """\
WARNING: Qwen3.6 is not part of Klartraum.
  These scripts download and/or process third-party Qwen model weights,
  which Alibaba Cloud distributes under their own license (Apache 2.0).
  The GGUF conversions are made by third parties (e.g. Unsloth). Klartraum
  neither ships nor endorses these models. You are solely responsible for
  complying with the model license and for how you use the models and
  anything they generate.
"""


def print_qwen_notice() -> None:
    print(QWEN_NOTICE, file=sys.stderr, flush=True)
