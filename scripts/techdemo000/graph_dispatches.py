# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Counts the GPU commands each element of a graph records.

Runs a program that compiles graphs with timestamp profiling and writes them as
JSON (klartraum::writeGraphJson()), under the api_dump layer, into a scratch
directory. Every profiled command buffer belongs to one graph, in the order the
graphs were compiled; within it, each element's commands lie between the
timestamp pair the graph writes around it. Writes <name>.commands.json next to
<name>.json: per element {vkCmd name: count}, in element order. The graph JSON
itself (with the GPU times of a full measurement) is left as it is.

    # one graph of techdemo000_graphs
    python3 scripts/techdemo000/graph_dispatches.py NAME [--out-dir build/TestingOutput/techdemo000_graphs]
    # the SD1.5 chain of techdemo000_example (graphs listed in DIR/chain.json under "compiled")
    python3 scripts/techdemo000/graph_dispatches.py --chain DIR
"""

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def profiled_command_buffers(dump: Path):
    """Per profiled command buffer, in recording order, the command counts of its elements."""
    calls = re.findall(r"\b(vkCmd[A-Za-z0-9]+|vkBeginCommandBuffer|vkEndCommandBuffer)\(", dump.read_text())
    buffers, elements, current, stamps = [], None, None, 0
    for call in calls:
        if call == "vkBeginCommandBuffer":
            elements, current = [], None
        elif call == "vkEndCommandBuffer":
            if elements:
                buffers.append(elements)
            elements = None
        elif elements is None:
            continue
        elif call == "vkCmdResetQueryPool" and current is None:
            current, stamps = collections.Counter(), 0
        elif call == "vkCmdWriteTimestamp" and current is not None:
            stamps += 1
            if stamps == 2:
                elements.append(dict(current))
                current = None
        elif current is not None:
            current[call] += 1
    return buffers


def run_dumped(command, dump: Path):
    env = dict(os.environ, VK_INSTANCE_LAYERS="VK_LAYER_LUNARG_api_dump", VK_APIDUMP_LOG_FILENAME=str(dump),
               VK_APIDUMP_DETAILED="false", VK_APIDUMP_OUTPUT_FORMAT="text")
    subprocess.run(command, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def write_commands(graph_path: Path, commands, out_dir: Path):
    graph = json.loads(graph_path.read_text())
    if len(commands) != len(graph["elements"]):
        raise SystemExit(f"{graph_path}: {len(commands)} recorded elements, {len(graph['elements'])} in the graph")
    (out_dir / f"{graph_path.stem}.commands.json").write_text(json.dumps(commands))
    total = collections.Counter()
    for counts in commands:
        total.update(counts)
    print(graph_path.stem, dict(total))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?")
    parser.add_argument("--out-dir", default="build/TestingOutput/techdemo000_graphs")
    parser.add_argument("--binary", default="build/examples/techdemo000_graphs")
    parser.add_argument("--chain", help="directory for the SD1.5 chain of techdemo000_example")
    parser.add_argument("--example", default="build/examples/techdemo000_example")
    parser.add_argument("--backend", default="raster")
    args = parser.parse_args()
    scratch = Path(tempfile.mkdtemp(prefix="graph_dispatches_", dir="build/TestingOutput"))
    try:
        dump = scratch / "api_dump.txt"
        if args.chain:
            out = Path(args.chain)
            out.mkdir(parents=True, exist_ok=True)
            # One generated frame is enough to record every graph of the chain.
            run_dumped([args.example, "--from", "6", "--to", "6", "--step", "100000", "--sd-path", "gpu", "--backend",
                        args.backend, "--dump-graphs", str(scratch), "--out-dir", str(scratch / "render")], dump)
            compiled = json.loads((scratch / "chain.json").read_text())["compiled"]
            buffers = profiled_command_buffers(dump)
            if len(buffers) != len(compiled):
                raise SystemExit(f"{len(buffers)} profiled command buffers for {len(compiled)} graphs")
            for name, commands in zip(compiled, buffers):
                write_commands(scratch / f"{name}.json", commands, out)
        else:
            out = Path(args.out_dir)
            run_dumped([args.binary, "--graph", args.name, "--frames", "1", "--backend", args.backend, "--out-dir",
                        str(scratch)], dump)
            write_commands(scratch / f"{args.name}.json", profiled_command_buffers(dump)[-1], out)
    finally:
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
