# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Download selected BVH files of the 100STYLE dataset.

100STYLE (Ian Mason, Sebastian Starke, Taku Komura, 2022; CC BY 4.0,
https://zenodo.org/records/8127870) is one 1.5 GB zip archive. This script
reads the archive's central directory with HTTP range requests and fetches
only the members it needs, in parallel, so the full archive is never
downloaded.

    uv run python download_100style.py --list
    uv run python download_100style.py --styles Neutral Proud --output ../../build/character_animation/100STYLE
"""

from __future__ import annotations

import argparse
import http.client
import struct
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

URL = "https://zenodo.org/records/8127870/files/100STYLE.zip?download=1"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "build" / "character_animation" / "100STYLE"

# Locomotion styles with natural, everyday gaits; the 100STYLE file names are
# <Style>/<Style>_<Motion>.bvh with motions such as FW (forward walk), FR
# (forward run), BW, BR, SW, SR, ID (idle) and TR1-TR3 (transitions).
DEFAULT_STYLES = [
    "Neutral", "Proud", "Old", "Elated", "Strutting", "Rushed", "BigSteps", "Heavyset", "ArmsBySide", "Angry",
    "Depressed", "March",
]


@dataclass
class Member:
    name: str
    method: int
    compressed_size: int
    size: int
    header_offset: int


def fetch(start: int, end: int, attempts: int = 5) -> bytes:
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end - 1}"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"server ignored the range request (HTTP {response.status})")
                return response.read()
        except (OSError, http.client.HTTPException):
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def archive_size() -> int:
    request = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def central_directory() -> list[Member]:
    size = archive_size()
    tail = fetch(max(0, size - 65536 - 22), size)
    position = tail.rfind(b"PK\x05\x06")
    if position < 0:
        raise RuntimeError("end of central directory not found")
    entries, directory_size, directory_offset = struct.unpack_from("<HII", tail, position + 10)
    # ZIP64 archives store the real values in the ZIP64 end record.
    if directory_offset == 0xFFFFFFFF or entries == 0xFFFF:
        locator = tail.rfind(b"PK\x06\x07")
        (record_offset,) = struct.unpack_from("<Q", tail, locator + 8)
        record = fetch(record_offset, record_offset + 56)
        entries, directory_size, directory_offset = struct.unpack_from("<QQQ", record, 32)
    directory = fetch(directory_offset, directory_offset + directory_size)
    members = []
    position = 0
    while position < len(directory):
        if directory[position : position + 4] != b"PK\x01\x02":
            raise RuntimeError("corrupt central directory")
        method, = struct.unpack_from("<H", directory, position + 10)
        compressed, uncompressed = struct.unpack_from("<II", directory, position + 20)
        name_length, extra_length, comment_length = struct.unpack_from("<HHH", directory, position + 28)
        (offset,) = struct.unpack_from("<I", directory, position + 42)
        name = directory[position + 46 : position + 46 + name_length].decode("utf-8")
        extra = directory[position + 46 + name_length : position + 46 + name_length + extra_length]
        if 0xFFFFFFFF in (compressed, uncompressed, offset):
            values = iter(_zip64_values(extra))
            if uncompressed == 0xFFFFFFFF:
                uncompressed = next(values)
            if compressed == 0xFFFFFFFF:
                compressed = next(values)
            if offset == 0xFFFFFFFF:
                offset = next(values)
        members.append(Member(name, method, compressed, uncompressed, offset))
        position += 46 + name_length + extra_length + comment_length
    return members


def _zip64_values(extra: bytes) -> list[int]:
    position = 0
    while position + 4 <= len(extra):
        tag, length = struct.unpack_from("<HH", extra, position)
        if tag == 0x0001:
            return [struct.unpack_from("<Q", extra, position + 4 + 8 * i)[0] for i in range(length // 8)]
        position += 4 + length
    return []


def extract(member: Member, output: Path) -> Path:
    target = output / member.name.split("/", 1)[-1]
    if target.exists() and target.stat().st_size == member.size:
        return target
    # The local header has its own name and extra field lengths; 1 KiB covers them.
    data = fetch(member.header_offset, member.header_offset + 30 + 1024 + member.compressed_size)
    name_length, extra_length = struct.unpack_from("<HH", data, 26)
    payload = data[30 + name_length + extra_length : 30 + name_length + extra_length + member.compressed_size]
    content = payload if member.method == 0 else zlib.decompress(payload, -15)
    if len(content) != member.size:
        raise RuntimeError(f"{member.name}: unexpected size")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--styles", nargs="+", default=DEFAULT_STYLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--list", action="store_true", help="list the styles in the archive and exit")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    # The archive also holds macOS resource forks (__MACOSX/, ._*); skip them.
    members = [
        member
        for member in central_directory()
        if member.name.lower().endswith(".bvh") and "__MACOSX" not in member.name and "/._" not in member.name
    ]
    if args.list:
        styles = sorted({member.name.split("/")[-2] for member in members})
        print(f"{len(styles)} styles: {' '.join(styles)}")
        print(f"example member: {members[0].name}")
        return

    wanted = [member for member in members if member.name.split("/")[-2] in set(args.styles)]
    missing = set(args.styles) - {member.name.split("/")[-2] for member in wanted}
    if missing:
        sys.exit(f"styles not in the archive: {' '.join(sorted(missing))}")
    total = sum(member.compressed_size for member in wanted)
    print(f"fetching {len(wanted)} files, {total / 2**20:.1f} MiB compressed")
    with ThreadPoolExecutor(args.workers) as pool:
        for path in pool.map(lambda member: extract(member, args.output), wanted):
            print(path)


if __name__ == "__main__":
    main()
