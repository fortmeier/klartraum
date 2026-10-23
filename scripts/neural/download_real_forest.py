# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Downloads real photographs of forests seen from a distance from Wikimedia Commons, for the adversarial loss of
train_tree_enhancer.py (--real).

The files of the category "Aerial photographs of forests" (and its subcategories, three levels deep) and of a few
searches are kept when their licence allows the use (public domain, CC0, CC BY, CC BY-SA); each is fetched as its
960 px wide thumbnail (a size the servers keep, so they do not throttle it), a few per second, waiting when asked
to. Writes <out-dir>/images/commons_NNNNN.jpg and <out-dir>/attribution.csv (file, title, author, credit, licence,
its URL, source page); a second run continues where the first stopped.

    python3 scripts/neural/download_real_forest.py [out-dir]   # default build/TestingOutput/real_forest
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "klartraum-research/0.1 (forest photos for training a renderer)"
API = "https://commons.wikimedia.org/w/api.php"
ROOT_CATEGORIES = ["Aerial_photographs_of_forests"]
SEARCHES = ["aerial view forest", "drone forest", "Luftbild Wald", "forest landscape panorama"]
ACCEPTED = re.compile(r"^(Public domain|CC0|PDM-owner|No restrictions|Attribution|CC BY(-SA)? [0-9.]+( \w+)?)$")


def api(params: dict, post: bool = False) -> dict:
    encoded = urllib.parse.urlencode({**params, "format": "json"})
    request = (urllib.request.Request(API, data=encoded.encode(), headers={"User-Agent": USER_AGENT}) if post
               else urllib.request.Request(API + "?" + encoded, headers={"User-Agent": USER_AGENT}))
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(request, timeout=60))
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    return {}


def category_files(roots: list[str], depth: int = 3) -> set[str]:
    seen, files, queue = set(), set(), [(root, 0) for root in roots]
    while queue:
        category, level = queue.pop(0)
        if category in seen:
            continue
        seen.add(category)
        cont: dict = {}
        while True:
            d = api({"action": "query", "list": "categorymembers", "cmtitle": "Category:" + category,
                     "cmtype": "file|subcat", "cmlimit": "500", **cont})
            for member in d.get("query", {}).get("categorymembers", []):
                if member["ns"] == 6:
                    files.add(member["title"])
                elif member["ns"] == 14 and level < depth:
                    queue.append((member["title"][len("Category:"):], level + 1))
            if "continue" not in d:
                break
            cont = d["continue"]
    return files


def search_files(queries: list[str]) -> set[str]:
    files: set[str] = set()
    for query in queries:
        offset = 0
        while offset < 1000:
            d = api({"action": "query", "list": "search", "srsearch": query + " filetype:bitmap", "srnamespace": "6",
                     "srlimit": "500", "sroffset": str(offset)})
            files.update(hit["title"] for hit in d.get("query", {}).get("search", []))
            if "continue" not in d:
                break
            offset = d["continue"]["sroffset"]
        time.sleep(1.0)
    return files


def metadata(titles: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(titles), 50):
        d = api({"action": "query", "prop": "imageinfo", "titles": "|".join(titles[i:i + 50]),
                 "iiprop": "url|size|extmetadata|mime", "iiurlwidth": "960"}, post=True)
        for page in d.get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata", {})

            def value(key: str) -> str:
                return re.sub("<[^>]+>", "", meta.get(key, {}).get("value", "")).strip()

            out.append({"title": page.get("title"), "page": info.get("descriptionurl"), "thumb": info.get("thumburl"),
                        "mime": info.get("mime"), "license": value("LicenseShortName"),
                        "license_url": value("LicenseUrl"), "artist": value("Artist"), "credit": value("Credit")})
        time.sleep(0.5)
    return out


def download(url: str, path: Path) -> bool:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            path.write_bytes(urllib.request.urlopen(request, timeout=60).read())
            return True
        except urllib.error.HTTPError as error:
            wait = float(error.headers.get("Retry-After", 0) or 0) if error.headers else 0.0
            time.sleep(min(max(wait, 2.0 * (attempt + 1)), 30.0))
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    return False


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "build/TestingOutput/real_forest")
    (out / "images").mkdir(parents=True, exist_ok=True)
    listing = out / "metadata.json"
    if listing.exists():
        files = json.loads(listing.read_text())
    else:
        titles = sorted(category_files(ROOT_CATEGORIES) | search_files(SEARCHES))
        print(f"{len(titles)} candidate files", flush=True)
        files = metadata(titles)
        listing.write_text(json.dumps(files, indent=0))
    kept = [f for f in files if f["thumb"] and ACCEPTED.match(f["license"] or "")
            and f["mime"] in ("image/jpeg", "image/png")]
    print(f"{len(kept)} with an accepted licence", flush=True)
    rows = []
    for i, f in enumerate(kept):
        name = f"commons_{i:05d}.jpg"
        path = out / "images" / name
        if not path.exists():
            download(f["thumb"], path)
            time.sleep(0.5)
        if path.exists():
            rows.append({"file": name, "title": f["title"], "author": f["artist"], "credit": f["credit"],
                         "license": f["license"], "license_url": f["license_url"], "source": f["page"]})
        if i % 100 == 0:
            print(f"{i} of {len(kept)}", flush=True)
    with open(out / "attribution.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} photographs in {out}", flush=True)


if __name__ == "__main__":
    main()
