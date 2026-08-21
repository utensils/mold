#!/usr/bin/env python3
"""Fetch the public-domain face photographs the #1222 parity fixtures use.

Provenance only: this script is committed so the fixture set can be rebuilt and
audited, never run by the test suite. It downloads from Wikimedia Commons,
records each file's license metadata, and downscales so the committed images
stay small.

    python3 crates/mold-inference/testdata/pulid/fetch_faces.py crates/mold-inference/testdata/pulid/faces

Every selected file must be public domain or CC0. The script refuses anything
else rather than quietly committing a restrictively licensed portrait.
"""

import io
import json
import os
import re
import sys
import urllib.parse
import urllib.request

from PIL import Image

USER_AGENT = "mold-pulid-fixture/1.0 (https://github.com/utensils/mold; research)"

# Chosen for pose and lighting variety; all NASA/ESA official portraits, which
# Commons records as public domain.
TITLES = [
    "File:Kayla Barron official portrait.jpg",
    "File:Raja Chari official portrait.jpg",
    "File:Frank Rubio official portrait.jpg",
    "File:Mae Jemison - Official portrait of 1987 astronaut candidate.jpg",
]

ACCEPTABLE = ("public domain", "cc0")
MAX_WIDTH = 800
JPEG_QUALITY = 82
MAX_BYTES = 300 * 1024


def strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value or "").strip()


def query(titles):
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "iiurlwidth": str(MAX_WIDTH),
        "titles": "|".join(titles),
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return json.load(urllib.request.urlopen(req))["query"]["pages"]


def slug(title: str) -> str:
    base = title[len("File:"):].rsplit(".", 1)[0]
    return re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")


def main(out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for page in query(TITLES).values():
        title = page.get("title")
        if "imageinfo" not in page:
            print(f"MISSING on Commons: {title}", file=sys.stderr)
            return 1
        info = page["imageinfo"][0]
        meta = info.get("extmetadata", {})
        license_name = strip_html(meta.get("LicenseShortName", {}).get("value", ""))
        if not any(token in license_name.lower() for token in ACCEPTABLE):
            print(f"REFUSED (license `{license_name}`): {title}", file=sys.stderr)
            return 1
        source_url = info.get("thumburl") or info["url"]
        req = urllib.request.Request(source_url, headers={"User-Agent": USER_AGENT})
        raw = urllib.request.urlopen(req).read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        if image.width > MAX_WIDTH:
            height = round(image.height * MAX_WIDTH / image.width)
            image = image.resize((MAX_WIDTH, height), Image.LANCZOS)
        name = f"{slug(title)}.jpg"
        path = os.path.join(out_dir, name)
        quality = JPEG_QUALITY
        while True:
            image.save(path, "JPEG", quality=quality, optimize=True)
            if os.path.getsize(path) <= MAX_BYTES or quality <= 50:
                break
            quality -= 6
        manifest.append(
            {
                "file": name,
                "title": title,
                "license": license_name,
                "usage_terms": strip_html(meta.get("UsageTerms", {}).get("value", "")),
                "credit": strip_html(meta.get("Artist", {}).get("value", "")),
                "description_url": info.get("descriptionurl"),
                "source_url": source_url.split("?")[0],
                "width": image.width,
                "height": image.height,
                "bytes": os.path.getsize(path),
            }
        )
        print(f"{name}  {image.width}x{image.height}  {os.path.getsize(path)} bytes  [{license_name}]")
    manifest.sort(key=lambda entry: entry["file"])
    with open(os.path.join(out_dir, "sources.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "crates/mold-inference/testdata/pulid/faces"))
