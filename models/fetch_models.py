#!/usr/bin/env python3
"""Fetch and verify the models listed in models/manifest.json (standard library only).

    python models/fetch_models.py --list
    python models/fetch_models.py --pair en-hi          # everything one language pair needs
    python models/fetch_models.py --only asr.whisper.base.q5_1
    python models/fetch_models.py --pin                 # write hashes of unpinned models into the manifest

Downloads go next to the manifest, at each entry's `path`. Every file is checked against the
manifest's size and SHA-256; entries with an empty sha256 are hashed and, with --pin, pinned
("trust on first use"). Sources starting with `convert:` are produced by the ml/ factory; the
script prints the command instead of downloading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "emotionedge-fetch/0.1"})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    tmp.replace(dest)


def verify(entry: dict, path: Path) -> tuple[bool, str]:
    target = path / entry["check_file"] if entry.get("check_file") else path
    if not target.is_file():
        return False, "missing"
    if entry.get("bytes") and target.stat().st_size != entry["bytes"]:
        return False, f"size {target.stat().st_size} != {entry['bytes']}"
    actual = sha256_of(target)
    expected = entry.get("sha256", "").lower()
    if not expected:
        return True, f"unpinned (sha256 {actual})"
    return (actual == expected), ("ok" if actual == expected else f"sha256 {actual} != {expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show models and their status, download nothing")
    parser.add_argument("--only", nargs="*", default=[], help="model ids to fetch")
    parser.add_argument("--pair", help="fetch the models of a language pack, e.g. en-hi")
    parser.add_argument("--pin", action="store_true", help="write hashes of unpinned models into the manifest")
    parser.add_argument("--dest", type=Path, help="store models in this directory instead (a copy of the "
                        "manifest goes there too; run with --set pipeline.models=DEST/manifest.json)")
    args = parser.parse_args()

    global HERE, MANIFEST
    if args.dest:
        args.dest.mkdir(parents=True, exist_ok=True)
        if not (args.dest / "manifest.json").exists():
            shutil.copy(MANIFEST, args.dest / "manifest.json")
        HERE = args.dest.resolve()
        MANIFEST = HERE / "manifest.json"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    wanted = set(args.only)
    if args.pair:
        pack = manifest.get("language_packs", {}).get(args.pair)
        if pack is None:
            print(f"unknown language pack {args.pair}", file=sys.stderr)
            return 2
        wanted |= set(pack.values())

    failures = 0
    changed = False
    for entry in manifest["models"]:
        if wanted and entry["id"] not in wanted:
            continue
        path = HERE / entry["path"]
        print(f"{entry['id']}  [{entry.get('license', 'license?')}]")
        ok, status = verify(entry, path)
        if args.list:
            print(f"  {status}")
            continue
        source = entry.get("source", "")
        if status == "missing" or not ok:
            if source.startswith("convert:"):
                print(f"  produced by the ml/ factory: run {source[len('convert:'):]}")
                continue
            download(source, path)
            for sidecar in entry.get("sidecars", []):
                download(sidecar, path.parent / sidecar.rsplit("/", 1)[-1])
            ok, status = verify(entry, path)
        print(f"  {status}")
        if not ok:
            failures += 1
        elif args.pin and status.startswith("unpinned"):
            entry["sha256"] = status.split()[-1].rstrip(")")
            changed = True
    if changed:
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("manifest updated with pinned hashes")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
