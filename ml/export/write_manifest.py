#!/usr/bin/env python3
"""P4 "sign + write manifest.json": register exported models in models/manifest.json with their
size and SHA-256 (which the C++ model registry verifies before loading), and optionally sign
the manifest with Ed25519.

    python ml/export/write_manifest.py add --id mt.nllb200.distilled600m.int8 --task mt --format ct2 \\
        --path mt/nllb-200-distilled-600M-int8 --check-file model.bin --license "CC-BY-NC-4.0"
    python ml/export/write_manifest.py rehash            # refresh size + sha256 of every present file
    python ml/export/write_manifest.py sign --key ed25519.pem   # writes manifest.json.sig (needs `cryptography`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "models" / "manifest.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def target_file(manifest_dir: Path, entry: dict) -> Path:
    base = manifest_dir / entry["path"]
    return base / entry["check_file"] if entry.get("check_file") else base


def refresh(manifest_dir: Path, entry: dict) -> bool:
    """Updates bytes/sha256 from disk; returns False if the file is missing."""
    path = target_file(manifest_dir, entry)
    if not path.is_file():
        return False
    entry["bytes"] = path.stat().st_size
    entry["sha256"] = sha256_of(path)
    return True


def canonical(manifest: dict) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def cmd_add(args: argparse.Namespace, manifest: dict, manifest_dir: Path) -> None:
    entry = {"id": args.id, "task": args.task, "format": args.format, "path": args.path,
             "languages": args.languages or ["*"], "devices": args.devices or ["cpu"],
             "license": args.license or "unknown", "source": args.source or "exported by ml/"}
    if args.check_file:
        entry["check_file"] = args.check_file
    if args.precision:
        entry["precision"] = args.precision
    if not refresh(manifest_dir, entry):
        raise SystemExit(f"model file not found: {target_file(manifest_dir, entry)}")
    models = [m for m in manifest.setdefault("models", []) if m["id"] != args.id]
    models.append(entry)
    manifest["models"] = models
    print(f"registered {args.id}: {entry['bytes']} bytes, sha256 {entry['sha256']}")


def cmd_rehash(manifest: dict, manifest_dir: Path) -> None:
    for entry in manifest.get("models", []):
        status = "updated" if refresh(manifest_dir, entry) else "missing (unchanged)"
        print(f"{entry['id']}: {status}")


def cmd_sign(args: argparse.Namespace, manifest: dict, manifest_path: Path) -> None:
    try:
        from cryptography.hazmat.primitives import serialization  # type: ignore
    except ImportError:
        raise SystemExit("signing needs the 'cryptography' package")
    key = serialization.load_pem_private_key(Path(args.key).read_bytes(), password=None)
    signature = key.sign(canonical(manifest))
    sig_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")
    sig_path.write_bytes(signature)
    print(f"signed: {sig_path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="register or replace a model")
    add.add_argument("--id", required=True)
    add.add_argument("--task", required=True, choices=["vad", "asr", "emotion.acoustic", "emotion.lexical", "speaker", "mt", "tts"])
    add.add_argument("--format", required=True, choices=["onnx", "ggml", "ct2", "piper", "kokoro"])
    add.add_argument("--path", required=True, help="relative to the manifest directory")
    add.add_argument("--check-file", help="file inside a directory model to hash (CTranslate2: model.bin)")
    add.add_argument("--languages", nargs="*")
    add.add_argument("--devices", nargs="*")
    add.add_argument("--precision")
    add.add_argument("--license")
    add.add_argument("--source")
    sub.add_parser("rehash", help="refresh sizes and hashes of present files")
    sign = sub.add_parser("sign", help="Ed25519-sign the manifest")
    sign.add_argument("--key", required=True, help="PEM private key")
    args = parser.parse_args(argv)

    manifest_path: Path = args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.command == "add":
        cmd_add(args, manifest, manifest_path.parent)
    elif args.command == "rehash":
        cmd_rehash(manifest, manifest_path.parent)
    elif args.command == "sign":
        cmd_sign(args, manifest, manifest_path)
        return 0
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
