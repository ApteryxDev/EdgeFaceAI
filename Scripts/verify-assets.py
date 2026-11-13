#!/usr/bin/env python3
"""Check published models and synthetic fixtures without ML dependencies."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "Models/demo-assets.json").read_text())
failures = []
for name, expected in manifest["sha256"].items():
    path = root / name
    if not path.is_file():
        failures.append(f"Missing: {name}")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(f"Checksum mismatch: {name}")
if failures:
    raise SystemExit("\n".join(failures))
print(f"PASS: {len(manifest['sha256'])} model/resource files match their SHA-256 checksums.")
