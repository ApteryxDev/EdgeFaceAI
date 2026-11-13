#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 Scripts/verify-assets.py
xcodebuild -project EdgeFaceAI.xcodeproj -scheme EdgeFaceAI \
  -configuration Debug -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath .build/DerivedData CODE_SIGNING_ALLOWED=NO build
python3 - <<'PY'
from pathlib import Path
root = Path('.build/DerivedData/Build/Products/Debug-iphonesimulator/EdgeFaceAI.app')
for name in ['StudentAttributes', 'FastFaceAttributes', 'OpenCVExpression']:
    assert (root / (name + '.mlmodelc')).is_dir(), f'Missing compiled model: {name}'
for name in ['expected.json', 'expression-expected.json', 'age-interval-calibration.json']:
    assert (root / 'ConversionFixtures' / name).is_file(), f'Missing resource: {name}'
print('PASS: app bundle contains all three compiled models and calibration/self-check resources.')
PY
