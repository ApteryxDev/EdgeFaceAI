# Run EdgeFace AI

The repository includes the three converted Core ML models and synthetic
self-check/calibration resources. No model download, Python ML environment, dataset
installation, XcodeGen or retraining is needed for the default app.

## Requirements

- A Mac with full Xcode and an installed iOS SDK. Development uses Xcode 27.2;
  deployment target: iOS 17. Older Xcode versions are unverified.
- For live camera: an iPhone running iOS 17 or later, a trusted connection,
  Developer Mode and your own Xcode signing team.
- A simulator supports the interface and photo selection. Add a single-face image
  you have permission to use to its Photos app. Camera/device timing evaluation
  requires real hardware.

## Quick start

```sh
git clone https://github.com/ApteryxDev/EdgeFaceAI.git
cd EdgeFaceAI
python3 Scripts/verify-assets.py
open EdgeFaceAI.xcodeproj
```

1. Choose the shared **EdgeFaceAI** scheme.
2. Select an installed iPhone simulator and press Run, or select your connected iPhone.
   On iPhone, open target **Signing & Capabilities**, select your own team and use
   a unique bundle identifier if required.
3. The app opens in **Photos** mode. Select a clear photograph containing exactly one
   face, then tap **Analyze photo on device**. Results appear beneath the image.
4. Compare **My trained model** and **FastFace baseline**. Both use the included OpenCV
   expression model.
5. On a physical iPhone, select **Camera**, tap **Start camera**, and grant camera
   access if prompted. Frames are processed locally and are not recorded as video.
6. The two **Check ... conversion** buttons run bundled synthetic fixtures. Passing
   validates those fixed inputs, not recognition accuracy. The public student check
   uses three synthetic inputs; the original private report also tested two UTK images.

No signing identity or credentials are shared. Use your own team for a physical device.

## Check the build without signing

```sh
bash Scripts/check-build.sh
```

This verifies file hashes, builds for a generic iOS simulator and checks the resulting
bundle for compiled models and calibration/fixture files. It does not claim a simulator
launch or a physical-device inference test.

## Optional research galleries

FACES, UTK40 and the personal/web gallery are optional. Their pickers appear when local
resources are available. None is needed for photo or live-camera inference.
No participant photographs or private reference manifests are published.

For authorised research reproduction, place the original local resources under
`Resources/FACESDevelopment`, `Resources/UTKDevelopment` and `Resources/InternetCandidates`.
FACES and UTK drivers validate 72 and 40 images respectively, unique identifiers and
matching image files. Formats are defined by `FACESExample` and `InternetAgeExample`
in the app source. Folder references bundle authorised resources added before building.
Do not commit private images or manifests.

## Optional training and conversion

Only needed to reproduce or change weights. Install `Training/requirements.txt` and
`Conversion/requirements.txt` in a separate Python environment. Obtain datasets and
upstream assets under their source terms. Use `Training/prepare_utk.py`,
`Training/train_student.py` and `Training/calibrate_and_evaluate.py`; inspect each
script's `--help` and the retained training configuration. Converters are in `Conversion/`.
Retraining may not produce identical weights on a different platform.

## Optional Mac benchmark

```sh
mkdir -p .build
swiftc -O App/FaceAlignment.swift Benchmark/FacePipeline.swift Benchmark/Benchmark.swift -o .build/mac-benchmark
.build/mac-benchmark "$PWD" "$PWD/private-data/mac-latency"
```

The research benchmark requires the private frozen gallery. It is separate from the
demo quick start. Published aggregate measurements remain under `Results/`.

## Troubleshooting

- **Missing model/checksum failure:** obtain a complete checkout including `Models/`.
  Restore recorded artifacts instead of substituting weights.
- **Signing error:** select your own team and an available bundle identifier.
- **No face/multiple faces:** select an image with exactly one clear face.
- **No simulator camera:** use Photos mode or a physical iPhone.
- **Galleries absent:** expected when optional research data is not installed.

Model provenance and terms: [THIRD_PARTY_NOTICES.md](../Models/THIRD_PARTY_NOTICES.md).
