# Examiner demo verification

Prepared on 13 November 2026 with Xcode 27.2.

- All 18 required model/resource files pass the recorded SHA-256 checks.
- The three model packages exactly match the original working project's artifacts.
- Native Xcode built the iOS Simulator target successfully with zero errors and zero
  warnings recorded in the build log.
- The generated app bundle contains StudentAttributes.mlmodelc,
  FastFaceAttributes.mlmodelc and OpenCVExpression.mlmodelc, plus both synthetic
  fixture manifests and the conservative age-interval calibration JSON.
- The default app starts in Photos mode. Missing optional research galleries are
  hidden; the unavailable legacy expression model is not offered.

The command-line build in the assistant's restricted environment could not invoke
Xcode's nested compiler sandbox or Simulator services. The successful build above
was performed through native Xcode. This is a build verification, not a new hardware
benchmark or a new recognition experiment. A new interactive photo/camera inference
session and the in-app conversion buttons were not verified during this packaging pass.

The included synthetic fixtures preserve their source-runtime expected values.
The public student check deliberately uses three synthetic images, not the two
additional UTKFace images used in the original private report's five-input check.
