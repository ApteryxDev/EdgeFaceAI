# Current model set — 6 November 2026

## Age and gender: FastFace Large 128 (replacement)

Repository: https://github.com/iFurySt/fastface
Weights/model card: https://huggingface.co/iFurySt/fastface
Pinned source and weight commits, SHA-256 hashes, input/output contract and conversion versions are in FastFace/provenance.json. Source/license, serving contract, model card and numerical conversion evidence are preserved in FastFace/.

The replacement is a joint MobileNetV3-Large model. Its Core ML artifact is FastFaceAttributes.mlpackage, converted locally using PyTorch 2.7.0, torchvision 0.22.0 and coremltools 9.0, FP32, target iOS 17. The input is aligned RGB 128x128. Image 1/255 scaling plus exact ImageNet channel mean/std normalization are embedded in the model. Outputs: numeric expected age over bins 0–100 and softmax category probabilities ordered female,male.

The app detects landmarks with Vision and estimates a similarity transform to the UniFace 128px reference template used by the source serving pipeline. Vision-derived eye/nose centroids and lip extrema differ from RetinaFace's five landmarks. That detector/alignment difference requires real-image evaluation; no equivalence to the original end-to-end pipeline is claimed.

Operational age group boundary: under-18 <18, adult 18≤age<65, elderly ≥65. This is a prespecified project definition, not a boundary supplied by IU. No confidence probability is fabricated for numeric age. Raw gender softmax scores are not calibrated correctness probabilities. Export records include numeric age, crop mode and model-set identifier.

Verification so far:
- Source checkpoint loaded with weights_only=True and exact architecture/state keys.
- Converted model saved, compiled and predicted through native macOS Core ML.
- Four synthetic crops (black, white, seeded random, gradient) compared against PyTorch and released ONNX. All specified numerical checks passed. Max Core ML age deviation was 0.000519 years; max gender-probability deviation was 0.000060.
- These are conversion checks on identical synthetic crops, not face-recognition accuracy measurements.
- Native Xcode iPhone-target build succeeded; device inference/accuracy confirmation remains pending.

Source model artifacts are released as Apache-2.0; included license applies to the replacement. Respect separately documented upstream dataset terms. Model-card metrics are author-reported on their data; they are not results from our required 20-image sample. No universal accuracy guarantee is inferred.

## Expression: legacy CNNEmotions

ExpressionClassifier.mlmodel remains the old seven-label expression model. Its original provenance is in provenance.json. It is large (~498 MB), metadata license Unknown, and source preprocessing/parity remains unresolved. Expression replacement is a separate outstanding task. Live overlay uses Apple Vision independently of this model.

## Baseline preservation

AgeClassifier.mlmodel and GenderClassifier.mlmodel are preserved locally as the legacy baseline but removed from the native Xcode Sources build phase and excluded by project.yml. They are no longer loaded by FacePipeline. Original hashes remain in provenance.json. Do not redistribute these legacy weights as if covered by the demo repository's code license.

Reproduction scripts are in ../Conversion/. The required 20-image test must remain untouched while selecting models and validating preprocessing. New real-image development comparisons and the final frozen study are still needed.

Update: physical iPhone launch confirmed by Xcode. Native Swift tests also passed for alignment identity, a known similarity transform and degenerate-point rejection. Real-face prediction accuracy and alignment quality still require evaluation.
