# Models bundled for academic reproduction

The demo includes the three FP32 Core ML packages used by the project. They are
already converted: neither Python ML packages nor retraining is needed to run the app.
File hashes are in `demo-assets.json`; verify with `python3 Scripts/verify-assets.py`.

## StudentAttributes.mlpackage

Project-fitted MobileNetV3-Small, epoch 16, using torchvision ImageNet initialisation
and UTKFace for training. Conversion embeds normalisation and category temperature
scaling. This is transfer learning, not training from random initialisation.

This project artifact is supplied for non-commercial academic reproduction. UTKFace's
dataset terms restrict use to non-commercial research; original image owners retain
copyright. No UTKFace images or participant photographs are bundled. The artifact
does not grant rights to underlying images. Backbone: https://github.com/pytorch/vision ;
dataset: https://susanqq.github.io/UTKFace/ . The torchvision BSD license is retained
under `Models/Student/torchvision-LICENSE`.

## FastFaceAttributes.mlpackage

Converted from iFurySt/fastface Large 128, weight revision
`15240f03bb257385305e5985e04e6d2477a242fa`. The conversion embeds pixel scaling
and ImageNet normalisation. Source model terms and provenance are retained under
`Models/FastFace/`, including Apache-2.0. Underlying dataset terms remain separate.
Sources: https://github.com/iFurySt/fastface and https://huggingface.co/iFurySt/fastface .

## OpenCVExpression.mlpackage

Converted from OpenCV Zoo's Progressive Teacher / MobileFaceNet expression model,
ONNX SHA-256 `4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9`.
Conversion embeds RGB scaling to [-1, 1] and maps fearful/surprised to fear/surprise.
The directory's Apache-2.0 license and provenance are retained under
`Models/OpenCVExpression/`. Source:
https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition .

These converted packages are modified artifacts, not unmodified upstream releases.
Legacy FacesVisionDemo weights are excluded. Bundling models makes the implementation
runnable; it does not validate predictions for consequential decisions or establish
rights for arbitrary commercial deployment.
