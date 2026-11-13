# Build and reproduce EdgeFace AI


The project was developed with Xcode 27.2, an iPhone Air and an iOS 17 deployment target. The exported source is not a self-contained app build: the local model binaries and authorised image resources below must first be supplied.

1. Obtain the original datasets under their source terms. UTKFace: https://susanqq.github.io/UTKFace/ . FACES: https://faces.mpdl.mpg.de/ . Do not assume restricted participant imagery can be redistributed.
2. Install the training/conversion dependencies in a Python virtual environment using `Training/requirements.txt` and `Conversion/requirements.txt`. Preserve the model versions/provenance documented under `Models/`.
3. Inspect each script's command-line options. Prepare grouped splits with `Training/prepare_utk.py`, train using `Training/train_student.py`, then calibrate and explicitly evaluate the frozen test using `Training/calibrate_and_evaluate.py`. Training uses ImageNet initialisation; it does not use FastFace age/category weights. Model-specific source assets must be obtained separately before conversion.
4. Run the appropriate conversion scripts from `Conversion/`. The app expects `StudentAttributes.mlpackage`, `FastFaceAttributes.mlpackage`, `OpenCVExpression.mlpackage` and the optional legacy `ExpressionClassifier.mlmodel` in `Models/`. The project also retains legacy age/category model references, which require local originals or removal of those unused build references.
5. Prepare authorised gallery manifests/images under the matching `Resources/` folders. FACES and UTK study drivers expect 72 and 40 images respectively. Personal/web gallery entries use `id`, `filename`, `source` and optional reference/crop metadata. Private manifests are deliberately omitted.
6. Prepare `Resources/ConversionFixtures/` with `prepare_parity_fixtures.py`, the expression fixtures from its converter, and the calibrated interval JSON. Model availability and conversion should be verified rather than silently substituted.
7. Open `EdgeFaceAI.xcodeproj`, set your own signing team, connect a trusted iPhone and run. Do not reuse another developer's signing identity. The app reports missing models and malformed resources explicitly.

The `age-interval-calibration.json` conservative radius must be produced from the retained calibration audit before selecting the conservative interval. An image crop alone is not a UTK pre-cropped training chip; ordinary inputs still use Vision alignment.


## Native Mac benchmark

With local assets available, compile the benchmark from the repository root:

```sh
mkdir -p .build
swiftc -O App/FaceAlignment.swift Benchmark/FacePipeline.swift Benchmark/Benchmark.swift -o .build/mac-benchmark
.build/mac-benchmark "$PWD" "$PWD/private-data/mac-latency"
```

The original automation environment restricted Core ML/Vision command-line execution. A native Xcode command-line tool without debugger attachment was used for the successful benchmark. The runner compiles local model packages if needed; real model files and the private frozen gallery are required.


[Return to the project overview](../README.md)
