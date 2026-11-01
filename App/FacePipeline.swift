import Foundation
import Vision
import CoreML
import ImageIO
import CoreGraphics
import CoreImage

// the models have to expose the Vision classifier interface, labels are not guessed
enum ComputePolicy: String, CaseIterable, Identifiable {
    case cpuOnly, cpuAndGPU, all
    var id: String { rawValue }
    // mapping onto the Core ML compute units
    var units: MLComputeUnits {
        switch self {
        case .cpuOnly: return .cpuOnly
        case .cpuAndGPU: return .cpuAndGPU
        case .all: return .all
        }
    }
}

// one attribute output including its own request timing
struct AttributeResult: Codable, Identifiable {
    var id: String { attribute }
    let attribute: String
    let model: String
    let label: String
    let score: Float?
    let requestMilliseconds: Double
}

// full record of one inference, interval fields stay nil without a calibration file
struct InferenceRecord: Codable {
    let timestamp: Date
    var source: String = "photo"
    let policy: String
    let faceDetectionMilliseconds: Double
    let pipelineMilliseconds: Double
    let results: [AttributeResult]
    var ageIntervalLow: Double? = nil
    var ageIntervalHigh: Double? = nil
    var intervalTargetCoverage: Double? = nil
    var calibratedCategoryScore: Bool? = nil
    var ageIntervalMethod: String? = nil
    var estimatedAge: Double? = nil
    var cropMode: String? = nil
    var modelSet: String = "legacy"
    // extra figures for the statistics overlay, all optional so older exports still decode
    var categoryProbabilities: [String: Double]? = nil
    var expressionScores: [String: Double]? = nil
    var faceDetectionConfidence: Double? = nil
}

enum PipelineError: LocalizedError {
    case invalidImage, faceCount(Int), missingModel(String), unsupportedOutput(String)
    var errorDescription: String? {
        switch self {
        case .invalidImage: return "Cannot decode the image."
        case .faceCount(let n): return n == 0 ? "No face detected. Face the camera or choose a clear photo." : "Found \(n) faces. Keep one face in view."
        case .missingModel(let n): return "Missing \(n). Add the verified Core ML classifier to the app target."
        case .unsupportedOutput(let n): return "\(n) does not expose classification labels. A model-specific adapter is required."
        }
    }
}

// actor serialises inference and keeps the synchronous Vision work off the UI actor
actor FacePipeline {
    // cached models and the policy they were loaded with
    private var cachedPolicy: ComputePolicy?
    private var cachedModelName: String?
    private var cachedExpressionName: String?
    private var conservativeAgeRadius: Double?
    // which model the pickers currently point at
    private var selectedExpressionName: String { UserDefaults.standard.string(forKey: "expressionModel") == "legacy" ? "ExpressionClassifier" : "OpenCVExpression" }
    private var selectedModelName: String { UserDefaults.standard.string(forKey: "attributeModel") == "fastface" ? "FastFaceAttributes" : "StudentAttributes" }
    private var attributes: MLModel?
    private var expression: VNCoreMLModel?
    private let context = CIContext(options: [.cacheIntermediates: false])

    // loads the models, but only when the name or the policy changed
    private func load(_ policy: ComputePolicy, includeExpression: Bool) throws {
        let modelName = selectedModelName
        let expressionName = selectedExpressionName
        // already loaded with the same policy, nothing to do
        if cachedModelName == modelName, cachedPolicy == policy, attributes != nil,
           (!includeExpression || (cachedExpressionName == expressionName && expression != nil)) { return }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = policy.units
        guard let attributeURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
            throw PipelineError.missingModel(modelName)
        }
        // reuse the attribute model when only the expression model changed
        let nextAttributes: MLModel
        if cachedModelName == modelName, cachedPolicy == policy, let attributes { nextAttributes = attributes }
        else { nextAttributes = try MLModel(contentsOf: attributeURL, configuration: configuration) }
        var nextExpression = cachedPolicy == policy && cachedExpressionName == expressionName ? expression : nil
        if includeExpression && nextExpression == nil {
            guard let url = Bundle.main.url(forResource: expressionName, withExtension: "mlmodelc") else {
                throw PipelineError.missingModel(expressionName)
            }
            nextExpression = try VNCoreMLModel(for: MLModel(contentsOf: url, configuration: configuration))
        }
        attributes = nextAttributes
        expression = nextExpression
        cachedPolicy = policy
        cachedModelName = modelName
        cachedExpressionName = expressionName
        // radius from the calibration file, read once
        if conservativeAgeRadius == nil,
           let url = Bundle.main.url(forResource: "age-interval-calibration", withExtension: "json", subdirectory: "ConversionFixtures"),
           let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] {
            conservativeAgeRadius = object["candidate_radius_years"] as? Double
        }
    }

    // entry point for encoded image data, reads the EXIF orientation first
    func run(data: Data, policy: ComputePolicy, includeExpression: Bool = true, isPrecropped: Bool = false) throws -> InferenceRecord {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw PipelineError.invalidImage
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let rawOrientation = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value ?? 1
        let orientation = CGImagePropertyOrientation(rawValue: rawOrientation) ?? .up
        return try run(image: image, orientation: orientation, policy: policy,
                       includeExpression: includeExpression, isPrecropped: isPrecropped)
    }

    // shared pipeline for photos and rotated camera frames
    func run(image: CGImage, orientation: CGImagePropertyOrientation = .up,
             policy: ComputePolicy, includeExpression: Bool = true, isPrecropped: Bool = false) throws -> InferenceRecord {
        // Model loading is excluded from reported pipeline latency; no load timing is exported.
        try load(policy, includeExpression: includeExpression)
        let start = ProcessInfo.processInfo.systemUptime
        // normalise EXIF once so alignment, attributes and expression see the same pixels
        let input = CIImage(cgImage: image).oriented(orientation)
        guard let upright = context.createCGImage(input, from: input.extent),
              let attributes else { throw PipelineError.invalidImage }
        // exactly one face is required, otherwise there is nothing to attribute
        let detection = VNDetectFaceLandmarksRequest()
        try VNImageRequestHandler(cgImage: upright, orientation: .up).perform([detection])
        let faces = detection.results ?? []
        guard faces.count == 1, let face = faces.first else { throw PipelineError.faceCount(faces.count) }
        let detectionMS = (ProcessInfo.processInfo.systemUptime - start) * 1000
        let attributeStart = ProcessInfo.processInfo.systemUptime
        // UTK chips are already cropped, aligning again would shift the training distribution
        let directChip = cachedModelName == "StudentAttributes" && isPrecropped
        let aligned = directChip ? upright : try FaceAlignment.crop(image: upright, face: face, context: context)
        guard let constraint = attributes.modelDescription.inputDescriptionsByName["face"]?.imageConstraint else {
            throw PipelineError.unsupportedOutput("FastFaceAttributes input")
        }
        // the model returns age plus two category probabilities in one request
        let feature = try MLFeatureValue(cgImage: aligned, constraint: constraint, options: nil)
        let prediction = try attributes.prediction(from: MLDictionaryFeatureProvider(dictionary: ["face":feature]))
        guard let ages = prediction.featureValue(for: "estimated_age")?.multiArrayValue,
              let gender = prediction.featureValue(for: "gender_probabilities")?.multiArrayValue,
              ages.count == 1, gender.count == 2 else {
            throw PipelineError.unsupportedOutput("FastFaceAttributes")
        }
        let isStudent = cachedModelName == "StudentAttributes"
        let modelLabel = isStudent ? "Student-MobileNetV3-Small" : "FastFace-Large-128"
        let age = ages[0].doubleValue
        let female = gender[0].doubleValue, male = gender[1].doubleValue
        // sanity check the outputs before they end up in a record
        guard age.isFinite, age >= 0, age <= (isStudent ? 116 : 100),
              female.isFinite, male.isFinite, (0...1).contains(female), (0...1).contains(male) else {
            throw PipelineError.unsupportedOutput("FastFaceAttributes nonfinite/range output")
        }
        let sharedMS = (ProcessInfo.processInfo.systemUptime - attributeStart) * 1000
        // fixed boundary: adult 18–64, elderly from 65, not a medical split
        let ageGroup = age < 18 ? "under-18" : (age < 65 ? "adult" : "elderly")
        var outputs = [
            AttributeResult(attribute: "age-group", model: modelLabel, label: ageGroup,
                            score: nil, requestMilliseconds: sharedMS),
            // the higher of the two probabilities is the label, kept as the score
            AttributeResult(attribute: "gender-category", model: modelLabel, label: male > female ? "male" : "female",
                            score: Float(max(female,male)), requestMilliseconds: sharedMS)
        ]
        // every class score of the expression model, kept for the statistics overlay
        var expressionScores: [String: Double]? = nil
        // expression is optional, the UTK run skips it
        if includeExpression {
            guard let expression else { throw PipelineError.missingModel("ExpressionClassifier") }
            let request = VNCoreMLRequest(model: expression)
            let isOpenCV = cachedExpressionName == "OpenCVExpression"
            let expressionImage: CGImage
            // OpenCV expects a 112 crop, the legacy model takes the full image plus a region of interest
            if isOpenCV { expressionImage = try FaceAlignment.crop(image: upright, face: face, context: context, size: 112) }
            else { expressionImage = upright; request.regionOfInterest = face.boundingBox }
            request.imageCropAndScaleOption = .scaleFill
            let expressionStart = ProcessInfo.processInfo.systemUptime
            try VNImageRequestHandler(cgImage: expressionImage, orientation: .up).perform([request])
            let expressionMS = (ProcessInfo.processInfo.systemUptime - expressionStart) * 1000
            // highest confidence observation wins
            guard let predictions = request.results as? [VNClassificationObservation],
                  let best = predictions.max(by: { $0.confidence < $1.confidence }) else {
                throw PipelineError.unsupportedOutput("ExpressionClassifier")
            }
            outputs.append(AttributeResult(attribute: "expression", model: isOpenCV ? "OpenCV-ProgressiveTeacher-MobileFaceNet" : "ExpressionClassifier", label: best.identifier,
                                           score: best.confidence, requestMilliseconds: expressionMS))
            var scores: [String: Double] = [:]
            for observation in predictions { scores[observation.identifier] = Double(observation.confidence) }
            expressionScores = scores
        }
        var record = InferenceRecord(timestamp: Date(), policy: policy.rawValue,
            faceDetectionMilliseconds: detectionMS,
            pipelineMilliseconds: (ProcessInfo.processInfo.systemUptime - start) * 1000, results: outputs)
        record.estimatedAge = age
        record.calibratedCategoryScore = isStudent
        record.categoryProbabilities = ["female": female, "male": male]
        record.expressionScores = expressionScores
        record.faceDetectionConfidence = Double(face.confidence)
        // interval only for the student model, the radius comes from its metadata plus the calibration file
        if isStudent,
           let metadata = attributes.modelDescription.metadata[.creatorDefinedKey] as? [String: String],
           let radiusText = metadata["age_interval_radius_years"], let originalRadius = Double(radiusText),
           let coverageText = metadata["target_coverage"], let coverage = Double(coverageText) {
            let conservative = UserDefaults.standard.string(forKey: "ageIntervalPolicy") != "original"
            guard !conservative || conservativeAgeRadius != nil else { throw PipelineError.missingModel("Age interval calibration") }
            let radius = conservative ? (conservativeAgeRadius ?? originalRadius) : originalRadius
            guard radius.isFinite, radius >= 0, coverage.isFinite, (0...1).contains(coverage) else {
                throw PipelineError.unsupportedOutput("Age calibration metadata")
            }
            record.ageIntervalMethod = conservative ? "calibration-strata-max-v1" : "pooled-calibration-v1"
            // clamp to the possible age range
            record.ageIntervalLow = max(0, age - radius)
            record.ageIntervalHigh = min(116, age + radius)
            record.intervalTargetCoverage = coverage
        }
        // the crop mode is recorded so runs can be told apart later
        record.cropMode = directChip ? "utk-chip-direct-resize-128-v2" : "vision-five-point-similarity-128"
        record.modelSet = (isStudent ? "student-small-cropped-A-epoch16/coreml-fp32/preprocess-v2" : "fastface-large-128@15240f03/coreml-fp32")
            + (includeExpression ? (cachedExpressionName == "OpenCVExpression" ? " + opencv-expression-v1" : " + legacy-expression") : " (age/gender only)")
        return record
    }

    // pushes the stored fixtures through the converted model and compares them with the Python values
    func verifyStudentConversion() throws -> String {
        struct Fixture: Decodable { let filename: String; let age: Double; let probabilities: [Double] }
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
        guard let url = Bundle.main.url(forResource: "StudentAttributes", withExtension: "mlmodelc"),
              let manifest = Bundle.main.url(forResource: "expected", withExtension: "json", subdirectory: "ConversionFixtures") else {
            throw PipelineError.missingModel("Student conversion fixtures")
        }
        let model = try MLModel(contentsOf: url, configuration: configuration)
        let fixtures = try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: manifest))
        guard !fixtures.isEmpty, let constraint = model.modelDescription.inputDescriptionsByName["face"]?.imageConstraint else {
            throw PipelineError.invalidImage
        }
        var maxAge = 0.0, maxProbability = 0.0
        for fixture in fixtures {
            guard let imageURL = Bundle.main.url(forResource: fixture.filename, withExtension: nil, subdirectory: "ConversionFixtures"),
                  let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw PipelineError.invalidImage }
            let value = try MLFeatureValue(cgImage: image, constraint: constraint, options: nil)
            let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["face": value]))
            guard let age = output.featureValue(for: "estimated_age")?.multiArrayValue,
                  let probabilities = output.featureValue(for: "gender_probabilities")?.multiArrayValue,
                  age.count == 1, probabilities.count == 2, fixture.probabilities.count == 2 else {
                throw PipelineError.unsupportedOutput("Student parity output")
            }
            // largest deviation over all fixtures decides pass or fail
            let ageError = abs(age[0].doubleValue-fixture.age)
            let probabilityError = max(abs(probabilities[0].doubleValue-fixture.probabilities[0]),
                                       abs(probabilities[1].doubleValue-fixture.probabilities[1]))
            guard ageError.isFinite, probabilityError.isFinite else { throw PipelineError.unsupportedOutput("Nonfinite parity result") }
            maxAge = max(maxAge, ageError); maxProbability = max(maxProbability, probabilityError)
        }
        return String(format: "%@: %d identical-input checks. Maximum age difference %.6f years; category probability difference %.6f. This checks conversion, not recognition accuracy.",
            maxAge < 0.01 && maxProbability < 0.001 ? "PASS" : "FAIL", fixtures.count, maxAge, maxProbability)
    }

    // same parity check for the expression model, compared per class label
    func verifyExpressionConversion() throws -> String {
        struct Fixture: Decodable { let filename: String; let probabilities: [Double] }
        let labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuOnly
        guard let url = Bundle.main.url(forResource: "OpenCVExpression", withExtension: "mlmodelc"),
              let manifest = Bundle.main.url(forResource: "expression-expected", withExtension: "json", subdirectory: "ConversionFixtures") else {
            throw PipelineError.missingModel("Expression fixtures")
        }
        let model = try MLModel(contentsOf: url, configuration: configuration)
        let fixtures = try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: manifest))
        guard !fixtures.isEmpty, let constraint = model.modelDescription.inputDescriptionsByName["face"]?.imageConstraint else {
            throw PipelineError.invalidImage
        }
        var maximum = 0.0
        for fixture in fixtures {
            guard let url = Bundle.main.url(forResource: fixture.filename, withExtension: nil, subdirectory: "ConversionFixtures"),
                  let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
                  fixture.probabilities.count == labels.count else { throw PipelineError.invalidImage }
            let value = try MLFeatureValue(cgImage: image, constraint: constraint, options: nil)
            let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["face": value]))
            // probabilities come back as a label dictionary, so walk the label list
            guard let probabilities = output.featureValue(for: "classLabel_probs")?.dictionaryValue else {
                throw PipelineError.unsupportedOutput("Expression parity")
            }
            for (i,label) in labels.enumerated() {
                guard let value = probabilities[AnyHashable(label)]?.doubleValue, value.isFinite else {
                    throw PipelineError.unsupportedOutput("Expression probability")
                }
                maximum = max(maximum, abs(value-fixture.probabilities[i]))
            }
        }
        return String(format: "%@: %d expression conversion checks; maximum probability difference %.6f. Not recognition accuracy.",
            maximum < 0.001 ? "PASS" : "FAIL", fixtures.count, maximum)
    }
}
