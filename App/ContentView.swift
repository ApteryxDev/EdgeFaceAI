import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

// wrapper so a JSON blob can go through .fileExporter
struct JSONExport: FileDocument {
    // only json, the exporter picks the extension from this
    static var readableContentTypes: [UTType] { [.json] }
    var data: Data
    init(data: Data) { self.data = data }
    // importing is not used but FileDocument requires it
    init(configuration: ReadConfiguration) throws { data = configuration.file.regularFileContents ?? Data() }
    // hands the bytes over unchanged, no conversion on the way out
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}

// which input source the form shows, only one section is visible at a time
enum InputMode: String { case live, photo, faces, utk, internet }

struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var camera = LiveCamera()
    // toggle for the landmark overlay, applies to preview and photo view
    @State private var showOverlay = true
    // model choices survive a restart, otherwise every launch resets them
    @AppStorage("attributeModel") private var attributeModel = "student"
    @AppStorage("expressionModel") private var expressionModel = "opencv"
    @AppStorage("ageIntervalPolicy") private var ageIntervalPolicy = "conservative"
    // default FACES study, the UTK one is a second instance with its own folder
    @StateObject private var facesStudy = FACESStudy()
    @StateObject private var utkStudy = FACESStudy(resourceFolder: "UTKDevelopment", datasetTitle: "UTKFace",
        datasetDescription: "40 randomly selected images · balanced adult/elderly and male/female categories",
        expectedCount: 40)
    @State private var inputMode = InputMode.photo
    @State private var selection: PhotosPickerItem?
    @State private var imageData: Data?
    @State private var policy = ComputePolicy.all
    // kept results only, capped at 500 in keep()
    @State private var records: [InferenceRecord] = []
    @State private var photoResult: InferenceRecord?
    @State private var message = "Select a single-face photo."
    @State private var busy = false
    @State private var internetRunning = false
    // result text of the two conversion self-checks
    @State private var conversionCheck = ""
    @State private var showExport = false

    // locks the pickers while something is running, otherwise the model changes mid-analysis
    private var locked: Bool { busy || internetRunning || camera.running || camera.starting || facesStudy.running || utkStudy.running }

    private func hasResource(_ name: String, in folder: String) -> Bool {
        Bundle.main.url(forResource: name, withExtension: "json", subdirectory: folder) != nil
    }

    var body: some View {
        NavigationStack {
            Form {
                // top block: source, models, interval policy, compute policy, self-checks
                Section {
                    Picker("Input", selection: $inputMode) {
                        Text("Camera").tag(InputMode.live)
                        Text("Photos").tag(InputMode.photo)
                        if !facesStudy.examples.isEmpty { Text("FACES 72").tag(InputMode.faces) }
                        if !utkStudy.examples.isEmpty { Text("UTK 40").tag(InputMode.utk) }
                        if hasResource("manifest", in: "InternetCandidates") { Text("On device / Web").tag(InputMode.internet) }
                    }
                    .pickerStyle(.menu)
                    .disabled(locked)
                    // attribute model is the one that outputs age and gender category
                    Picker("Age and category model", selection: $attributeModel) {
                        Text("My trained model").tag("student")
                        Text("FastFace baseline").tag("fastface")
                    }.disabled(locked)
                    // expression runs as a separate second model
                    Picker("Expression model", selection: $expressionModel) {
                        Text("OpenCV candidate").tag("opencv")
                        if Bundle.main.url(forResource: "ExpressionClassifier", withExtension: "mlmodelc") != nil {
                            Text("Legacy baseline").tag("legacy")
                        }
                    }.disabled(locked)
                    // interval policy only affects the student model, the baseline has no calibration
                    Picker("Age interval", selection: $ageIntervalPolicy) {
                        Text("Conservative ±16.5 years").tag("conservative")
                        Text("Original ±8.8 years").tag("original")
                    }.disabled(locked)
                    // Allowed compute units; this setting does not prove which processor executes each layer.
                    Picker("Compute policy", selection: $policy) {
                        ForEach(ComputePolicy.allCases) { Text($0.rawValue).tag($0) }
                    }.disabled(locked)
                    Toggle("Face outlines & points", isOn: $showOverlay)
                    // runs the fixtures through the converted model and compares against the Python values
                    checkButton("Check student model conversion") { try await camera.verifyStudentConversion() }
                    checkButton("Check expression model conversion") { try await camera.verifyExpressionConversion() }
                    if !conversionCheck.isEmpty { Text(conversionCheck).font(.caption).textSelection(.enabled) }
                }
                // live section only exists in live mode, tearing it down stops the preview
                if inputMode == .live {
                    Section("Live camera") {
                        CameraPreview(camera: camera, showOverlay: showOverlay)
                            .frame(height: 300)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                            .overlay(alignment: .bottomLeading) {
                                if showOverlay, let live = camera.latest { OverlayHUD(record: live) }
                            }
                            .accessibilityLabel("Live camera preview")
                        HStack {
                            // one button that switches between start and stop depending on state
                            if camera.running || camera.starting {
                                Button("Stop camera", role: .destructive) { camera.stop() }
                            } else {
                                Button("Start camera") { Task { await camera.start(policy: policy) } }
                            }
                            Spacer()
                            // flipping is only allowed while stopped, the session gets rebuilt
                            Button(camera.usingFrontCamera ? "Use rear camera" : "Use front camera") { camera.flip() }
                                .disabled(camera.running || camera.starting)
                        }
                        .buttonStyle(.borderless)
                        if camera.starting { ProgressView("Starting camera…") }
                        Text(camera.message).font(.footnote)
                        // deep link into the app settings when the permission was denied
                        if camera.permissionDenied, let settings = URL(string: UIApplication.openSettingsURLString) {
                            Link("Open camera settings", destination: settings)
                        }
                        Text("Live results update as processing allows. Stop the camera before changing lenses.")
                            .font(.caption).foregroundStyle(.secondary)
                        if let result = camera.latest {
                            resultRows(result)
                            Button("Keep this result for export") { keep(result) }
                                .disabled(records.count >= 500)
                        }
                    }
                } else if inputMode == .photo {
                    Section("Photo analysis") {
                        PhotosPicker("Select photo", selection: $selection, matching: .images)
                            .disabled(busy)
                        // preview with overlay as soon as the data is there
                        if let data = imageData, let image = UIImage(data: data) {
                            AnnotatedPhoto(image: image, overlay: showOverlay ? camera.photoOverlay : nil,
                                           stats: showOverlay ? photoResult : nil)
                                .frame(maxHeight: 300)
                        }
                        if let result = photoResult { resultRows(result) }
                        Button("Analyze photo on device") { Task { await analyze() } }
                            .disabled(imageData == nil || locked)
                        if busy { ProgressView("Analyzing…") }
                        Text(message).font(.footnote)
                    }
                }
                // the three gallery/study sections, each rendered by its own view
                if inputMode == .internet { InternetAgeGallery(camera: camera, policy: policy, showOverlay: showOverlay, running: $internetRunning) }
                if inputMode == .utk {
                    FACESGallery(study: utkStudy, camera: camera, policy: policy, showOverlay: showOverlay)
                }
                if inputMode == .faces {
                    FACESGallery(study: facesStudy, camera: camera, policy: policy, showOverlay: showOverlay)
                }
                // footer: disclaimer plus export button and the note about in-memory history
                Section {
                    Text("Research prototype: model categories can be wrong and do not establish gender identity or emotional state. Age is an estimate; adult/elderly uses a 65-year boundary.")
                        .font(.footnote).foregroundStyle(.secondary)
                    Button("Export saved results (\(records.count))") { showExport = true }
                        .disabled(records.isEmpty || busy)
                    Text("Live frames are processed locally and are not recorded. Only results you keep are exported. Save the JSON before closing; history is held in memory (maximum 500 results).")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("EdgeFace AI")
        }
        .onAppear {
            if Bundle.main.url(forResource: "ExpressionClassifier", withExtension: "mlmodelc") == nil {
                expressionModel = "opencv"
            }
        }
        // switching the source shuts down whatever the previous source had running
        .onChange(of: inputMode) { _, mode in
            if mode != .live { camera.stop() }
            if mode != .faces { facesStudy.cancel() }
            if mode != .utk { utkStudy.cancel() }
        }
        // overlay toggle is pushed into the camera object, it decides whether landmarks are computed
        .onChange(of: showOverlay) { _, enabled in camera.setOverlayEnabled(enabled) }
        .onChange(of: policy) { _, newValue in camera.setPolicy(newValue) }
        // background kills the studies, an inactive scene only pauses the camera
        .onChange(of: scenePhase) { _, phase in
            if phase == .background { facesStudy.cancel(); utkStudy.cancel() }
            if phase == .background || (phase == .inactive && camera.running) {
                camera.stop(message: "Camera paused. Tap Start camera to resume.")
            }
        }
        // leaving the view has to stop everything, otherwise the session keeps running
        .onDisappear { camera.stop(); facesStudy.cancel(); utkStudy.cancel() }
        // loads the picked photo as raw data, the guard drops late results from an older pick
        .onChange(of: selection) { _, newValue in
            imageData = nil
            photoResult = nil
            camera.clearPhotoOverlay()
            Task {
                do {
                    let data = try await newValue?.loadTransferable(type: Data.self)
                    guard selection == newValue else { return }
                    imageData = data
                    message = data == nil ? "No image loaded." : "Photo ready."
                } catch {
                    guard selection == newValue else { return }
                    message = error.localizedDescription
                }
            }
        }
        // export dialog for the kept results
        .fileExporter(isPresented: $showExport, document: JSONExport(data: exportData), contentType: .json,
                      defaultFilename: "edgeface-session") { result in
            if case .failure(let error) = result { message = error.localizedDescription }
        }
    }

    // both conversion checks run the same way, only the call behind them differs
    private func checkButton(_ title: String, _ check: @escaping () async throws -> String) -> some View {
        Button(title) {
            busy = true; conversionCheck = "Checking…"
            Task {
                do { conversionCheck = try await check() }
                catch { conversionCheck = error.localizedDescription }
                busy = false
            }
        }.disabled(locked)
    }

    // builds the result block, age first, then the interval, then one row per attribute
    @ViewBuilder private func resultRows(_ record: InferenceRecord) -> some View {
        if let age = record.estimatedAge {
            Text(String(format: "Estimated age: %.1f years", age)).font(.headline)
        }
        // interval is only present for the student model with a calibration file
        if let low = record.ageIntervalLow, let high = record.ageIntervalHigh {
            Text(String(format: "Age interval: %.1f–%.1f years", low, high)).font(.subheadline)
            Text("80% calibration target on UTKFace. A wider interval does not improve the age estimate; live-camera coverage is unverified.")
                .font(.caption).foregroundStyle(.secondary)
        }
        // one row per attribute, the score format depends on whether it is a calibrated probability
        ForEach(record.results) { result in
            VStack(alignment: .leading) {
                Text("\(result.attribute): \(result.label)").font(.headline)
                if let score = result.score {
                    // the student model outputs a calibrated probability, the baseline a raw softmax
                    Text(String(format: result.attribute == "gender-category" && record.calibratedCategoryScore == true ? "Calibrated category score %.3f" : "Raw score %.3f", score))
                        .font(.caption).foregroundStyle(.secondary)
                }
                // age and gender share one request, expression has its own timing
                Text(String(format: "%@ %.1f ms", result.attribute != "expression" ? "Shared age/gender request" : "Request", result.requestMilliseconds))
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        Text(String(format: "Pipeline %.1f ms · %@", record.pipelineMilliseconds, record.policy)).font(.caption)
        Text(record.timestamp, style: .time).font(.caption).foregroundStyle(.secondary)
        InferenceStatsView(record: record)
    }

    // pretty printed and sorted so the JSON stays diffable by hand
    private var exportData: Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return (try? encoder.encode(records)) ?? Data()
    }

    // appends a result, the 500 cap is checked here again
    @MainActor private func keep(_ result: InferenceRecord) {
        guard records.count < 500 else { return }
        records.append(result)
    }

    // photo path: analyze, show the result, keep it, the defer resets busy in every case
    @MainActor private func analyze() async {
        guard !locked, let data = imageData else { return }
        let selectedPhoto = selection
        busy = true
        photoResult = nil
        defer { busy = false }
        do {
            let record = try await camera.analyzePhoto(data: data, policy: policy)
            guard selection == selectedPhoto else { return }
            photoResult = record
            keep(record)
            message = "Result saved locally. Export before closing."
        } catch { message = error.localizedDescription }
    }
}

// one entry of the on-device/web manifest, reference labels are optional
private struct InternetAgeExample: Decodable, Identifiable {
    let id: String
    let filename: String
    let source: String
    let originalFilename: String?
    let preprocessing: String?
    let referenceAge: Double?
    let referenceCategory: String?
    let note: String?
    // reference labels of the final 20 image run, missing in older manifests
    let referenceExpression: String?
    let plannedAgeGroup: String?
    let finalId: String?
}

// output row per image, inference stays nil when the run failed
private struct InternetAgeOutput: Codable {
    let filename: String
    var referenceAge: Double?
    var collectionSource: String
    var originalFilename: String?
    var preprocessing: String?
    var referenceCategory: String?
    var inference: InferenceRecord?
    var error: String?
    var referenceExpression: String?
    var plannedAgeGroup: String?
    var finalId: String?
}

// gallery for the web-sourced candidates, runs them one by one and stores the outputs
private struct InternetAgeGallery: View {
    @ObservedObject var camera: LiveCamera
    let policy: ComputePolicy
    let showOverlay: Bool
    @State private var examples: [InternetAgeExample] = []
    @State private var outputs: [InternetAgeOutput] = []
    @Binding var running: Bool
    @State private var analysisTask: Task<Void, Never>?
    @Environment(\.scenePhase) private var scenePhase
    @State private var export = false
    @State private var message = ""

    var body: some View {
        Section("On device / Web · \(examples.count) photos") {
            Text("Reference labels come from the image list. Ages of stock images are unknown. Age ranges use UTKFace calibration. A failed image counts as wrong in the statistics.")
                .font(.footnote)
            Button("Analyze all \(examples.count) on device") {
                guard !running else { return }
                running = true
                analysisTask = Task { await run() }
            }
                .disabled(running || examples.isEmpty)
            Button("Export On device / Web results") { export = true }.disabled(outputs.isEmpty || running)
            Text(message).font(.caption)
            if !outputs.isEmpty {
                GalleryTallyView(tally: GalleryTally(outputs: outputs), total: examples.count)
            }
            ForEach(examples) { example in
                InternetCard(example: example,
                             output: outputs.first(where: { $0.filename == example.filename }),
                             showOverlay: showOverlay)
            }
        }
        // manifest is bundled, a decode error ends up in the message line
        .task {
            do {
                guard let url = Bundle.main.url(forResource: "manifest", withExtension: "json", subdirectory: "InternetCandidates") else { throw PipelineError.missingModel("Internet gallery manifest") }
                examples = try JSONDecoder().decode([InternetAgeExample].self, from: Data(contentsOf: url))
            } catch { message = error.localizedDescription }
        }
        .onDisappear { analysisTask?.cancel() }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background { analysisTask?.cancel() }
        }
        .fileExporter(isPresented: $export, document: JSONExport(data: galleryExportData), contentType: .json, defaultFilename: "on-device-web-results") { result in
            if case .failure(let error) = result { message = error.localizedDescription }
        }
    }

    private var galleryExportData: Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return (try? encoder.encode(outputs)) ?? Data()
    }

    // loop over every image, failed ones keep their error row and continue
    @MainActor private func run() async {
        running = true; outputs = []
        defer { running = false; analysisTask = nil }
        camera.stop()
        for example in examples {
            if Task.isCancelled { break }
            message = "Analyzing \(outputs.count + 1) of \(examples.count)"
            var output = InternetAgeOutput(filename: example.filename, referenceAge: example.referenceAge,
                collectionSource: example.source, originalFilename: example.originalFilename,
                preprocessing: example.preprocessing, referenceCategory: example.referenceCategory)
            output.referenceExpression = example.referenceExpression
            output.plannedAgeGroup = example.plannedAgeGroup
            output.finalId = example.finalId
            do {
                guard let url = Bundle.main.url(forResource: example.filename, withExtension: nil, subdirectory: "InternetCandidates") else { throw PipelineError.invalidImage }
                // Manual target-face crops still use Vision alignment; only UTK training chips bypass it.
                var result = try await camera.analyzeDevelopmentImage(data: Data(contentsOf: url), policy: policy, includeExpression: true, isPrecropped: false)
                try Task.checkCancellation()
                result.source = "on-device-web-gallery/" + (example.preprocessing ?? "original") + "/" + example.source
                output.inference = result
            } catch is CancellationError { break }
            catch { output.error = error.localizedDescription }
            outputs.append(output)
        }
        message = (Task.isCancelled ? "Stopped. Partial results: " : "Finished: ") + "\(outputs.filter { $0.inference != nil }.count) successful; \(outputs.filter { $0.error != nil }.count) failed."
    }
}

// one picture of the gallery with its own landmark overlay and comparison rows
private struct InternetCard: View {
    let example: InternetAgeExample
    let output: InternetAgeOutput?
    let showOverlay: Bool
    @State private var overlay: FaceOverlayFrame?

    private var imageURL: URL? {
        Bundle.main.url(forResource: example.filename, withExtension: nil, subdirectory: "InternetCandidates")
    }

    // predicted label in the manifest spelling, nil when the image failed
    private func predicted(_ attribute: String) -> String? {
        guard let label = output?.inference?.results.first(where: { $0.attribute == attribute })?.label else { return nil }
        return attribute == "expression" ? canonicalExpression(label) : label.lowercased()
    }

    // prediction next to the reference with a tick or a cross
    private func comparison(_ title: String, _ attribute: String, _ reference: String?) -> String {
        let value = predicted(attribute) ?? "no result"
        guard let reference else { return title + ": " + value }
        let expected = reference.lowercased()
        return title + ": " + value + " · reference " + expected + (value == expected ? " ✓" : " ✗")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let url = imageURL, let image = UIImage(contentsOfFile: url.path) {
                AnnotatedPhoto(image: image, overlay: showOverlay ? overlay : nil,
                               stats: showOverlay ? output?.inference : nil)
                    .frame(maxHeight: 320)
            }
            Text((example.finalId ?? example.id) + " · " + example.source).font(.caption)
            if let age = example.referenceAge {
                Text(String(format: "Reference age: %.0f years", age)).font(.caption)
            } else {
                Text("Reference age: unknown").font(.caption)
            }
            if let output { resultBlock(output) }
        }
        // landmarks are computed once per card and only while the overlay is switched on
        .task(id: showOverlay) {
            guard showOverlay, overlay == nil, let url = imageURL else { return }
            let detected = try? await Task.detached(priority: .utility) {
                try FaceLandmarkDetector.detect(data: Data(contentsOf: url))
            }.value
            overlay = detected
        }
    }

    @ViewBuilder private func resultBlock(_ output: InternetAgeOutput) -> some View {
        if let record = output.inference {
            Text(comparison("Age group", "age-group", example.plannedAgeGroup)).font(.caption)
            Text(comparison("Category", "gender-category", example.referenceCategory)).font(.caption)
            Text(comparison("Expression", "expression", example.referenceExpression)).font(.caption)
            InferenceStatsView(record: record, referenceAge: example.referenceAge)
        } else {
            Text(output.error ?? "Pending").font(.caption)
        }
    }
}

// running agreement counts over the gallery, failed images stay in the denominators
private struct GalleryTally {
    var processed = 0
    var failed = 0
    var groupTotal = 0
    var groupCorrect = 0
    var categoryTotal = 0
    var categoryCorrect = 0
    var expressionTotal = 0
    var expressionCorrect = 0
    var ageErrors: [Double] = []
    var intervalTotal = 0
    var intervalCovered = 0
    var pipeline: [Double] = []
    var expressionConfidence: [Double] = []

    init(outputs: [InternetAgeOutput]) {
        for output in outputs {
            processed += 1
            let record = output.inference
            if record == nil { failed += 1 }
            let group = record?.results.first(where: { $0.attribute == "age-group" })?.label.lowercased()
            let category = record?.results.first(where: { $0.attribute == "gender-category" })?.label.lowercased()
            let expression = record?.results.first(where: { $0.attribute == "expression" })
            if let reference = output.plannedAgeGroup {
                groupTotal += 1
                if group == reference.lowercased() { groupCorrect += 1 }
            }
            if let reference = output.referenceCategory {
                categoryTotal += 1
                if category == reference.lowercased() { categoryCorrect += 1 }
            }
            if let reference = output.referenceExpression {
                expressionTotal += 1
                if let label = expression?.label, canonicalExpression(label) == reference.lowercased() { expressionCorrect += 1 }
            }
            guard let done = record else { continue }
            pipeline.append(done.pipelineMilliseconds)
            if let score = expression?.score { expressionConfidence.append(Double(score)) }
            if let reference = output.referenceAge, let age = done.estimatedAge {
                ageErrors.append(abs(age - reference))
                if let low = done.ageIntervalLow, let high = done.ageIntervalHigh {
                    intervalTotal += 1
                    if reference >= low && reference <= high { intervalCovered += 1 }
                }
            }
        }
    }

    var meanAbsoluteAgeError: Double? {
        ageErrors.isEmpty ? nil : ageErrors.reduce(0, +) / Double(ageErrors.count)
    }
    var withinFiveYears: Int { ageErrors.filter { $0 <= 5 }.count }
    var medianPipeline: Double? {
        guard !pipeline.isEmpty else { return nil }
        let sorted = pipeline.sorted()
        let middle = sorted.count / 2
        return sorted.count % 2 == 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
    }
    var meanExpressionConfidence: Double? {
        expressionConfidence.isEmpty ? nil : expressionConfidence.reduce(0, +) / Double(expressionConfidence.count)
    }
}

// statistics block above the gallery pictures
private struct GalleryTallyView: View {
    let tally: GalleryTally
    let total: Int

    private func share(_ correct: Int, _ count: Int) -> Double {
        count == 0 ? 0 : Double(correct) / Double(count)
    }

    // the figures that are not percentages go into one text block
    private var details: String {
        var lines: [String] = []
        if let error = tally.meanAbsoluteAgeError {
            lines.append(String(format: "Mean age error %.1f years on %d images with a known age", error, tally.ageErrors.count))
            lines.append("Within five years: \(tally.withinFiveYears) of \(tally.ageErrors.count)")
        }
        if tally.intervalTotal > 0 {
            lines.append("Age inside the interval: \(tally.intervalCovered) of \(tally.intervalTotal)")
        }
        if let confidence = tally.meanExpressionConfidence {
            lines.append(String(format: "Mean score of the chosen expression %.1f%%", confidence * 100))
        }
        if let median = tally.medianPipeline {
            lines.append(String(format: "Median pipeline time %.1f ms", median))
        }
        return lines.joined(separator: "\n")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Statistics: \(tally.processed) of \(total) processed, \(tally.failed) failed")
                .font(.subheadline.bold())
            if tally.groupTotal > 0 {
                ProbabilityBar(label: "Age group \(tally.groupCorrect)/\(tally.groupTotal)",
                               value: share(tally.groupCorrect, tally.groupTotal), labelWidth: 130)
            }
            if tally.categoryTotal > 0 {
                ProbabilityBar(label: "Category \(tally.categoryCorrect)/\(tally.categoryTotal)",
                               value: share(tally.categoryCorrect, tally.categoryTotal), labelWidth: 130)
            }
            if tally.expressionTotal > 0 {
                ProbabilityBar(label: "Expression \(tally.expressionCorrect)/\(tally.expressionTotal)",
                               value: share(tally.expressionCorrect, tally.expressionTotal), labelWidth: 130)
            }
            if !details.isEmpty { Text(details).font(.caption) }
            Text("Agreement with the reference labels of these images only. Not a general accuracy claim.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }
}

// maps the expression labels of the models onto the manifest spelling
func canonicalExpression(_ value: String) -> String {
    switch value.lowercased() {
    case "happiness": return "happy"
    case "sadness": return "sad"
    case "anger": return "angry"
    default: return value.lowercased()
    }
}

// one labelled bar for a value between 0 and 1
struct ProbabilityBar: View {
    let label: String
    let value: Double
    var highlight = false
    var labelWidth: CGFloat = 84

    var body: some View {
        HStack(spacing: 8) {
            Text(label).font(.caption).frame(width: labelWidth, alignment: .leading)
            ProgressView(value: min(max(value, 0), 1)).tint(highlight ? Color.green : Color.blue)
            Text(String(format: "%.1f%%", min(max(value, 0), 1) * 100)).font(.caption.monospacedDigit())
                .frame(width: 56, alignment: .trailing)
        }
    }
}

// age estimate as a dot on a 0 to 100 year line, the interval as a band, the reference as an orange mark
struct AgeIntervalBar: View {
    let age: Double
    let low: Double?
    let high: Double?
    var reference: Double? = nil

    private func fraction(_ value: Double) -> CGFloat {
        CGFloat(min(max(value, 0), 100) / 100)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.gray.opacity(0.2)).frame(height: 8)
                    if let low, let high, high >= low {
                        Capsule().fill(Color.blue.opacity(0.35))
                            .frame(width: max(2, geometry.size.width * (fraction(high) - fraction(low))), height: 8)
                            .offset(x: geometry.size.width * fraction(low))
                    }
                    Circle().fill(Color.blue).frame(width: 10, height: 10)
                        .offset(x: geometry.size.width * fraction(age) - 5)
                    if let reference {
                        Rectangle().fill(Color.orange).frame(width: 2, height: 14)
                            .offset(x: geometry.size.width * fraction(reference) - 1)
                    }
                }
                .frame(height: 14)
            }
            .frame(height: 14)
            HStack {
                Text("0")
                Spacer()
                Text("50")
                Spacer()
                Text("100 years")
            }
            .font(.caption2).foregroundStyle(.secondary)
        }
    }
}

// confidence figures of one result: detection, age interval, category probabilities, expression scores
struct InferenceStatsView: View {
    let record: InferenceRecord
    var referenceAge: Double? = nil

    private var rankedExpressions: [(key: String, value: Double)] {
        (record.expressionScores ?? [:]).sorted { $0.value > $1.value }
    }

    private var timing: String {
        String(format: "Face detection %.1f ms · whole pipeline %.1f ms", record.faceDetectionMilliseconds, record.pipelineMilliseconds)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Model confidence").font(.subheadline.bold())
            if let confidence = record.faceDetectionConfidence {
                ProbabilityBar(label: "Face found", value: confidence)
            }
            if let age = record.estimatedAge { ageBlock(age) }
            if let probabilities = record.categoryProbabilities { categoryBlock(probabilities) }
            if !rankedExpressions.isEmpty { expressionBlock }
            Text(timing).font(.caption2).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private func ageBlock(_ age: Double) -> some View {
        Text(String(format: "Age estimate %.1f years", age)).font(.caption)
        AgeIntervalBar(age: age, low: record.ageIntervalLow, high: record.ageIntervalHigh, reference: referenceAge)
        if let referenceAge {
            Text(String(format: "Orange mark: reference age %.0f, error %.1f years", referenceAge, abs(age - referenceAge)))
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private func categoryBlock(_ probabilities: [String: Double]) -> some View {
        let female = probabilities["female"] ?? 0
        let male = probabilities["male"] ?? 0
        Text(record.calibratedCategoryScore == true ? "Category probability, calibrated" : "Category score, raw")
            .font(.caption).foregroundStyle(.secondary)
        ProbabilityBar(label: "Female", value: female, highlight: female >= male)
        ProbabilityBar(label: "Male", value: male, highlight: male > female)
    }

    @ViewBuilder private var expressionBlock: some View {
        let ranked = rankedExpressions
        Text("Expression scores, top \(min(4, ranked.count)) of \(ranked.count)")
            .font(.caption).foregroundStyle(.secondary)
        ForEach(Array(ranked.prefix(4).enumerated()), id: \.offset) { pair in
            ProbabilityBar(label: canonicalExpression(pair.element.key).capitalized,
                           value: pair.element.value, highlight: pair.offset == 0)
        }
        if ranked.count > 1 {
            Text(String(format: "Lead over the second choice: %.1f points", (ranked[0].value - ranked[1].value) * 100))
                .font(.caption2).foregroundStyle(.secondary)
        }
    }
}

// small label drawn on the picture or the camera preview
struct OverlayHUD: View {
    let record: InferenceRecord

    private func ageText(_ age: Double) -> String {
        if let low = record.ageIntervalLow, let high = record.ageIntervalHigh {
            return String(format: "Age %.0f (%.0f to %.0f)", age, low, high)
        }
        return String(format: "Age %.0f", age)
    }

    private func attributeText(_ result: AttributeResult) -> String {
        let label = result.attribute == "expression" ? canonicalExpression(result.label) : result.label
        if let score = result.score {
            return String(format: "%@ %.0f%%", label, Double(score) * 100)
        }
        return label
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            if let age = record.estimatedAge { Text(ageText(age)) }
            ForEach(record.results) { result in
                Text(attributeText(result))
            }
            Text(String(format: "%.0f ms", record.pipelineMilliseconds))
        }
        .font(.caption2.monospacedDigit())
        .foregroundStyle(Color.white)
        .padding(6)
        .background(Color.black.opacity(0.6))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .padding(6)
    }
}
