import Foundation
import Combine
import SwiftUI
import ImageIO
import UniformTypeIdentifiers

extension FACESExample {
    // downsampled thumbnail through ImageIO, avoids loading the full image into memory
    func thumbnail(maximum: Int) -> UIImage? {
        guard let url = imageURL, let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                  kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceCreateThumbnailWithTransform: true,
                  kCGImageSourceThumbnailMaxPixelSize: maximum
              ] as CFDictionary) else { return nil }
        return UIImage(cgImage: cg)
    }
}

// study driver for one dataset folder, runs the images sequentially and writes a run file after each one
@MainActor final class FACESStudy: ObservableObject {
    @Published private(set) var examples: [FACESExample] = []
    @Published private(set) var run: FACESRun?
    @Published private(set) var savedRuns: [FACESRun] = []
    @Published private(set) var running = false
    @Published private(set) var currentID: String?
    @Published private(set) var message = "Select an image or analyze all 72."
    @Published private(set) var persistenceMessage: String?
    let resourceFolder: String
    let datasetTitle: String
    let datasetDescription: String
    let expectedCount: Int
    // UTK chips are already cropped and have no expression labels
    var isUTK: Bool { resourceFolder == "UTKDevelopment" }
    private var task: Task<Void, Never>?

    init(resourceFolder: String = "FACESDevelopment", datasetTitle: String = "FACES",
         datasetDescription: String = "72 photos · 6 people · dataset reference labels", expectedCount: Int = 72) {
        self.resourceFolder = resourceFolder
        self.datasetTitle = datasetTitle
        self.datasetDescription = datasetDescription
        self.expectedCount = expectedCount
        self.message = "Select an image or analyze all \(expectedCount)."
        // manifest must match the expected count, unique ids and all files present
        do {
            guard let url = Bundle.main.url(forResource: "manifest", withExtension: "json", subdirectory: resourceFolder) else {
                throw FACESFailure.missing("\(datasetTitle) manifest")
            }
            examples = try JSONDecoder().decode([FACESExample].self, from: Data(contentsOf: url))
            guard examples.count == expectedCount, Set(examples.map(\.id)).count == expectedCount,
                  examples.allSatisfy({ $0.imageURL != nil }) else { throw FACESFailure.missing("\(expectedCount) bundled images") }
        } catch { message = error.localizedDescription; examples = [] }
        reloadHistory()
        // restore the most recent run so an interrupted session is not lost
        if let latest = savedRuns.first {
            run = latest
            message = "Restored saved development run. Export it or start a new run."
        }
    }

    // counters for the progress bar
    var completed: Int { run?.observations.filter { ["success","failed"].contains($0.status) }.count ?? 0 }
    var succeeded: Int { run?.observations.filter { $0.status == "success" }.count ?? 0 }
    var failed: Int { run?.observations.filter { $0.status == "failed" }.count ?? 0 }
    // looks up the observation of one image for the grid
    func observation(for example: FACESExample) -> FACESObservation? {
        run?.observations.first { $0.example.id == example.id }
    }
    func analyzeAll(camera: LiveCamera, policy: ComputePolicy) { start(examples, camera: camera, policy: policy) }
    func analyze(_ example: FACESExample, camera: LiveCamera, policy: ComputePolicy) {
        start([example], camera: camera, policy: policy)
    }
    // stops after the image that is currently running
    func cancel() {
        guard running else { return }
        task?.cancel()
        message = "Stopping after the current image…"
    }

    private func start(_ selected: [FACESExample], camera: LiveCamera, policy: ComputePolicy) {
        guard !running, !selected.isEmpty else { return }
        camera.stop()
        running = true
        persistenceMessage = nil
        let now = Date()
        // every run gets its own id and stays on disk, cancelled runs included
        run = FACESRun(id: UUID(), dataset: "\(datasetTitle): \(datasetDescription)",
            purpose: isUTK ? "development-only; possible FastFace training overlap; no expression reference labels"
                           : "development-only; not IU final device-collected 20-image sample",
            started: now, updated: now,
            observations: selected.map { FACESObservation(example: $0) })
        persist()
        task = Task { [weak self, camera] in
            guard let self else { return }
            // cleanup runs on every exit path, cancelled or not
            defer {
                self.running = false
                self.currentID = nil
                self.task = nil
                self.persist()
                self.reloadHistory()
            }
            for (index, example) in selected.enumerated() {
                if Task.isCancelled { break }
                self.currentID = example.id
                self.message = "Analyzing \(index+1) of \(selected.count): \(example.filename)"
                self.run?.observations[index].status = "running"
                self.run?.observations[index].policy = policy.rawValue
                do {
                    guard let url = example.imageURL else { throw FACESFailure.missing(example.filename) }
                    // reading the file off the main actor
                    let data = try await Task.detached(priority: .userInitiated) { try Data(contentsOf: url) }.value
                    try Task.checkCancellation()
                    // UTK chips skip alignment and expression
                    var record = try await camera.analyzeDevelopmentImage(data: data, policy: policy,
                        includeExpression: !self.isUTK, isPrecropped: self.isUTK)
                    try Task.checkCancellation()
                    record.source = self.isUTK ? "utk-development" : "faces-development"
                    self.run?.observations[index].inference = record
                    self.run?.observations[index].status = "success"
                } catch is CancellationError {
                    // cancelled row keeps its state, the loop breaks
                    self.run?.observations[index].status = "cancelled"
                    break
                } catch {
                    self.run?.observations[index].status = "failed"
                    self.run?.observations[index].error = error.localizedDescription
                }
                self.run?.updated = Date()
                self.persist()
            }
            self.message = Task.isCancelled ? "Stopped. Partial results saved." : "Finished: \(self.succeeded) successful, \(self.failed) failed."
        }
    }

    // loads an older run from the menu
    func loadSaved(_ saved: FACESRun) {
        guard !running else { return }
        run = saved
        message = "Loaded saved run from \(saved.started.formatted(date: .abbreviated, time: .shortened))."
    }

    // documents/<resource folder>, shared by reading and writing
    private var historyDirectory: URL? {
        try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent(resourceFolder, isDirectory: true)
    }

    // reads all run files back, newest first
    private func reloadHistory() {
        do {
            guard let directory = historyDirectory,
                  FileManager.default.fileExists(atPath: directory.path) else { return }
            let urls = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .iso8601
            savedRuns = urls.filter { $0.pathExtension == "json" }.compactMap {
                try? decoder.decode(FACESRun.self, from: Data(contentsOf: $0))
            }.sorted { $0.updated > $1.updated }
        } catch { persistenceMessage = "Could not read saved runs: \(error.localizedDescription)" }
    }

    // shared encoder settings for persist and export
    private var encoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    // export snapshot with the comparison filled in
    var exportData: Data {
        guard var snapshot = run else { return Data() }
        snapshot.comparison = DatasetComparison(run: snapshot)
        return (try? encoder.encode(snapshot)) ?? Data()
    }

    // writes the current run file, the comparison is refreshed before every write
    private func persist() {
        guard let run, let directory = historyDirectory else { return }
        self.run?.comparison = DatasetComparison(run: run)
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try exportData.write(to: directory.appendingPathComponent("run-\(run.id.uuidString).json"), options: .atomic)
        } catch {
            persistenceMessage = "Could not save this run: \(error.localizedDescription). Export the current results."
        }
    }
}

private enum FACESFailure: LocalizedError {
    case missing(String)
    var errorDescription: String? {
        switch self { case .missing(let name): return "Missing bundled resource: \(name)." }
    }
}

// gallery with age and expression filters plus a sheet per image
struct FACESGallery: View {
    @ObservedObject var study: FACESStudy
    @ObservedObject var camera: LiveCamera
    let policy: ComputePolicy
    let showOverlay: Bool
    @State private var ageFilter = "all"
    @State private var expressionFilter = "all"
    @State private var selected: FACESExample?
    @State private var export = false

    // filtered grid content
    var visible: [FACESExample] {
        study.examples.filter {
            (ageFilter == "all" || $0.ageGroup == ageFilter) &&
            (expressionFilter == "all" || $0.expression == expressionFilter)
        }
    }

    var body: some View {
        Section("\(study.datasetTitle) development gallery") {
            Text(study.datasetDescription).font(.subheadline)
            Text(study.isUTK ? "Random balanced sample; age and category labels are dataset estimates. Training overlap is possible. No expression labels."
                             : "All six expressions are included. Age references are groups, not exact ages. This is a development set.")
                .font(.caption).foregroundStyle(.secondary)
            // start/stop on the left, export on the right
            HStack {
                if study.running {
                    Button("Stop analysis", role: .destructive) { study.cancel() }
                } else {
                    Button("Analyze all \(study.expectedCount)") { study.analyzeAll(camera: camera, policy: policy) }
                        .disabled(study.examples.isEmpty)
                }
                Spacer()
                Button("Export \(study.datasetTitle) run") { export = true }
                    .disabled(study.run == nil || study.running)
            }.buttonStyle(.borderless)
            // progress bar follows the completed rows
            if study.running {
                ProgressView(value: Double(study.completed), total: Double(study.run?.observations.count ?? study.expectedCount))
            }
            if let run = study.run { DatasetComparisonView(comparison: DatasetComparison(run: run)) }
            // saved runs are listed with time, count and a short id
            if !study.savedRuns.isEmpty {
                Menu("Previous saved runs (\(study.savedRuns.count))") {
                    ForEach(study.savedRuns) { saved in
                        Button("\(saved.started.formatted(date: .abbreviated, time: .shortened)) · \(saved.observations.count) images · \(saved.id.uuidString.prefix(6))") {
                            study.loadSaved(saved)
                        }
                    }
                }.disabled(study.running)
            }
            Text(study.message).font(.caption)
            if let warning = study.persistenceMessage { Text(warning).font(.caption).foregroundStyle(.orange) }
            // age filter, the young/middle-aged options only exist for FACES
            Picker("Age group", selection: $ageFilter) {
                Text("All ages").tag("all")
                if study.isUTK { Text("Adult").tag("adult") }
                else { Text("Young").tag("young"); Text("Middle-aged").tag("middle-aged") }
                Text("Older").tag("older")
            }
            if !study.isUTK {
                Picker("Expression", selection: $expressionFilter) {
                    Text("All expressions").tag("all")
                    ForEach(["happy","sad","neutral","angry","disgust","fear"], id: \.self) { Text($0.capitalized).tag($0) }
                }
            }
            Text("Showing \(visible.count) of \(study.examples.count)").font(.caption)
            // adaptive grid, tapping a card opens the detail sheet
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 260))], spacing: 12) {
                ForEach(visible) { example in
                    Button { selected = example } label: {
                        VStack(spacing: 4) {
                            DatasetPicture(example: example, showOverlay: showOverlay, maximumHeight: 250)
                            if let observation = study.observation(for: example) {
                                ImagePredictionStatistics(example: example, observation: observation)
                            } else {
                                Text("Not analyzed yet").font(.caption).foregroundStyle(.secondary)
                            }
                            Text("\(example.personID) · \(example.pictureSet)").font(.caption)
                            Text("\(example.ageGroup) · \(example.genderCategory)").font(.caption2)
                            if let age = example.referenceAge {
                                Text(String(format: "Reference age %.0f", age)).font(.caption2)
                            } else {
                                Text(example.expression.capitalized).font(.caption2)
                            }
                        }
                    }.buttonStyle(.plain)
                }
            }
            Text(study.isUTK ? "UTKFace research sample · fixed seed 20261108 · 10 images per age/category stratum"
                             : "FACES: Ebner, Riediger & Lindenberger (2010), Behavior Research Methods, 42, 351–362.")
                .font(.caption2).foregroundStyle(.secondary)
            Link("Dataset source and usage information", destination: URL(string: study.isUTK ? "https://susanqq.github.io/UTKFace/" : "https://faces.mpdl.mpg.de/")!)
        }
        // detail sheet is bound to the selected example
        .sheet(item: $selected) { example in
            FACESDetail(example: example, study: study, camera: camera, policy: policy, showOverlay: showOverlay)
        }
        .fileExporter(isPresented: $export, document: JSONExport(data: study.exportData), contentType: .json,
                      defaultFilename: "\(study.datasetTitle.lowercased())-development-run") { _ in }
    }
}

// sheet with the full image, the reference labels and the model output
private struct FACESDetail: View {
    let example: FACESExample
    @ObservedObject var study: FACESStudy
    @ObservedObject var camera: LiveCamera
    let policy: ComputePolicy
    let showOverlay: Bool
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                DatasetPicture(example: example, showOverlay: showOverlay, maximumHeight: 350)
                if let observation = study.observation(for: example) {
                    ImagePredictionStatistics(example: example, observation: observation)
                } else {
                    Text("Analyze this image to see its predictions here.").font(.caption)
                }
                // dataset reference block
                Section("Dataset reference") {
                    Text("Age group: \(example.ageGroup)")
                    Text("Gender category: \(example.genderCategory)")
                    Text("Expression: \(example.expression)")
                    Text("\(study.isUTK ? "Sample" : "Person") \(example.personID) · \(example.pictureSet)")
                    if let age = example.referenceAge {
                        Text(String(format: "Dataset age label: %.0f years", age))
                    } else { Text("Exact age unavailable").foregroundStyle(.secondary) }
                }
                Button("Analyze this image") { study.analyze(example, camera: camera, policy: policy) }
                    .disabled(study.running)
                // model output including the match/mismatch lines against the labels
                if let observation = study.observation(for: example) {
                    Section("Model output · \(observation.status)") {
                        if let record = observation.inference {
                            if let age = record.estimatedAge { Text(String(format: "Estimated age: %.1f years", age)) }
                            if let low = record.ageIntervalLow, let high = record.ageIntervalHigh {
                                Text(String(format: "Age interval: %.1f–%.1f years (UTK target 80%%)", low, high)).font(.caption)
                            }
                            ForEach(record.results) { result in Text("\(result.attribute): \(result.label)") }
                            // error only when there is an exact reference age
                            if let age = record.estimatedAge, let reference = example.referenceAge {
                                Text(String(format: "Absolute age error: %.1f years", abs(age-reference)))
                            }
                            if let predicted = record.results.first(where: { $0.attribute == "gender-category" }) {
                                Text(predicted.label.lowercased() == example.genderCategory ? "Gender-category label: match" : "Gender-category label: mismatch")
                            }
                            if let predicted = record.results.first(where: { $0.attribute == "age-group" }) {
                                Text(predicted.label.lowercased() == example.expectedAgeGroup ? "Age-group label: match" : "Age-group label: mismatch")
                            }
                            if study.isUTK { Text("Expression is not scored: no reference label.").font(.caption) }
                            Text(String(format: "Pipeline %.1f ms · %@", record.pipelineMilliseconds, record.policy)).font(.caption)
                        }
                        if let error = observation.error { Text(error).foregroundStyle(.orange) }
                    }
                }
            }
            .navigationTitle(example.id)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
    }
}

// compact comparison block, shown above the grid
private struct DatasetComparisonView: View {
    let comparison: DatasetComparison
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("Compared with dataset labels").font(.headline)
            Text("Completed \(comparison.completed)/\(comparison.selected) · failures \(comparison.failed)")
            // match counters only once something completed
            if comparison.completed > 0 {
                Text("Gender-category matches: \(comparison.genderCorrect)/\(comparison.completed)")
                Text("Adult/elderly matches: \(comparison.ageGroupCorrect)/\(comparison.completed)")
            }
            if let mae = comparison.meanAbsoluteAgeError {
                Text(String(format: "Age MAE: %.1f years (%d numeric predictions)", mae, comparison.numericAgePredictions))
                Text("Within 5 years: \(comparison.withinFiveYears)/\(comparison.numericAgePredictions)")
            }
            if let covered = comparison.intervalCovered, let total = comparison.intervalEvaluable, total > 0 {
                Text("Age interval covers reference: \(covered)/\(total)")
            }
            if comparison.expressionEvaluable > 0 {
                Text("Expression matches: \(comparison.expressionCorrect)/\(comparison.expressionEvaluable)")
            }
            Text("Categorical totals include failures; age error uses successful numeric predictions only.")
                .font(.caption2).foregroundStyle(.secondary)
        }.font(.subheadline)
    }
}

// own overlay state per card, otherwise old points stay on screen when the grid changes
private struct DatasetPicture: View {
    let example: FACESExample
    let showOverlay: Bool
    let maximumHeight: CGFloat
    @State private var overlay: FaceOverlayFrame?
    @State private var overlayMessage: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let image = example.thumbnail(maximum: 1100) {
                AnnotatedPhoto(image: image, overlay: showOverlay ? overlay : nil)
                    .frame(maxHeight: maximumHeight)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }
            if showOverlay, let overlayMessage { Text(overlayMessage).font(.caption2).foregroundStyle(.secondary) }
        }
        // landmarks are computed once per card, off the main actor
        .task(id: showOverlay) {
            guard showOverlay, overlay == nil, let url = example.imageURL else { return }
            do {
                let geometry = try await Task.detached(priority: .utility) {
                    try FaceLandmarkDetector.detect(data: Data(contentsOf: url))
                }.value
                guard !Task.isCancelled else { return }
                overlay = geometry
                overlayMessage = geometry.faces.isEmpty ? "No face landmarks detected" : nil
            } catch { overlayMessage = "Face overlay unavailable" }
        }
    }
}

// per-image statistics used in the grid and in the sheet
private struct ImagePredictionStatistics: View {
    let example: FACESExample
    let observation: FACESObservation
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            if let record = observation.inference {
                Text("Prediction statistics").font(.subheadline.bold())
                if let age = record.estimatedAge {
                    Text(String(format: "Estimated age: %.1f years", age))
                    if let reference = example.referenceAge {
                        Text(String(format: "Reference: %.0f · age error: %.1f years", reference, abs(age-reference)))
                    }
                }
                if let low = record.ageIntervalLow, let high = record.ageIntervalHigh {
                    Text(String(format: "Age interval: %.1f–%.1f years", low, high))
                    Text("80% target on UTKFace; coverage on this image is not guaranteed.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                // one row per attribute plus the score in percent
                ForEach(record.results) { result in
                    Text("\(result.attribute): \(result.label)")
                    if let score = result.score {
                        Text(String(format: result.attribute == "gender-category" && record.calibratedCategoryScore == true ? "Calibrated category score: %.1f%%" : "Raw model score: %.1f%%", Double(score)*100))
                    }
                }
                Text(String(format: "Processing: %.1f ms · %@", record.pipelineMilliseconds, record.policy))
                    .foregroundStyle(.secondary)
                Text(record.modelSet).font(.caption2).foregroundStyle(.secondary)
            } else if let error = observation.error {
                Text(error).foregroundStyle(.orange)
            } else { Text(observation.status) }
        }.font(.caption).frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 6)
    }
}
