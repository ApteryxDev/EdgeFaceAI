import Foundation

// one image of a development set plus its dataset labels
struct FACESExample: Codable, Identifiable {
    let id: String
    let filename: String
    let personID: String
    let ageGroup: String
    let genderCategory: String
    let expression: String
    let pictureSet: String
    let sha256: String
    // exact age only for the UTK sample, not for FACES
    var referenceAge: Double? = nil
    // set for the UTK instance so both studies share the same model
    var resourceFolder: String? = nil
    // young and middle-aged both count as adult, otherwise the group is not evaluable
    var expectedAgeGroup: String { ["older", "elderly"].contains(ageGroup) ? "elderly" : "adult" }
    // bundle url, nil when the file is missing
    var imageURL: URL? {
        Bundle.main.url(forResource: filename, withExtension: nil, subdirectory: resourceFolder ?? "FACESDevelopment")
    }
}

// per-image run state, inference only filled on success
struct FACESObservation: Codable {
    let example: FACESExample
    var status: String = "not-run"
    var policy: String?
    var inference: InferenceRecord?
    var error: String?
}

// one run over a selection, the comparison is computed on export
struct FACESRun: Codable, Identifiable {
    let id: UUID
    let dataset: String
    let purpose: String
    let started: Date
    var updated: Date
    var observations: [FACESObservation]
    var comparison: DatasetComparison? = nil
}
