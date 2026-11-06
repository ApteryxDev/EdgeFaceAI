import Foundation

/// label agreement against the dataset, failures count in the denominators of the categorical totals
struct DatasetComparison: Codable {
    let selected: Int
    let completed: Int
    let succeeded: Int
    let failed: Int
    let unprocessed: Int
    let genderCorrect: Int
    let ageGroupCorrect: Int
    let expressionCorrect: Int
    let expressionEvaluable: Int
    let numericAgePredictions: Int
    let meanAbsoluteAgeError: Double?
    let withinFiveYears: Int
    let intervalCovered: Int?
    let intervalEvaluable: Int?

    init(run: FACESRun) {
        selected = run.observations.count
        // only rows that actually ran go into the counters
        let attempted = run.observations.filter { ["success","failed"].contains($0.status) }
        completed = attempted.count
        succeeded = attempted.filter { $0.status == "success" }.count
        failed = attempted.filter { $0.status == "failed" }.count
        unprocessed = selected - completed
        // helper to pull one predicted label, nil on failed rows
        func label(_ row: FACESObservation, _ attribute: String) -> String? {
            guard row.status == "success" else { return nil }
            return row.inference?.results.first { $0.attribute == attribute }?.label.lowercased()
        }
        genderCorrect = attempted.filter { label($0,"gender-category") == $0.example.genderCategory }.count
        ageGroupCorrect = attempted.filter { label($0,"age-group") == $0.example.expectedAgeGroup }.count
        // rows without a usable expression reference are skipped
        let emotionRows = attempted.filter { !$0.example.expression.isEmpty && $0.example.expression != "unlabelled" }
        expressionEvaluable = emotionRows.count
        // OpenCV writes happiness/sadness/anger, the manifest writes happy/sad/angry
        func canonical(_ value: String?) -> String? {
            switch value { case "happiness": return "happy"; case "sadness": return "sad"; case "anger": return "angry"; default: return value }
        }
        expressionCorrect = emotionRows.filter { canonical(label($0,"expression")) == $0.example.expression }.count
        // age error only for successful rows that have both a reference and a finite prediction
        let errors = attempted.compactMap { row -> Double? in
            guard row.status == "success", let reference = row.example.referenceAge,
                  let predicted = row.inference?.estimatedAge, predicted.isFinite, reference.isFinite else { return nil }
            return abs(predicted-reference)
        }
        numericAgePredictions = errors.count
        meanAbsoluteAgeError = errors.isEmpty ? nil : errors.reduce(0,+)/Double(errors.count)
        withinFiveYears = errors.filter { $0 <= 5 }.count
        // interval coverage: does the interval contain the reference age
        let coverage = attempted.compactMap { row -> Bool? in
            guard row.status == "success", let reference = row.example.referenceAge,
                  let low = row.inference?.ageIntervalLow, let high = row.inference?.ageIntervalHigh,
                  reference.isFinite, low.isFinite, high.isFinite, low <= high else { return nil }
            return (low...high).contains(reference)
        }
        intervalEvaluable = coverage.count
        intervalCovered = coverage.filter { $0 }.count
    }
}
