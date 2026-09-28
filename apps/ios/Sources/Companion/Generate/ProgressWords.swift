import Foundation
import MoldClient

/// What a render is doing, as a sentence a first-timer can read, and the
/// technical truth beside it in mono (docs/design/README.md §1): "Adding
/// detail — about 12s left" over "denoise 18/28". The server names its stages
/// for engineers ("Loading VAE (GPU)", "VAE decode"); this says them for
/// people, and falls back to the server's own words for any it does not know.
enum ProgressWords {
    /// The sentence. `remaining` is the time left when a rate is known.
    static func sentence(_ progress: JobProgress?, position: Int? = nil, remaining: Duration? = nil) -> String {
        if let position, position > 0 {
            return position == 1 ? String(localized: "Waiting — next in line")
                                 : String(localized: "Waiting — \(position - 1) ahead")
        }
        guard let progress else { return String(localized: "Getting ready…") }
        let phase = plain(progress)
        guard let remaining, remaining > .seconds(1) else { return phase }
        let left = remaining.formatted(.units(allowed: [.minutes, .seconds], width: .narrow, maximumUnitCount: 2))
        return String(localized: "\(phase) — about \(left) left")
    }

    /// The mono figure: the stage the machine named, and the step count.
    static func figure(_ progress: JobProgress?) -> String? {
        guard let progress else { return nil }
        let stage = progress.stage.map { isDenoising($0) ? "denoise" : $0.lowercased() }
        if let step = progress.step, let total = progress.total, total > 0 {
            return "\(stage ?? "step") \(step)/\(total)"
        }
        return stage
    }

    /// What VoiceOver reads for the bar: "18 of 28, about 12 seconds left".
    static func spoken(_ progress: JobProgress?, remaining: Duration? = nil) -> String {
        guard let step = progress?.step, let total = progress?.total else { return String(localized: "In progress") }
        var words = String(localized: "\(step) of \(total)")
        if let remaining, remaining > .seconds(1) {
            words += ", " + String(localized: "about \(remaining.formatted(.units(allowed: [.minutes, .seconds], width: .wide))) left")
        }
        return words
    }

    static func plain(_ progress: JobProgress) -> String {
        let stage = (progress.stage ?? "").lowercased()
        if isDenoising(stage) || (stage.isEmpty && progress.step != nil) { return String(localized: "Adding detail") }
        if stage.contains("encoding prompt") || stage.contains("encoder") && !stage.contains("load") {
            return String(localized: "Reading your prompt")
        }
        if stage.contains("encoding source") { return String(localized: "Studying your picture") }
        if stage.hasPrefix("loading") || stage.hasPrefix("reloading") || stage.hasPrefix("selecting") {
            return String(localized: "Getting the model ready")
        }
        if stage.contains("decode") || stage.contains("decoding") { return String(localized: "Finishing the picture") }
        if stage.contains("upscal") { return String(localized: "Making it bigger") }
        if stage.contains("background") { return String(localized: "Removing the background") }
        if stage.contains("texture") || stage.contains("paint") { return String(localized: "Painting the surface") }
        guard let first = progress.stage?.first else { return String(localized: "Working on it") }
        return first.uppercased() + (progress.stage ?? "").dropFirst()
    }

    private static func isDenoising(_ stage: String) -> Bool {
        let lowered = stage.lowercased()
        return lowered.contains("denois") || lowered.contains("sampling") || lowered == "step"
    }
}

/// Time left, from how fast the steps are going: measured, never guessed --
/// nothing until two steps have been seen.
struct StepRate {
    private var first: (step: Int, at: Date)?

    mutating func remaining(_ progress: JobProgress?, now: Date = .now) -> Duration? {
        guard let step = progress?.step, let total = progress?.total, total > step else { return nil }
        guard let first, step > first.step else {
            if first == nil || (first.map { step < $0.step } ?? false) { self.first = (step, now) }
            return nil
        }
        let perStep = now.timeIntervalSince(first.at) / Double(step - first.step)
        return .seconds(perStep * Double(total - step))
    }

    mutating func reset() { first = nil }
}
