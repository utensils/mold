import Foundation
import Observation

/// Feedback for the latest Generate press, separate from the canvas following
/// an earlier job. Late replies cannot overwrite a newer press's status.
@MainActor
@Observable
final class GenerationSubmissionFeedback {
    enum Phase { case preparing, submitting, accepted, refused }
    private(set) var id = UUID()
    private(set) var message: String?
    private(set) var phase: Phase = .accepted
    var isPending: Bool { message != nil && (phase == .preparing || phase == .submitting) }
    var isPreparing: Bool { message != nil && phase == .preparing }

    @discardableResult
    func begin(_ message: String, phase: Phase = .preparing) -> UUID {
        id = UUID()
        self.message = message
        self.phase = phase
        return id
    }

    func update(_ message: String, phase: Phase, for id: UUID) {
        guard self.id == id else { return }
        self.message = message
        self.phase = phase
    }

    func dismiss() { id = UUID(); message = nil }
}
