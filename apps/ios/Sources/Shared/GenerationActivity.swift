import ActivityKit
import AppIntents
import Foundation

/// The render Live Activity (DESIGN.md §5.7). The server has no push, so the
/// app updates it while it runs and from background refresh after; the state
/// stays well under ActivityKit's 4 KB -- the preview is a file in the App
/// Group, named here, never embedded.
nonisolated struct GenerationActivityAttributes: ActivityAttributes {
    nonisolated struct ContentState: Codable, Hashable {
        enum Phase: String, Codable, Hashable { case running, finished, failed }

        var phase: Phase
        /// "Adding detail", "Finished on workstation", or why it failed.
        var sentence: String
        /// The mono truth beside it: "denoise 18/28".
        var figure: String?
        var step: Int?
        var total: Int?
        /// When it should be done, for `Text(timerInterval:)`.
        var endsAt: Date?
        /// A JPEG in the App Group (`AppGroup.activityPreviews`), by name.
        var preview: String?
        /// More batches waiting behind this one.
        var waiting: Int
        /// The first print a finished render made, for View.
        var print: String? = nil

        var fraction: Double? {
            guard let step, let total, total > 0 else { return nil }
            return min(1, Double(step) / Double(total))
        }
    }

    var prompt: String
    var machine: String
    var clientBatchId: String
    /// The machine's id here, for View: the finished print, or the Queue
    /// while it runs.
    var host: String

    func link(for state: ContentState) -> DeepLink {
        if state.phase == .finished, let print = state.print, let host = UUID(uuidString: host) {
            return .print(host: host, filename: print)
        }
        return .queue(job: nil)
    }
}

/// Stop, from the Lock Screen or the Dynamic Island. A `LiveActivityIntent`
/// runs in the APP's process, which installs `handler`; the widget extension
/// only draws the button and never networks.
struct StopRenderIntent: LiveActivityIntent {
    static let title: LocalizedStringResource = "Stop Render"
    static let isDiscoverable = false

    @MainActor static var handler: (@MainActor @Sendable (String) async -> Void)?

    @Parameter(title: "Batch") var clientBatchId: String

    init() {}
    init(clientBatchId: String) { self.clientBatchId = clientBatchId }

    func perform() async throws -> some IntentResult {
        let handler = await Self.handler
        await handler?(clientBatchId)
        return .result()
    }
}
