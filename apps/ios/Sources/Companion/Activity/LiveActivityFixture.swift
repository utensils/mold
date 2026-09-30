#if DEBUG
import ActivityKit
import Foundation

/// Lock Screen UAT without a server or a render; absent from distribution builds.
enum LiveActivityFixture {
    static func startIfRequested() {
        let arguments = ProcessInfo.processInfo.arguments
        guard let index = arguments.firstIndex(of: "--live-activity-fixture"),
              arguments.indices.contains(index + 1) else { return }
        let mode = arguments[index + 1]
        Task {
            for activity in Activity<GenerationActivityAttributes>.activities
                where activity.attributes.clientBatchId == "live-activity-fixture" {
                await activity.end(nil, dismissalPolicy: .immediate)
            }
            guard mode != "clear" else { return }
            let phase: GenerationActivityAttributes.ContentState.Phase = switch mode {
            case "finished": .finished
            case "failed": .failed
            default: .running
            }
            let state = GenerationActivityAttributes.ContentState(phase: phase,
                sentence: mode == "waiting" ? "Working on it" : phase == .finished ? "Finished on hal9000-7680" : phase == .failed ? "Couldn't finish the render" : "Adding detail",
                figure: mode == "waiting" ? "hal9000-7680" : "denoise 18/28 · hal9000-7680",
                step: mode == "waiting" ? nil : 18, total: mode == "waiting" ? nil : 28,
                endsAt: mode == "waiting" ? nil : .now.addingTimeInterval(600), preview: "live-activity-fixture.jpg", waiting: 2,
                print: phase == .finished ? "live-activity-fixture.png" : nil)
            let attributes = GenerationActivityAttributes(prompt: "A sunlit glass pavilion beside a quiet lake",
                machine: "hal9000-7680", clientBatchId: "live-activity-fixture", host: UUID().uuidString)
            _ = try? Activity.request(attributes: attributes,
                content: ActivityContent(state: state, staleDate: mode == "stale" ? .now.addingTimeInterval(5) : .now.addingTimeInterval(300)),
                pushType: nil)
        }
    }
}
#endif
