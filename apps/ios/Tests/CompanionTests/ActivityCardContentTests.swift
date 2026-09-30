import Foundation
import Testing
@testable import MoldCompanion

struct ActivityCardContentTests {
    private func state(_ phase: GenerationActivityAttributes.ContentState.Phase = .running,
                       figure: String? = "denoise 18/28 · hal9000-7680") -> GenerationActivityAttributes.ContentState {
        .init(phase: phase, sentence: "Adding detail", figure: figure, step: 18, total: 28,
              endsAt: .now.addingTimeInterval(20), preview: nil, waiting: 2)
    }

    @Test func machineAndProgressHaveSeparateLabelsWithoutRepeatingTheMachine() {
        let content = ActivityCardContent(state: state(), machine: "hal9000-7680", isStale: false)
        #expect(content.detail == "denoise 18/28")
        #expect(content.machine == "hal9000-7680")
        #expect(content.progress == 18.0 / 28.0)
        #expect(content.canStop)
        #expect(ActivityCardContent(state: state(figure: "hal9000-7680"),
                                    machine: "hal9000-7680", isStale: false).detail == nil)
    }

    @Test func staleAndTerminalCardsDoNotPretendProgressIsLive() {
        let stale = ActivityCardContent(state: state(), machine: "m", isStale: true)
        #expect(stale.title == "Open Mold Studio to refresh")
        #expect(stale.progress == nil)
        #expect(stale.detail == nil)
        #expect(stale.canStop)
        for phase in [GenerationActivityAttributes.ContentState.Phase.finished, .failed] {
            let terminal = ActivityCardContent(state: state(phase), machine: "m", isStale: false)
            #expect(terminal.progress == nil)
            #expect(!terminal.canStop)
        }
    }

    @Test func unknownProgressAndFutureFiguresRemainHonest() {
        var waiting = state(figure: "Waiting for memory")
        waiting.step = nil
        waiting.total = nil
        let content = ActivityCardContent(state: waiting, machine: "m", isStale: false)
        #expect(content.progress == nil)
        #expect(content.detail == "Waiting for memory")
    }
}
