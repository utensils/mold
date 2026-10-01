import Foundation
import Testing
import UIKit
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

@MainActor struct ActivityCardLayoutTests {
    @Test func theThreeNormalRowsFitInsideTheSystemHeightIncludingPadding() {
        let traits = UITraitCollection(preferredContentSizeCategory: .large)
        let brand = UIFont.preferredFont(forTextStyle: .caption2, compatibleWith: traits).lineHeight
        let status = UIFont.preferredFont(forTextStyle: .subheadline, compatibleWith: traits).lineHeight
        let prompt = UIFont.preferredFont(forTextStyle: .caption1, compatibleWith: traits).lineHeight
        let header = max(ActivityCardLayout.previewSide, brand + status + 2 * prompt + 2 * ActivityCardLayout.titleSpacing)
        let progress = ActivityCardLayout.progressHeight + ActivityCardLayout.progressSpacing + brand
        let rows = header + progress + brand + 2 * ActivityCardLayout.rowSpacing
        #expect(rows <= ActivityCardLayout.contentHeight)
        #expect(ActivityCardLayout.contentHeight + 2 * ActivityCardLayout.inset == ActivityCardLayout.maximumHeight)
        #expect(ActivityCardLayout.stopSide >= 44)
    }
}
