import Testing

@testable import Mold

/// `StepSegments.reading` -- the pure split behind the denoise strip's
/// VoiceOver announcement (design S7). The old version folded "Step 3 of
/// 20" into the label, so VoiceOver re-read the whole sentence on every
/// step; the label now names the control once and the value is what moves.
struct StepSegmentsTests {
    @Test func aProgressStripNamesItselfOnceAndReReadsOnlyItsValue() {
        let reading = StepSegments.reading(done: 3, total: 20)

        #expect(reading.label == "Progress")
        #expect(reading.value == "Step 3 of 20")
    }
}
