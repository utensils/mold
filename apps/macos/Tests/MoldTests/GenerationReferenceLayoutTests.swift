import SwiftUI
import Testing
@testable import Mold

@MainActor
struct GenerationReferenceLayoutTests {
    @Test(arguments: [1, 2, 12])
    func referencePreviewAndActionsFitNarrowInspector(count: Int) {
        // A 260 pt inspector subtracts its 24 pt outer padding and 18 pt
        // disclosure inset. Filename and actions occupy separate lines.
        let available: CGFloat = 260 - 24 - 18
        let preview = GenerationReferenceLayout.previewSize(count: count)
        let actions = 3 * GenerationReferenceLayout.actionSize + 2 * GenerationReferenceLayout.actionGap
        #expect(preview >= 80)
        #expect(preview + GenerationReferenceLayout.columnGap + actions <= available)
        #expect(GenerationReferenceLayout.actionSize >= 24)
    }

    @Test func singleReferenceReceivesLargerPreview() {
        #expect(GenerationReferenceLayout.previewSize(count: 1) > GenerationReferenceLayout.previewSize(count: 2))
    }
}
