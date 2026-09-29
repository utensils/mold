import SwiftUI
import Testing

@testable import MoldCompanion

/// DESIGN.md §6 rule 3: a label/value row stacks at accessibility sizes and
/// never clips. These pin the ends of the scale; what happens in between is
/// `RowAxis.for(_:)`'s decision.
struct RowAxisTests {
    @Test func everyStandardSizeStaysOnOneRow() {
        for size in [DynamicTypeSize.xSmall, .small, .medium, .large, .xLarge, .xxLarge] {
            #expect(RowAxis.for(size) == .horizontal, "\(size)")
        }
    }

    @Test func everyAccessibilitySizeStacks() {
        for size in DynamicTypeSize.allCases where size.isAccessibilitySize {
            #expect(RowAxis.for(size) == .vertical, "\(size)")
        }
    }

    @Test func phoneGenerateControlsStackBeforeAccessibilitySizes() {
        #expect(GenerateRow.stacks(at: .large) == false)
        #expect(GenerateRow.stacks(at: .xxLarge))
        #expect(GenerateRow.stacks(at: .xxxLarge))
        #expect(GenerateRow.stacks(at: .accessibility1))
        #expect(GenerateRow.stacks(at: .large, phone: true))
    }
}
