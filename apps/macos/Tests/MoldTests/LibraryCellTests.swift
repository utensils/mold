import Foundation
import Testing

/// The grid tile's hover behaviour.
struct LibraryCellTests {
    /// A tile carries no hover tooltip: the prompt already reads in the
    /// inspector, and a paragraph floating over the next tile was the most
    /// distracting thing in the grid. VoiceOver still hears it through the
    /// cell's accessibility label.
    @Test func aTileHasNoPromptTooltip() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent()
                .appending(path: "Sources/Mold/Library/LibraryCell.swift"),
            encoding: .utf8)
        #expect(source.count > 500, "the source file was not found")
        let body = try #require(source.components(separatedBy: "var body: some View").dropFirst().first)
        let firstView = body.components(separatedBy: "@ViewBuilder").first ?? body
        #expect(!firstView.contains(".help("))
        #expect(firstView.contains(".accessibilityLabel("))
    }
}
