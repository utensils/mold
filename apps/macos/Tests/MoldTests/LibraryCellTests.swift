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

// Older hosts omit `format`; the public kind contract still recognizes the file.
import MoldClient
import SwiftUI
@testable import Mold

@MainActor @Test func legacyClipKeepsItsPlaybackBadge() throws {
    let host = MoldHost(name: "fixture", baseURL: URL(string: "http://fixture")!)
    let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(#"{"filename":"clip.mp4","timestamp":1,"metadata":{}}"#.utf8))
    let cell = LibraryCell(entry: LibraryEntry(host: host, print: print), host: host, edge: 96,
                           isSelected: false, isLead: false, showsHostBadge: true)
    #expect(cell.mediaSymbol == "play.fill")
}
