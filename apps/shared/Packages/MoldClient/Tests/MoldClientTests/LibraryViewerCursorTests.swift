import Testing
@testable import MoldClient

struct LibraryViewerCursorTests {
    @Test func removalFollowsTheVisibleOrder() {
        #expect(LibraryViewerCursor.afterRemoval("b", previous: ["a", "b", "c", "d"], remaining: ["d", "a"]) == "d")
        #expect(LibraryViewerCursor.afterRemoval("c", previous: ["a", "b", "c"], remaining: ["a", "b"]) == "b")
    }
    @Test func survivingCurrentAndEmptyLibrary() {
        #expect(LibraryViewerCursor.afterRemoval("b", previous: ["a", "b", "c"], remaining: ["b", "c"]) == "b")
        #expect(LibraryViewerCursor.afterRemoval("a", previous: ["a"], remaining: []) == nil)
    }
}

import Foundation

extension LibraryViewerCursorTests {
    @Test func aSurvivingRenamedMirrorKeepsTheSameMediaOpen() throws {
        let first = MoldHost(name: "a", baseURL: URL(string: "http://a")!)
        let second = MoldHost(name: "b", baseURL: URL(string: "http://b")!)
        func entry(_ name: String, on host: MoldHost) throws -> LibraryEntry {
            let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data("{\"filename\":\"\(name)\",\"metadata\":{},\"timestamp\":1}".utf8))
            return LibraryEntry(host: host, print: print)
        }
        var original = try entry("original.png", on: first)
        let mirror = try entry("renamed.png", on: second)
        original.copies = [mirror]
        let next = try entry("next.png", on: first)
        #expect(LibraryViewerCursor.afterRemoval(original.id, previous: [original, next], remaining: [mirror, next]) == mirror.id)
        let removed = try entry("removed.png", on: first)
        #expect(LibraryViewerCursor.afterRemoval(removed.id, previous: [removed, original, next], remaining: [mirror, next]) == mirror.id)
    }
}
