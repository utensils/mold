import Foundation
import Testing
@testable import MoldClient

struct SourceMediaDisplayTests {
    private func row(_ id: String, name: String, size: Int, timestamp: UInt64 = 1_000, outputSize: Int = 100) -> SourceMediaCatalog.Row {
        let owner = PrintFixtures.entry(id + ".mp4", host: UUID(), timestamp: timestamp, bytes: outputSize)
        return .init(owner: owner, member: .init(memberId: id, role: "source_image", displayName: name, sizeBytes: size), route: nil, instance: nil)
    }
    private func ordered(_ rows: [SourceMediaCatalog.Row], sort: LibrarySort) -> [String] {
        let allowed = Dictionary(uniqueKeysWithValues: rows.map { ($0.owner.id, $0.authority) })
        return SourceMediaDisplay.filter(SourceMediaCatalog.group(rows), allowed: allowed, kind: .all, search: "", sort: sort)
            .map { $0.usages[0].member.memberId }
    }
    @Test func sizeAndNameSortUseTheSourceRatherThanItsOwningPrint() {
        let first = row("a-output", name: "z-source.png", size: 10, outputSize: 10_000)
        let second = row("z-output", name: "a-source.png", size: 1_000, outputSize: 10)
        #expect(ordered([first, second], sort: .largest) == ["z-output", "a-output"])
        #expect(ordered([first, second], sort: .name) == ["z-output", "a-output"])
    }
    @Test func dateSortUsesOwningPrintTimestampsAndTiesAreStableAcrossInputOrder() {
        let first = row("a", name: "same.png", size: 10, timestamp: 100)
        let second = row("b", name: "same.png", size: 10, timestamp: 200)
        #expect(ordered([first, second], sort: .newest) == ["b", "a"])
        #expect(ordered([second, first], sort: .oldest) == ["a", "b"])
        for sort in [LibrarySort.name, .largest] {
            #expect(ordered([first, second], sort: sort) == ordered([second, first], sort: sort))
        }
    }
    @Test func deduplicatedSourceUsesNewestOrOldestMatchingOwnerForDateSort() {
        let first = row("a", name: "same.png", size: 10, timestamp: 100)
        let newest = SourceMediaCatalog.Row(owner: PrintFixtures.entry("new.mp4", host: UUID(), timestamp: 300), member: first.member, route: nil, instance: nil)
        let shared = SourceMediaCatalog.Source(id: "shared", usages: [first, newest], searchKey: "")
        let other = row("b", name: "other.png", size: 20, timestamp: 200)
        let sources = [SourceMediaCatalog.Source(id: "other", usages: [other], searchKey: ""), shared]
        let allowed = Dictionary(uniqueKeysWithValues: [first, newest, other].map { ($0.owner.id, $0.authority) })
        #expect(SourceMediaDisplay.filter(sources, allowed: allowed, kind: .all, search: "", sort: .newest).map(\.id) == ["shared", "other"])
        #expect(SourceMediaDisplay.filter(sources.reversed(), allowed: allowed, kind: .all, search: "", sort: .oldest).map(\.id) == ["shared", "other"])
    }
}
