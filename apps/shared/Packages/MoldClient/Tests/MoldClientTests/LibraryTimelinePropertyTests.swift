import Foundation
import Testing

@testable import MoldClient

/// Whatever a run of random deltas does, a patch that is accepted must equal
/// the full merge of the same rows.
struct LibraryTimelinePropertyTests {
    struct Seeded: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state >> 1
        }
    }

    let hosts = [UUID(), UUID(), UUID()]

    func make(_ random: inout Seeded) -> LibraryEntry {
        let host = hosts.randomElement(using: &random)!
        return PrintFixtures.entry("p\(Int.random(in: 0..<30, using: &random)).png", host: host,
                                   hostName: host.uuidString,
                                   timestamp: UInt64.random(in: 1...40, using: &random),
                                   favorite: Bool.random(using: &random),
                                   bytes: [100, 200].randomElement(using: &random))
    }

    func ordered(_ rows: [PrintID: LibraryEntry]) -> [LibraryEntry] {
        hosts.flatMap { host in
            rows.values.filter { $0.hostID == host }.sorted { $0.print.filename < $1.print.filename }
        }
    }

    func step(_ rows: [PrintID: LibraryEntry], _ random: inout Seeded) -> ([LibraryEntry], [PrintID]) {
        var upserted: [LibraryEntry] = []
        var removed: [PrintID] = []
        func untouched(_ id: PrintID) -> Bool { !removed.contains(id) && !upserted.contains { $0.id == id } }
        for _ in 0..<Int.random(in: 1...3, using: &random) {
            switch Int.random(in: 0..<3, using: &random) {
            case 0:
                let entry = make(&random)
                if rows[entry.id] == nil, untouched(entry.id) { upserted.append(entry) }
            case 1:
                if let id = rows.keys.randomElement(using: &random), untouched(id) { removed.append(id) }
            default:
                if let old = rows.values.randomElement(using: &random), untouched(old.id) {
                    upserted.append(PrintFixtures.entry(
                        old.print.filename, host: old.hostID, hostName: old.hostName,
                        timestamp: old.print.timestamp, favorite: !old.print.isFavorite,
                        bytes: old.print.sizeBytes))
                }
            }
        }
        return (upserted, removed)
    }

    @Test func anAcceptedPatchIsAlwaysTheFullMerge() {
        var accepted = 0
        for seed in 1...40 {
            var random = Seeded(state: UInt64(seed))
            var rows: [PrintID: LibraryEntry] = [:]
            for _ in 0..<20 {
                let entry = make(&random)
                rows[entry.id] = entry
            }
            var timeline = LibraryTimeline()
            timeline.rebuild(ordered(rows), localHost: hosts[0])
            for _ in 0..<30 {
                let (upserted, removed) = step(rows, &random)
                for entry in upserted { rows[entry.id] = entry }
                for id in removed { rows[id] = nil }
                var full = LibraryTimeline()
                full.rebuild(ordered(rows), localHost: hosts[0])
                if timeline.patch(upserted: upserted, removed: removed) {
                    accepted += 1
                } else {
                    timeline.rebuild(ordered(rows), localHost: hosts[0])
                }
                #expect(timeline.tiles == full.tiles, "seed \(seed)")
                #expect(timeline.tiles.flatMap(\.everyCopy).allSatisfy { timeline.tile(containing: $0.id) != nil })
            }
        }
        // The property is only worth something if patches are actually taken.
        #expect(accepted > 50)
    }
}
