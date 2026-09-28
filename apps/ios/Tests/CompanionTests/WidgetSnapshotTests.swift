import Foundation
import Testing

@testable import MoldCompanion

/// The file the widgets read: it round-trips, filters by machine and
/// favourites, and says the queue in words.
struct WidgetSnapshotTests {
    private let a = UUID(), b = UUID()

    private var snapshot: WidgetSnapshot {
        WidgetSnapshot(updated: Date(timeIntervalSince1970: 5), prints: [
            .init(host: a, machine: "workstation", filename: "1.png", title: "owl", image: "1.jpg", favourite: true,
                  kind: .picture, made: Date(timeIntervalSince1970: 3)),
            .init(host: b, machine: "laptop", filename: "2.mp4", title: "sea", image: "2.jpg", favourite: false,
                  kind: .clip, made: Date(timeIntervalSince1970: 2)),
        ], machines: [.init(id: a, name: "workstation"), .init(id: b, name: "laptop")],
        rendering: 2, held: 1, waiting: 0, progress: 0.5)
    }

    @Test func itRoundTripsThroughTheAppGroupFile() throws {
        let url = FileManager.default.temporaryDirectory.appending(path: "snapshot-\(UUID()).json")
        try snapshot.save(to: url)
        #expect(WidgetSnapshot.load(from: url) == snapshot)
    }

    @Test func aMissingFileIsAnEmptySnapshotNotACrash() {
        #expect(WidgetSnapshot.load(from: URL(fileURLWithPath: "/nonexistent/snapshot.json")) == .empty)
    }

    @Test func aConfiguredWidgetShowsOneMachineOrFavourites() {
        #expect(snapshot.prints(machine: b, favouritesOnly: false).map(\.filename) == ["2.mp4"])
        #expect(snapshot.prints(machine: nil, favouritesOnly: true).map(\.filename) == ["1.png"])
        #expect(snapshot.prints(machine: nil, favouritesOnly: false).count == 2)
    }

    @Test func theQueueReadsAsASentence() {
        #expect(snapshot.queueSummary == "2 rendering · 1 held")
        #expect(WidgetSnapshot.empty.queueSummary == "Nothing waiting")
    }

    @Test func aPrintLinksToItselfInTheApp() {
        #expect(snapshot.prints[0].link == .print(host: a, filename: "1.png"))
    }
}
