import Foundation
import Testing

@testable import MoldClient

// `Fixtures/history-hal9000.json` is a live `GET /api/history?limit=3` from
// hal9000, real millisecond stamps included -- the top row is the cancelled
// `not-a-real-model-probe` admission from the design's fact 4: history is
// recorded before dispatch, so a job that never ran still left a row.

private func live() throws -> HistoryListing {
    try MoldJSON.decoder.decode(
        HistoryListing.self, from: RepoFixtures.fixture("history-hal9000.json"))
}

@Test func aHistoryRowIdentifiesItselfByItsContent() throws {
    let listing = try live()
    #expect(listing.entries.count == 3)

    let probe = listing.entries[0]
    #expect(probe.prompt == "probe")
    #expect(probe.model == "not-a-real-model-probe")
    #expect(probe.usedAt == 1_789_614_459_994)

    let sibling = HistoryEntry(prompt: probe.prompt, model: probe.model, usedAt: probe.usedAt + 1)
    #expect(sibling.id != probe.id)
    #expect(probe.id == "1789614459994|not-a-real-model-probe|probe")
}

@Test func aHistoryRowsDateMatchesItsMillisecondStamp() throws {
    let probe = try live().entries[0]
    #expect(probe.usedAtDate.timeIntervalSince1970 == 1_789_614_459_994.0 / 1000)
}
