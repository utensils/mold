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

@Test func historySearchEscapesPromptTextAndPreservesHostPrefix() {
    let backend = HTTPBackend(host: MoldHost(name: "box", baseURL: URL(string: "http://box/mold")!, apiKey: "key"))
    let request = backend.request(backend.historyPath(limit: 50, query: "sea & sky + #1"))
    let query = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)?.queryItems
    #expect(query?.first { $0.name == "query" }?.value == "sea & sky + #1")
    #expect(request.url?.path == "/mold/api/history")
}

@Test func queuedMetadataHasFullInspectorSettingsWithoutAnOutputFile() throws {
    let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(#"{"prompt":"coast","model":"ltx","seed":42,"frames":49,"fps":24,"source_image_name":"source.png"}"#.utf8))
    let rows = PrintDetails.groups(for: metadata).flatMap(\.rows)
    #expect(rows.contains { $0.label == "Seed" && $0.value == "42" })
    #expect(rows.contains { $0.label == "Frames" && $0.value == "49" })
    #expect(rows.contains { $0.value.contains("source.png") })
}
