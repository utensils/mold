import Foundation
import MoldClientTesting
import Testing

@testable import MoldClient

/// `docs/contracts/gallery-changes-v1.json`, the feed's golden scenarios: the
/// same pages the Rust and TypeScript implementations replay must leave this
/// replica holding exactly what they hold.
struct GalleryChangesContractTests {
    struct Scenario: @unchecked Sendable {
        let name: String
        let initial: [String]
        let since: String?
        let pages: [Data]
        let expected: [String: [String: Any]]
        let finalCursor: String
    }

    static func scenarios() throws -> [Scenario] {
        let root = try #require(RepoFixtures.repoRoot)
        let data = try Data(contentsOf: root.appending(path: "docs/contracts/gallery-changes-v1.json"))
        let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["protocol_version"] as? Int == 1)
        let rows = try #require(object["scenarios"] as? [[String: Any]])
        return try rows.map { row in
            Scenario(name: row["name"] as? String ?? "?",
                     initial: row["initial"] as? [String] ?? [],
                     since: row["since"] as? String,
                     pages: try (row["pages"] as? [Any] ?? []).map { try JSONSerialization.data(withJSONObject: $0) },
                     expected: row["expected"] as? [String: [String: Any]] ?? [:],
                     finalCursor: row["final_cursor"] as? String ?? "")
        }
    }

    @Test func everyScenarioReplaysToItsExpectedState() async throws {
        let scenarios = try Self.scenarios()
        #expect(scenarios.count >= 4)
        for scenario in scenarios {
            let pages = try scenario.pages.map { data in
                HTTPBackend.preservingRawMetadata(try MoldJSON.decoder.decode(GalleryChangesPage.self, from: data),
                                                  from: data)
            }
            let asked = Counter()
            let expectedSince = [scenario.since] + pages.dropLast().map { Optional($0.cursor) }
            let backend = FakeBackend()
            backend.stub("galleryChanges(since:limit:)") { args in
                let index = await asked.next() - 1
                #expect(args[0] as? String == expectedSince[index], "\(scenario.name) page \(index)")
                return pages[index]
            }
            let replica = GalleryReplica(hostID: UUID())
            let instance = pages.first?.instanceId
            await replica.seed(scenario.initial.map { PrintFixtures.print($0) },
                               cursor: scenario.since, instanceID: scenario.since == nil ? nil : instance)
            try await replica.pull(using: backend)

            let rows = await replica.rows
            #expect(Set(rows.keys) == Set(scenario.expected.keys), "\(scenario.name)")
            for (name, fields) in scenario.expected {
                let row = try #require(rows[name], "\(scenario.name): \(name)")
                for (field, value) in fields {
                    switch field {
                    case "timestamp": #expect(row.timestamp == (value as? NSNumber)?.uint64Value, "\(scenario.name)")
                    case "favorite": #expect(row.favorite == value as? Bool, "\(scenario.name)")
                    case "title": #expect(row.title == value as? String, "\(scenario.name)")
                    case "tags": #expect(row.tags == value as? [String], "\(scenario.name)")
                    case "trashed_at": #expect(row.trashedAt == (value as? NSNumber)?.uint64Value, "\(scenario.name)")
                    case "purge_at": #expect(row.purgeAt == (value as? NSNumber)?.uint64Value, "\(scenario.name)")
                    default: Issue.record("\(scenario.name): the replay does not check `\(field)` yet")
                    }
                }
            }
            #expect(await replica.cursor == scenario.finalCursor, "\(scenario.name)")
            #expect(await asked.next() - 1 == pages.count, "\(scenario.name) asked every page")
        }
    }
}
