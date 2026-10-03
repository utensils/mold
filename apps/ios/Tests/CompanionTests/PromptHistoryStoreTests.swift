import Foundation
import MoldClient
import MoldClientTesting
import Testing
@testable import MoldCompanion

@MainActor struct PromptHistoryStoreTests {
    @Test func unavailableHistoryIsDifferentFromEmptyHistory() async throws {
        let (_, hosts, fake) = try await QueueStoreTests.setUp()
        let host = hosts.hosts[0]
        let history = PromptHistoryStore(host: host, hosts: hosts)
        fake.stub("history(limit:query:)") { _ in
            throw MoldClientError.http(status: 503, code: "HISTORY_UNAVAILABLE", message: "Off")
        }
        await history.load(query: "")
        #expect(history.state == .unavailable)
        fake.stub("history(limit:query:)", returning: HistoryListing(entries: []))
        await history.load(query: "")
        #expect(history.state == .ready)
        #expect(history.entries.isEmpty)
    }

    @Test func lateHistoryRepliesCannotReplaceTheNewSearch() async throws {
        let (_, hosts, fake) = try await QueueStoreTests.setUp()
        let history = PromptHistoryStore(host: hosts.hosts[0], hosts: hosts)
        fake.stub("history(limit:query:)") { call in
            if call.last as? String == "old" {
                await history.load(query: "new")
                return HistoryListing(entries: [HistoryEntry(prompt: "old", model: "flux-dev:q4", usedAt: 1)])
            }
            return HistoryListing(entries: [HistoryEntry(prompt: "new", model: "flux-dev:q4", usedAt: 2)])
        }
        await history.load(query: "old")
        #expect(history.entries.map(\.prompt) == ["new"])
    }

    @Test func changingSearchDuringClearLoadsTheLatestQuery() async throws {
        let (_, hosts, fake) = try await QueueStoreTests.setUp()
        let history = PromptHistoryStore(host: hosts.hosts[0], hosts: hosts)
        fake.stub("clearHistory(keeping:)") { _ in await history.load(query: "new"); return () }
        fake.stub("history(limit:query:)") { call in
            return HistoryListing(entries: [HistoryEntry(prompt: call.last as? String ?? "", model: "flux", usedAt: 1)])
        }
        await history.clear(query: "old")
        #expect(history.entries.map(\.prompt) == ["new"])
    }

    @Test func offlineHistoryRecoversWithoutChangingSearch() async throws {
        let (_, hosts, fake) = try await QueueStoreTests.setUp()
        let host = hosts.hosts[0]
        let history = PromptHistoryStore(host: host, hosts: hosts)
        hosts.setReachability(.down("Offline"), for: host.id)
        await history.load(query: "")
        #expect(history.state == .offline)
        await hosts.refreshAll()
        fake.stub("history(limit:query:)", returning: HistoryListing(entries: []))
        await history.load(query: "")
        #expect(history.state == .ready)
    }

    @Test func promptRecallPreservesEveryOtherDraftField() {
        var draft = RenderDraft()
        draft.prompt = "old"; draft.negativePrompt = "avoid blur"
        draft.steps = 14; draft.seed = 42
        var expected = draft; expected.prompt = "new"
        PromptHistoryStore.recall("new", into: &draft)
        #expect(draft == expected)
    }

    @Test func recalledPromptClearsPriorTransformationProvenance() {
        var draft = RenderDraft()
        draft.prompt = "expanded words"
        draft.originalPrompt = "short words"
        draft.promptTransform = PromptTransformProvenance(
            operation: .expand, rootPrompt: "short words", sourcePrompt: "short words", task: .textToImage)
        PromptHistoryStore.recall("saved prompt", into: &draft)
        #expect(draft.prompt == "saved prompt")
        #expect(draft.originalPrompt == nil)
        #expect(draft.promptTransform == nil)
    }

    @Test func failedClearKeepsTheLoadedRows() async throws {
        let (_, hosts, fake) = try await QueueStoreTests.setUp()
        let history = PromptHistoryStore(host: hosts.hosts[0], hosts: hosts)
        fake.stub("history(limit:query:)", returning: HistoryListing(entries: [HistoryEntry(prompt: "keep", model: "flux", usedAt: 1)]))
        fake.stub("clearHistory(keeping:)") { _ in throw MoldClientError.unreachable("Offline") }
        await history.load(query: "")
        await history.clear(query: "")
        #expect(history.entries.map(\.prompt) == ["keep"])
        #expect(history.state == .failed)
    }
}
