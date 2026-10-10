import Foundation
import MoldClient
import Testing

@testable import Mold

@MainActor
struct LibrarySyncSessionTests {
    @Test func intervalPersistsAndRejectsInvalidValues() {
        let defaults = UserDefaults(suiteName: UUID().uuidString)!
        let session = LibrarySyncSession(defaults: defaults)
        #expect(session.intervalMinutes == 5)
        session.intervalMinutes = 12
        #expect(LibrarySyncSession(defaults: defaults).intervalMinutes == 12)
        session.intervalMinutes = 0
        #expect(session.intervalMinutes == 1)
        session.intervalMinutes = 10000
        #expect(session.intervalMinutes == 1440)
    }

    @Test func editingIntervalReschedulesPendingWaitWithoutStartingAnotherRun() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let backend = FakeBackend(host: local)
        let hosts = HostStore(hosts: [local]) { _ in backend }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let session = LibrarySyncSession(defaults: UserDefaults(suiteName: UUID().uuidString)!, interval: 30)
        let library = LibraryStore(hosts: hosts, syncSession: session)
        session.start(in: library)
        await settle { session.nextRun != nil }
        let reads = backend.callCount("gallery")
        session.intervalMinutes = 12
        #expect((session.nextRun?.timeIntervalSinceNow ?? 0) > 700)
        await Task.yield()
        #expect(backend.callCount("gallery") == reads)
        session.stop(in: library)
        #expect(session.nextRun == nil)
    }

    @Test func stoppedWaitingTaskCannotClearRestartedSessionCountdown() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let backend = FakeBackend(host: local)
        let hosts = HostStore(hosts: [local]) { _ in backend }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let session = LibrarySyncSession(defaults: UserDefaults(suiteName: UUID().uuidString)!, interval: 30)
        let library = LibraryStore(hosts: hosts, syncSession: session)
        session.start(in: library)
        await settle { session.nextRun != nil }
        session.stop(in: library)
        session.start(in: library)
        await settle { session.nextRun != nil }
        await Task.yield()
        #expect(session.isEnabled)
        #expect(session.nextRun != nil)
        session.stop(in: library)
    }

    @Test func acknowledgmentsOnlySuppressExactIssuesAndCanBeReset() throws {
        let defaults = UserDefaults(suiteName: UUID().uuidString)!
        let session = LibrarySyncSession(defaults: defaults)
        #expect(session.hasNewIssues(["old"]))
        session.acknowledge(["old"])
        #expect(!session.hasNewIssues(["old"]))
        #expect(session.hasNewIssues(["changed"]))
        #expect(!LibrarySyncSession(defaults: defaults).hasNewIssues(["old"]))
        session.resetAcknowledgments()
        #expect(session.hasNewIssues(["old"]))
    }

    @Test func reportAcknowledgmentTracksPersistedIssuesAndCanBeCleared() {
        let defaults = UserDefaults(suiteName: UUID().uuidString)!
        LibrarySyncSession(defaults: defaults).acknowledge(["old"])
        let session = LibrarySyncSession(defaults: defaults)
        let library = LibraryStore(hosts: HostStore(hosts: []) { FakeBackend(host: $0) }, syncSession: session)
        library.localSaveIssueKeys = ["media": "old"]
        #expect(library.syncIssueAcknowledgment)
        library.syncIssueAcknowledgment = false
        #expect(session.hasNewIssues(["old"]))
        library.syncIssueAcknowledgment = true
        #expect(library.syncIssueAcknowledgment)
        library.localSaveIssueKeys = ["media": "changed", "other": "old"]
        #expect(!library.syncIssueAcknowledgment)
        library.syncIssueAcknowledgment = true
        library.localSaveIssueKeys = ["media": "changed"]
        library.syncIssueAcknowledgment = false
        #expect(!session.hasNewIssues(["old"]))
        #expect(session.hasNewIssues(["changed"]))
        library.localSaveIssueKeys = ["other": "old"]
        #expect(library.syncIssueAcknowledgment)
        session.resetAcknowledgments()
        #expect(!library.syncIssueAcknowledgment)
        library.localSaveIssueKeys = [:]
        #expect(!library.syncIssueAcknowledgment)
    }

    @Test func previousJoinedAcknowledgmentsMigrateWithoutRepeatingSources() {
        let defaults = UserDefaults(suiteName: UUID().uuidString)!
        defaults.set(["sourceA:sourceB"], forKey: "library.acknowledgedSyncIssues.v1")
        let session = LibrarySyncSession(defaults: defaults)
        #expect(!session.hasNewIssues(["sourceA", "sourceB"]))
        session.unacknowledge(["sourceA"])
        #expect(session.hasNewIssues(["sourceA"]))
        #expect(!session.hasNewIssues(["sourceB"]))
        #expect(!LibrarySyncSession(defaults: defaults).hasNewIssues(["sourceB"]))
    }

    @Test func acknowledgmentsFollowIndividualSourcesAcrossChangingGroups() {
        let session = LibrarySyncSession(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        let library = LibraryStore(hosts: HostStore(hosts: []) { FakeBackend(host: $0) }, syncSession: session)
        library.localSaveIssueKeys = ["legacy": "sourceA:v1sourceB"]
        library.syncIssueAcknowledgment = true
        library.localSaveIssueKeys = ["legacy": "sourceA"]
        #expect(library.syncIssueAcknowledgment)
        library.localSaveIssueKeys = ["legacy": "sourceA:v2sourceB"]
        #expect(!library.syncIssueAcknowledgment)
    }

    @Test func legacyNoticesRemainInDetailsButAcknowledgedSourcesLeaveBanner() {
        let defaults = UserDefaults(suiteName: UUID().uuidString)!
        let library = LibraryStore(hosts: HostStore(hosts: []) { FakeBackend(host: $0) },
                                   syncSession: LibrarySyncSession(defaults: defaults))
        library.localSaveNotices = ["source A legacy inputs", "source B legacy inputs"]
        library.localSaveIssueKeys = ["source A legacy inputs": "routeA-version1", "source B legacy inputs": "routeB-version1"]
        library.localSaveReport = "Copied 2 prints." + LibraryStore.legacyInputNotice(count: 2)
        #expect(library.syncHasUnacknowledgedIssues)
        library.syncIssueAcknowledgment = true
        #expect(library.syncBannerReport == "Copied 2 prints.")
        #expect(library.localSaveReport.contains("2 legacy prints"))
        #expect(!library.syncHasUnacknowledgedIssues)
        #expect(!LibrarySyncSession(defaults: defaults).hasNewIssues(["routeA-version1"]))
        library.localSaveIssueKeys["source B legacy inputs"] = "routeB-version2"
        #expect(library.syncBannerReport.contains("1 legacy print;"))
        #expect(!library.syncIssueAcknowledgment)
        library.syncIssueAcknowledgment = true
        library.localSaveFailures = ["Machine authentication failed"]
        library.localSaveReport += " 1 issues need attention."
        #expect(library.syncHasUnacknowledgedIssues)
        #expect(library.syncBannerReport.contains("1 issues need attention."))
        library.syncSession.resetAcknowledgments()
        #expect(library.syncBannerReport.contains("2 legacy prints"))
    }

    @Test func successfulSyncDoesNotPresentCompletionSheet() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let backend = FakeBackend(host: local)
        let hosts = HostStore(hosts: [local]) { _ in backend }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAllLocally()
        #expect(!library.localSaveReport.isEmpty)
        #expect(!library.localSaveAlertPresented)
    }

    @Test func stopDisablesFutureSyncAndClearsCountdown() {
        let library = LibraryStore(hosts: HostStore(hosts: []) { FakeBackend(host: $0) })
        library.syncSession.start(in: library)
        #expect(library.syncSession.isEnabled)
        library.syncSession.stop(in: library)
        #expect(!library.syncSession.isEnabled)
        #expect(library.syncSession.nextRun == nil)
    }
    @Test func recurringSyncRepeatsWithoutOverlapAndStopPreventsMoreRuns() async throws {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let backend = FakeBackend(host: local)
        backend.delays["gallery"] = .milliseconds(20)
        let hosts = HostStore(hosts: [local]) { _ in backend }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let session = LibrarySyncSession(defaults: UserDefaults(suiteName: UUID().uuidString)!, interval: 0.02)
        let library = LibraryStore(hosts: hosts, syncSession: session)
        session.start(in: library)
        await settle { backend.callCount("gallery") >= 4 && session.nextRun != nil && !library.localSaveRunning }
        #expect(session.nextRun != nil)
        session.stop(in: library)
        let reads = backend.callCount("gallery")
        try await Task.sleep(for: .milliseconds(80))
        #expect(backend.callCount("gallery") == reads)
        #expect(!session.isEnabled)
        #expect(session.nextRun == nil)
    }

}
