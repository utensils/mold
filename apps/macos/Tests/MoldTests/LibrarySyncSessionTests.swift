import Foundation
import MoldClient
import Testing

@testable import Mold

@MainActor
struct LibrarySyncSessionTests {
    /// A local-only fixture whose sync run completes without copying, so
    /// `gallery` reads count runs.
    private func bench(debounce: Duration = .milliseconds(20), keepInSync: Bool = false)
        -> (LibrarySyncSession, LibraryStore, FakeBackend) {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let backend = FakeBackend(host: local)
        let hosts = HostStore(hosts: [local]) { _ in backend }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let defaults = UserDefaults(suiteName: UUID().uuidString)!
        defaults.set(keepInSync, forKey: LibrarySyncSession.keepInSyncKey)
        let session = LibrarySyncSession(defaults: defaults, debounce: debounce)
        return (session, LibraryStore(hosts: hosts, syncSession: session), backend)
    }

    private func idle(_ library: LibraryStore) async {
        await settle { library.localSaveTask == nil && !library.localSaveRunning }
    }

    @Test func keepInSyncIsOffByDefaultAndPersists() {
        let defaults = UserDefaults(suiteName: UUID().uuidString)!
        let session = LibrarySyncSession(defaults: defaults)
        #expect(!session.keepInSync)
        let library = LibraryStore(hosts: HostStore(hosts: []) { FakeBackend(host: $0) }, syncSession: session)
        session.setKeepInSync(true, in: library)
        #expect(LibrarySyncSession(defaults: defaults).keepInSync)
        session.setKeepInSync(false, in: library)
        #expect(!LibrarySyncSession(defaults: defaults).keepInSync)
    }

    @Test func syncNowRunsExactlyOnceAndNeverRepeats() async throws {
        let (session, library, backend) = bench()
        let before = backend.callCount("gallery")
        session.syncNow(in: library)
        await idle(library)
        let after = backend.callCount("gallery")
        #expect(after > before)
        try await Task.sleep(for: .milliseconds(80))
        #expect(backend.callCount("gallery") == after, "Sync All is one run, not a schedule")
    }

    @Test func remoteChangesAreIgnoredUnlessKeepInSyncIsOn() async throws {
        let (session, library, backend) = bench()
        let before = backend.callCount("gallery")
        session.noteRemoteChange(in: library)
        try await Task.sleep(for: .milliseconds(80))
        #expect(backend.callCount("gallery") == before)
        #expect(library.localSaveTask == nil)
    }

    @Test func aBurstOfRemoteChangesCoalescesIntoOneRun() async throws {
        let (session, library, backend) = bench(keepInSync: true)
        let before = backend.callCount("gallery")
        for _ in 0 ..< 6 { session.noteRemoteChange(in: library) }
        await settle { library.localSaveTask != nil }
        await idle(library)
        let oneRun = backend.callCount("gallery") - before
        #expect(oneRun > 0)
        try await Task.sleep(for: .milliseconds(80))
        #expect(backend.callCount("gallery") - before == oneRun, "six announcements, one run")
    }

    @Test func aChangeDuringARunQueuesExactlyOneFollowUp() async throws {
        let (session, library, backend) = bench(keepInSync: true)
        backend.delays["gallery"] = .milliseconds(30)
        // What one run reads, measured rather than assumed.
        var start = backend.callCount("gallery")
        session.syncNow(in: library)
        await idle(library)
        let perRun = backend.callCount("gallery") - start
        #expect(perRun > 0)

        start = backend.callCount("gallery")
        session.syncNow(in: library)
        await settle { library.localSaveRunning }
        for _ in 0 ..< 3 { session.noteRemoteChange(in: library) }
        await settle { session.rerunPending }
        #expect(session.rerunPending)
        await settle { backend.callCount("gallery") - start >= 2 * perRun && library.localSaveTask == nil }
        try await Task.sleep(for: .milliseconds(80))
        await idle(library)
        #expect(backend.callCount("gallery") - start == 2 * perRun, "three changes mid-run, one follow-up")
        #expect(!session.rerunPending)
    }

    @Test func stopDropsTheQueuedFollowUpButKeepsTheChoice() async throws {
        let (session, library, backend) = bench(keepInSync: true)
        backend.delays["gallery"] = .milliseconds(30)
        session.syncNow(in: library)
        await settle { library.localSaveRunning }
        session.noteRemoteChange(in: library)
        try await Task.sleep(for: .milliseconds(30))
        session.stop(in: library)
        #expect(!session.rerunPending)
        await idle(library)
        let reads = backend.callCount("gallery")
        try await Task.sleep(for: .milliseconds(80))
        #expect(backend.callCount("gallery") == reads)
        #expect(session.keepInSync)
    }

    @Test func quitCancelsTheRunAndPendingWork() async {
        let (session, library, backend) = bench(keepInSync: true)
        backend.delays["gallery"] = .milliseconds(50)
        session.syncNow(in: library)
        await settle { library.localSaveRunning }
        session.cancelForQuit(in: library)
        #expect(library.localSaveStopRequested || library.localSaveTask?.isCancelled == true)
        await idle(library)
        #expect(!session.rerunPending)
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

}
