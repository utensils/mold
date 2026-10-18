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
        let backend = FakeBackend(host: local, syncOrganization: true)
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
        let library = LibraryStore(hosts: HostStore(hosts: []) { FakeBackend(host: $0, syncOrganization: true) }, syncSession: session)
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

    // Busy runs are simulated by holding `localSaveRunning` rather than by a
    // backend delay, so the ordering under test never races a timer.

    @Test func aChangeDuringARunQueuesExactlyOneFollowUp() async throws {
        let (session, library, backend) = bench(keepInSync: true)
        library.localSaveRunning = true
        for _ in 0 ..< 3 { session.noteRemoteChange(in: library) }
        await settle { session.rerunPending }
        #expect(session.rerunPending)
        #expect(library.localSaveTask == nil, "never overlaps the run in flight")
        let before = backend.callCount("gallery")
        library.localSaveRunning = false
        session.runFinished(in: library)
        #expect(!session.rerunPending)
        #expect(library.localSaveTask != nil, "the one follow-up starts as the run ends")
        await idle(library)
        let oneRun = backend.callCount("gallery") - before
        #expect(oneRun > 0)
        try await Task.sleep(for: .milliseconds(60))
        #expect(backend.callCount("gallery") - before == oneRun)
    }

    @Test func aPressDuringARunPreservesManualFollowUpRegardlessOfKeepInSync() {
        let (off, offLibrary, _) = bench()
        offLibrary.localSaveRunning = true
        off.syncNow(in: offLibrary)
        #expect(off.rerunPending)
        let (on, onLibrary, _) = bench(keepInSync: true)
        onLibrary.localSaveRunning = true
        on.syncNow(in: onLibrary)
        #expect(on.rerunPending)
    }

    @Test func stopDropsTheQueuedFollowUpButKeepsTheChoice() async throws {
        let (session, library, backend) = bench(keepInSync: true)
        library.localSaveRunning = true
        session.noteRemoteChange(in: library)
        await settle { session.rerunPending }
        session.stop(in: library)
        #expect(!session.rerunPending)
        #expect(library.localSaveStopRequested)
        library.localSaveRunning = false
        let reads = backend.callCount("gallery")
        session.runFinished(in: library)
        #expect(library.localSaveTask == nil)
        try await Task.sleep(for: .milliseconds(60))
        #expect(backend.callCount("gallery") == reads)
        #expect(session.keepInSync)
    }

    @Test func quitCancelsTheRunAndBlocksNewOnes() async throws {
        let (session, library, backend) = bench(keepInSync: true)
        library.localSaveRunning = true
        session.noteRemoteChange(in: library)
        await settle { session.rerunPending }
        session.cancelForQuit(in: library)
        #expect(!session.rerunPending)
        library.localSaveRunning = false
        let reads = backend.callCount("gallery")
        session.noteRemoteChange(in: library)
        session.syncNow(in: library)
        session.runFinished(in: library)
        try await Task.sleep(for: .milliseconds(60))
        #expect(library.localSaveTask == nil)
        #expect(backend.callCount("gallery") == reads, "nothing starts once quitting")
    }

    @Test func turningKeepInSyncOnCatchesUpImmediately() async {
        let (session, library, backend) = bench()
        let before = backend.callCount("gallery")
        session.setKeepInSync(true, in: library)
        await settle { backend.callCount("gallery") > before }
        await idle(library)
        #expect(backend.callCount("gallery") > before)
    }

    @Test func anAutomaticRunWaitsForThisMacsEngineInsteadOfAlerting() async throws {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let backend = FakeBackend(host: local, syncOrganization: true)
        let hosts = HostStore(hosts: [local]) { _ in backend }
        let defaults = UserDefaults(suiteName: UUID().uuidString)!
        defaults.set(true, forKey: LibrarySyncSession.keepInSyncKey)
        let session = LibrarySyncSession(defaults: defaults, debounce: .milliseconds(10))
        let library = LibraryStore(hosts: hosts, syncSession: session)
        session.noteRemoteChange(in: library)
        await settle { session.waitingForLocalEngine }
        #expect(session.waitingForLocalEngine)
        #expect(!library.localSaveAlertPresented)
        #expect(library.localSaveTask == nil)
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let before = backend.callCount("gallery")
        session.localEngineAvailable(in: library)
        await settle { backend.callCount("gallery") > before }
        await idle(library)
        #expect(!session.waitingForLocalEngine)
        #expect(backend.callCount("gallery") > before)
    }

    @Test func theRetiredIntervalSettingIsRemoved() {
        let defaults = UserDefaults(suiteName: UUID().uuidString)!
        defaults.set(12, forKey: LibrarySyncSession.retiredIntervalKey)
        _ = LibrarySyncSession(defaults: defaults)
        #expect(defaults.object(forKey: LibrarySyncSession.retiredIntervalKey) == nil)
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
        let library = LibraryStore(hosts: HostStore(hosts: []) { FakeBackend(host: $0, syncOrganization: true) }, syncSession: session)
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
        let library = LibraryStore(hosts: HostStore(hosts: []) { FakeBackend(host: $0, syncOrganization: true) }, syncSession: session)
        library.localSaveIssueKeys = ["legacy": "sourceA:v1sourceB"]
        library.syncIssueAcknowledgment = true
        library.localSaveIssueKeys = ["legacy": "sourceA"]
        #expect(library.syncIssueAcknowledgment)
        library.localSaveIssueKeys = ["legacy": "sourceA:v2sourceB"]
        #expect(!library.syncIssueAcknowledgment)
    }

    @Test func legacyNoticesRemainInDetailsButAcknowledgedSourcesLeaveBanner() {
        let defaults = UserDefaults(suiteName: UUID().uuidString)!
        let library = LibraryStore(hosts: HostStore(hosts: []) { FakeBackend(host: $0, syncOrganization: true) },
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
        let backend = FakeBackend(host: local, syncOrganization: true)
        let hosts = HostStore(hosts: [local]) { _ in backend }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAllLocally()
        #expect(!library.localSaveReport.isEmpty)
        #expect(!library.localSaveAlertPresented)
    }

    @Test func destinationKeepChoiceDoesNotFollowAReplacementRoute() {
        let destination = MoldHost(name: "Backup", baseURL: URL(string: "https://backup.example")!)
        let backend = FakeBackend(host: destination, syncOrganization: true)
        let hosts = HostStore(hosts: [destination]) { _ in backend }
        let defaults = UserDefaults(suiteName: UUID().uuidString)!
        let session = LibrarySyncSession(defaults: defaults)
        let library = LibraryStore(hosts: hosts, syncSession: session)
        library.localSaveRunning = true
        session.setKeepInSync(true, to: destination, in: library)
        #expect(session.keepInSync(to: destination, in: library))
        var replaced = destination; replaced.baseURL = URL(string: "https://replacement.example")!
        hosts.hosts = [replaced]
        #expect(!session.keepInSync(to: replaced, in: library))
        session.stop(in: library)
        #expect(!session.rerunPending)
        #expect(LibrarySyncSession(defaults: defaults).keepInSync(to: destination, in: library))
    }

    @Test func queuedDestinationRunsAreSerializedAndStopClearsEveryDestination() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let first = MoldHost(name: "First", baseURL: URL(string: "https://first.example")!)
        let second = MoldHost(name: "Second", baseURL: URL(string: "https://second.example")!)
        let backends = Dictionary(uniqueKeysWithValues: [local, first, second].map { ($0.id, FakeBackend(host: $0, syncOrganization: true)) })
        let hosts = HostStore(hosts: [local, first, second]) { backends[$0.id]! }
        for host in hosts.hosts { hosts.reachability[host.id] = .up(FakeFixtures.serverStatus()) }
        let session = LibrarySyncSession(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        let library = LibraryStore(hosts: hosts, syncSession: session)
        library.localSaveRunning = true
        session.syncNow(to: first, in: library)
        session.syncNow(to: second, in: library)
        session.syncNow(to: first, in: library)
        #expect(session.rerunPending)
        #expect(library.localSaveTask == nil)
        session.stop(in: library)
        library.localSaveRunning = false
        session.runFinished(in: library)
        #expect(library.localSaveTask == nil)
        #expect(!session.rerunPending)
    }

    @Test func automaticDestinationFailureRemainsInspectableWithoutPresentingAlert() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = MoldHost(name: "Unavailable source", baseURL: URL(string: "https://source.example")!)
        let target = FakeBackend(host: local, syncOrganization: true)
        let source = FakeBackend(host: remote, syncOrganization: true)
        source.plantedErrors["gallery"] = MoldClientError.unauthorized
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? target : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAll(to: local, automatic: true)
        #expect(!library.localSaveFailures.isEmpty)
        #expect(!library.localSaveAlertPresented)
    }

    @Test func remoteKeepDestinationFollowsNewThisMacGalleryEvents() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = MoldHost(name: "Backup", baseURL: URL(string: "https://destination.example")!)
        let source = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let destination = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        source.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? source : destination }
        for machine in hosts.hosts { hosts.reachability[machine.id] = .up(FakeFixtures.serverStatus()) }
        let session = LibrarySyncSession(defaults: UserDefaults(suiteName: UUID().uuidString)!, debounce: .milliseconds(5))
        let library = LibraryStore(hosts: hosts, syncSession: session)
        session.setKeepInSync(true, to: remote, in: library)
        await idle(library)
        let print = FakeFixtures.print("local-arrival.png")
        source.prints = [print]
        library.live.apply(.gallery(.added(filename: print.filename, row: print)), from: local.id, in: library)
        await settle { destination.importedNames == [print.filename] }
        await idle(library)
        #expect(destination.importedNames == [print.filename])
        #expect(!library.localSaveAlertPresented)
        session.stop(in: library)
    }

}
