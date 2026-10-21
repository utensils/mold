import Foundation
import MoldClient
import MoldClientTesting
import Testing
@testable import MoldCompanion

@MainActor struct DurationSourceRecoveryTests {
    @MainActor final class Fixture {
        let status: ServerStatus
        let caps: Capabilities
        let print: GalleryPrint
        let gate = AsyncStream<Void>.makeStream()
        var pauseHost: MoldHost?
        var paused = false
        var pauseNewExtension = false
        var backends: [FakeBackend] = []
        init() throws {
            status = try MoldJSON.decoder.decode(ServerStatus.self, from: Data(#"{"version":"fixture","busy":false,"uptime_secs":1,"instance_id":"installation"}"#.utf8))
            caps = try MoldJSON.decoder.decode(Capabilities.self, from: Data(#"{"video_jobs":{"supported":true,"literal_prompts":true,"library_extend":true}}"#.utf8))
            print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(#"{"filename":"original.mp4","format":"mp4","timestamp":1,"metadata":{"model":"fixture","prompt":"original","width":768,"height":768}}"#.utf8))
        }
        func shouldPause(_ host: MoldHost) -> Bool {
            guard pauseHost == host, !paused else { return false }; paused = true; return true
        }
        func backend(_ host: MoldHost) -> FakeBackend {
            let fake = FakeBackend(host: host), gate = gate
            fake.stub("status()", returning: status); fake.stub("capabilities()", returning: caps)
            fake.stub("gallery(etag:)") { _ in
                if await self.shouldPause(host) { for await _ in gate.stream { break } }
                return await Fetched<[GalleryPrint]>.fresh([self.print], etag: nil)
            }
            fake.stub("retainedSourceMedia(for:)") { args in
                if args.first as? String == "new.mp4", await self.pauseNewExtension {
                    await MainActor.run { self.paused = true }
                    for await _ in gate.stream { break }
                }
                return RetainedSourceMedia.Inventory(availability: .unavailableLegacy)
            }
            backends.append(fake); return fake
        }
        func setup() async throws -> (GenerateController, MoldHost) {
            let root = FileManager.default.temporaryDirectory.appending(path: "source-recovery-\(UUID())")
            let hosts = HostStore(list: HostListFile(url: root.appending(path: "hosts.json")),
                credentials: HostStoreTests.MemoryCredentials(), makeBackend: { self.backend($0) })
            let host = MoldHost(name: "Fixture", baseURL: URL(string: "http://127.0.0.1:17699")!, connectionMode: .regular)
            hosts.setHosts([host]); hosts.setReachability(.up(status), for: host.id); hosts.setCapabilities(caps, for: host.id)
            let defaults = UserDefaults(suiteName: UUID().uuidString)!
            let generate = GenerateController(hosts: hosts, ledger: PendingLedger(url: root.appending(path: "ledger.json")),
                drafts: DraftStore(directory: root), sequence: NativeSequenceSession(defaults: defaults),
                videoJobs: VideoJobCoordinator(defaults: defaults), videoRestarts: VideoRestartController(defaults: defaults))
            let source = try await VideoLibraryExtensionSource.inspect(filename: print.filename, title: "Original", metadata: print.metadata,
                owner: .init(host: host, instanceId: "installation"), backend: backend(host), current: { true })
            generate.videoDuration = source.authoring(seconds: "4"); generate.durationSource = source
            generate.draft.prompt = "My new prompt"
            return (generate, host)
        }
    }

    @Test func restoredSourceReinspectsAfterConnectionMetadataChangesWithoutChangingDraft() async throws {
        let fixture = try Fixture(); let (generate, host) = try await fixture.setup()
        let draft = generate.draft, authoring = generate.videoDuration
        var refreshed = host; refreshed.connectionInstanceID = "installation"
        generate.hosts.setHosts([refreshed])
        await generate.restoreDurationSource()
        #expect(generate.durationSource?.owner.host == refreshed)
        #expect(generate.durationSourceRefusal == nil)
        #expect(generate.draft == draft && generate.videoDuration == authoring)
    }

    @Test func replacementRestoreCanFinishBeforeOldOwnerRepliesAndOldCompletionCannotClearIt() async throws {
        let fixture = try Fixture(); let (generate, host) = try await fixture.setup()
        generate.durationSource = nil; fixture.pauseHost = host
        let old = Task { await generate.restoreDurationSource() }
        for _ in 0..<100 where !fixture.paused { await Task.yield() }
        #expect(fixture.paused)
        var refreshed = host; refreshed.connectionInstanceID = "installation"
        generate.hosts.setHosts([refreshed])
        await generate.restoreDurationSource()
        #expect(generate.durationSource?.owner.host == refreshed)
        fixture.gate.continuation.finish(); await old.value
        #expect(generate.durationSource?.owner.host == refreshed)
        #expect(!generate.durationSourceLoading)
    }

    @Test func automaticRestorationDoesNotSupersedeExplicitNewerExtendInspection() async throws {
        let fixture = try Fixture(); let (generate, host) = try await fixture.setup()
        generate.durationSource = nil; fixture.pauseNewExtension = true
        let newer = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(#"{"filename":"new.mp4","format":"mp4","timestamp":2,"metadata":{"model":"fixture","prompt":"new"}}"#.utf8))
        generate.extendVideo(LibraryEntry(host: host, print: newer))
        for _ in 0..<100 where !fixture.paused { await Task.yield() }
        #expect(fixture.paused)
        let explicitToken = generate.durationSourceToken
        await generate.restoreDurationSource()
        #expect(generate.durationSourceToken == explicitToken)
        fixture.gate.continuation.finish()
        for _ in 0..<100 where generate.durationSourceLoading { await Task.yield() }
        #expect(generate.durationSource?.source.filename == "new.mp4")
    }

    @Test func cancelledRestoreCanBeReplacedOnSameOwnerWhileDuplicateActiveReadIsCoalesced() async throws {
        let fixture = try Fixture(); let (generate, host) = try await fixture.setup()
        generate.durationSource = nil; fixture.pauseHost = host
        let old = Task { await generate.restoreDurationSource() }
        for _ in 0..<100 where !fixture.paused { await Task.yield() }
        #expect(fixture.paused)
        let originalToken = generate.durationSourceToken
        await generate.restoreDurationSource()
        #expect(generate.durationSourceToken == originalToken)
        old.cancel()
        await generate.restoreDurationSource()
        #expect(generate.durationSource?.owner.host == host)
        fixture.gate.continuation.finish(); await old.value
        #expect(!generate.durationSourceLoading)
        #expect(generate.durationSource?.owner.host == host)
    }
}
