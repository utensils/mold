import Foundation
import MoldClient
import Testing
@testable import Mold

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
            fake.serverStatus = status; fake.capabilityBlock = caps
            fake.prints = [print]
            fake.galleryPrintResponder = { _ in
                if self.shouldPause(host) { for await _ in gate.stream { break } }
                return self.print
            }
            fake.retainedInventoryResponder = { _ in
                RetainedSourceMedia.Inventory(availability: .unavailableLegacy)
            }
            backends.append(fake); return fake
        }
        func setup() async throws -> (GenerateController, MoldHost) {
            let host = MoldHost(name: "Fixture", baseURL: URL(string: "http://127.0.0.1:17699")!, connectionMode: .regular)
            let hosts = HostStore(hosts: [host], makeBackend: { self.backend($0) })
            hosts.reachability[host.id] = .up(status); hosts.capabilities[host.id] = caps
            let generate = GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts))
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
        generate.hosts.hosts = [refreshed]
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
        generate.hosts.hosts = [refreshed]
        await generate.restoreDurationSource()
        #expect(generate.durationSource?.owner.host == refreshed)
        fixture.gate.continuation.finish(); await old.value
        #expect(generate.durationSource?.owner.host == refreshed)
        #expect(!generate.durationSourceLoading)
    }

    @Test func automaticRestorationDoesNotSupersedeExplicitNewerLibraryInspection() async throws {
        let fixture = try Fixture(); let (generate, _) = try await fixture.setup()
        generate.durationSource = nil
        generate.durationSourceToken = UUID(); generate.durationSourceLoading = true
        let explicitToken = generate.durationSourceToken
        await generate.restoreDurationSource()
        #expect(generate.durationSourceToken == explicitToken)
        #expect(generate.durationSourceLoading)
        #expect(generate.durationSource == nil)
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
