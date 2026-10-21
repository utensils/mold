import Foundation
import CryptoKit
import MoldClient
import Testing

@testable import Mold

@MainActor
struct LibraryLocalSaveTests {
    @Test func syncStagesKeepReceiptsBeforeOrganizationAndRequireStableFinalEvidence() async throws {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("sync-stage-order-\(UUID().uuidString)")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let target = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        var mutable = GalleryPrint.Mutable(versionedPrint("stage-order.png", prompt: "fox"))
        mutable.favorite = true
        let print = mutable.build()
        source.prints = [print]
        source.mediaAnswer = Data([1, 2, 3])
        source.syncCheckpoint = [print.filename: "source-v1"]
        target.syncCheckpoint = [print.filename: "local-v1"]
        target.listsImports = { _ in print }
        var changedDuringVerification = false
        source.retainedTransferOfferResponder = { _ in
            if target.callCount("boundedSyncCheckpoint") == 1 {
                changedDuringVerification = true
                target.syncCheckpoint = [print.filename: "local-v2"]
            }
            return .init(archiveIdentitySha256: String(repeating: "a", count: 64), members: [],
                         outputSha256: "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81",
                         outputSizeBytes: 3, metadata: print.metadata)
        }
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? target : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAllLocally()

        #expect(library.localSaveFailures.isEmpty)
        #expect(changedDuringVerification)
        let calls = target.calls
        let imported = try #require(calls.firstIndex(of: "importPrint"))
        let organized = try #require(calls.firstIndex(of: "mutate"))
        let checkpoints = calls.indices.filter { calls[$0] == "gallerySyncCheckpoint" || calls[$0] == "boundedSyncCheckpoint" }
        #expect(checkpoints.count == 3)
        #expect(imported < organized)
        // The output receipt is the import's own answer: no listing read.
        #expect(!calls.contains("galleryPrint"))
        #expect(library.syncRecords.records(hosts.instanceID(of: local.id))[
            "\(remote.id.uuidString):\(print.filename)"]?.destinationFilename == print.filename)
        #expect(checkpoints.filter { $0 > organized }.count == 2)
        #expect(target.boundedCheckpointRequests.suffix(2).allSatisfy { $0 == [print.filename] })

        let probes = source.callCount("retainedMediaTransferOffer")
        await library.syncAllLocally()
        #expect(library.localSaveFailures.isEmpty)
        #expect(target.importedNames.count == 1)
        // A changed final destination must force verification next time,
        // even though its copied output and organization already succeeded.
        #expect(source.callCount("retainedMediaTransferOffer") > probes)
    }

    @Test func cachedCheckpointCannotSkipAfterRouteChangesDuringDestinationProbe() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("checkpoint-route-race")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let target = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let print = versionedPrint("checkpoint-route-race.png", prompt: "fox")
        source.prints = [print]
        source.mediaAnswer = Data([1, 2, 3])
        source.syncCheckpoint = [print.filename: "source-v1"]
        target.syncCheckpoint = [print.filename: "local-v1"]
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? target : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAllLocally()
        target.prints = [print]
        target.mediaAnswers[print.filename] = Data([1, 2, 3])
        await library.syncAllLocally()
        let probes = target.callCount("gallerySyncCheckpoint")
        let imports = target.importedNames.count
        target.delays["gallerySyncCheckpoint"] = .milliseconds(100)
        let run = Task { await library.syncAllLocally() }
        await settle { target.callCount("gallerySyncCheckpoint") > probes }
        var replacement = remote
        replacement.baseURL = URL(string: "http://replacement-checkpoint-origin")!
        hosts.hosts = [local, replacement]
        await run.value
        #expect(library.localSaveFailures.contains { $0.contains("machine route changed") })
        #expect(target.importedNames.count == imports)
    }

    @Test func unchangedCheckpointSkipsPerPrintProbesAndChangedInputsRepair() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("checkpoint-remote")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let target = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let print = versionedPrint("checkpoint.png", prompt: "fox")
        source.prints = [print]
        source.mediaAnswer = Data([1, 2, 3])
        source.syncCheckpoint = [print.filename: "source-v1"]
        target.syncCheckpoint = [print.filename: "local-v1"]
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? target : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAllLocally()
        target.prints = [print]
        target.mediaAnswers[print.filename] = Data([1, 2, 3])
        await library.syncAllLocally()
        let probes = source.callCount("retainedMediaTransferOffer")
        source.conditionalGallery = true
        await library.syncAllLocally()
        #expect(source.galleryETags.last! == "fake-etag")
        #expect(source.callCount("retainedMediaTransferOffer") == probes)
        source.syncCheckpoint = [print.filename: "source-v2"]
        // Keep in Sync's automatic run plans from the cursor: the row did not
        // change, so it is not re-verified...
        await library.syncAllLocally(fullPlan: false)
        #expect(source.callCount("retainedMediaTransferOffer") == probes, "an unchanged row is not re-verified")
        // ...but a press plans every print against fresh checkpoint evidence.
        await library.syncAllLocally()
        #expect(source.callCount("retainedMediaTransferOffer") > probes)
    }

    @Test func relaunchAfterOrganizationFailureUsesDurableCopyWithoutOutputDownload() async throws {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("resume-checkpoint-remote")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let target = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        var json = try JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(versionedPrint("resume-checkpoint.png", prompt: "fox"))) as! [String: Any]
        json["favorite"] = true
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: JSONSerialization.data(withJSONObject: json))
        source.prints = [print]
        source.mediaAnswer = Data([1, 2, 3])
        source.retainedTransferOfferResponder = { _ in
            // The destination fake lists the imported output without organization.
            return .init(archiveIdentitySha256: String(repeating: "a", count: 64), members: [],
                         outputSha256: "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81",
                         outputSizeBytes: 3, metadata: print.metadata)
        }
        target.refuses.insert("mutate")
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? target : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let files = relaunchDirectory()
        defer { try? FileManager.default.removeItem(at: files) }
        await LibraryStore(hosts: hosts, syncRecords: LibrarySyncRecords(directory: files)).syncAllLocally()
        let downloads = source.callCount("mediaFile")
        #expect(downloads == 1)
        target.refuses.remove("mutate")
        await LibraryStore(hosts: hosts, syncRecords: LibrarySyncRecords(directory: files)).syncAllLocally()
        #expect(source.callCount("mediaFile") == downloads)
        #expect(target.mutationRequests.contains { $0.favorite == true })
    }

    @Test func failedRetainedCopyResumesFromDurableOutputReceiptAfterRelaunch() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("output-receipt-remote")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let target = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let print = versionedPrint("output-receipt.png", prompt: "fox")
        source.prints = [print]
        source.mediaAnswer = Data([1, 2, 3])
        target.listsImports = { _ in print }
        var failRetained = true
        source.retainedTransferOfferResponder = { _ in
            if failRetained && !target.importedNames.isEmpty {
                throw MoldClientError.http(status: 409, code: "RETAINED_MEDIA_COPY_INCOMPLETE", message: "fixture interrupted")
            }
            return .init(archiveIdentitySha256: String(repeating: "a", count: 64), members: [],
                         outputSha256: "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81",
                         outputSizeBytes: 3, metadata: print.metadata)
        }
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? target : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let files = relaunchDirectory()
        defer { try? FileManager.default.removeItem(at: files) }
        let first = LibraryStore(hosts: hosts, syncRecords: LibrarySyncRecords(directory: files))
        await first.syncAllLocally()
        #expect(!first.localSaveFailures.isEmpty)
        let downloads = source.callCount("mediaFile")
        failRetained = false
        let resumed = LibraryStore(hosts: hosts, syncRecords: LibrarySyncRecords(directory: files))
        await resumed.syncAllLocally()
        #expect(resumed.localSaveFailures.isEmpty)
        #expect(source.callCount("mediaFile") == downloads)
        #expect(target.importedNames.count == 1)
    }

    private func retainedOffer(metadata: OutputMetadata) -> RetainedSourceMedia.TransferOffer {
        .init(archiveIdentitySha256: String(repeating: "a", count: 64), members: [
            .init(memberId: "source", role: "source_image", position: "scalar", sizeBytes: 3,
                  sha256: "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81"),
        ], outputSha256: "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81",
           outputSizeBytes: 3, metadata: metadata)
    }

    @Test func existingEmbeddedCopyRepairsArchiveSourcesWithoutDuplicatingOutput() async throws {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("archive-enriched-remote")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let target = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let embedded = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(
            #"{"filename":"archive-enriched.png","timestamp":1000,"size_bytes":3,"media_version":"1000:3","metadata":{"prompt":"fox","version":"0.32.0"}}"#.utf8))
        let archived = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(
            #"{"filename":"archive-enriched.png","timestamp":1000,"size_bytes":3,"media_version":"1000:3","metadata":{"prompt":"fox","version":"0.32.0 (5b61d17 2026-09-27)","job_id":"original-job","generation_time_ms":8516}}"#.utf8))
        source.prints = [archived]
        target.prints = [embedded]
        source.mediaAnswer = Data([1, 2, 3])
        target.mediaAnswers[embedded.filename] = Data([1, 2, 3])
        source.retainedTransferOffers[archived.filename] = retainedOffer(metadata: archived.metadata)
        source.retainedMemberBytes["source"] = Data([1, 2, 3])
        target.capabilityBlock = try MoldJSON.decoder.decode(Capabilities.self, from: Data(
            #"{"retained_media_transfer":{"protocol_version":1}}"#.utf8))
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? target : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAllLocally()
        #expect(target.importedNames.isEmpty)
        #expect(target.retainedTransfers.count == 1)
        #expect(library.localSaveFailures.isEmpty)
        let outputReads = source.callCount("mediaFile")
        await library.syncAllLocally()
        #expect(target.importedNames.isEmpty)
        #expect(target.retainedTransfers.count == 1)
        #expect(source.callCount("mediaFile") == outputReads)
        #expect(library.localSaveFailures.isEmpty)
    }

    @Test func unsupportedRetainedDestinationDoesNotImportOutput() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("unsupported-retained-remote")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let target = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        target.capabilityBlock = try! MoldJSON.decoder.decode(Capabilities.self, from: Data("{}".utf8))
        let print = versionedPrint("unsupported.png", prompt: "fox")
        source.prints = [print]
        source.mediaAnswer = Data([1, 2, 3])
        source.retainedTransferOffers[print.filename] = retainedOffer(metadata: print.metadata)
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? target : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.saveLocally([LibraryEntry(host: remote, print: print)])
        #expect(target.callCount("importPrint") == 0)
        #expect(source.callCount("mediaFile") == 0)
        #expect(!library.localSaveFailures.isEmpty)
    }

    private func legacyPrint(_ name: String, upscaled: Bool) throws -> GalleryPrint {
        var metadata: [String: Any] = ["prompt": "legacy", "model": "fixture"]
        if upscaled {
            metadata["model"] = "real-esrgan-x4plus:fp16"
            metadata["edit_image_sha256s"] = [String(repeating: "b", count: 64)]
        } else {
            metadata["source_image_sha256"] = String(repeating: "b", count: 64)
        }
        return try MoldJSON.decoder.decode(GalleryPrint.self, from: JSONSerialization.data(withJSONObject: [
            "filename": name, "timestamp": 1000, "size_bytes": 3,
            "media_version": "1000:3", "metadata": metadata,
        ]))
    }

    @Test(arguments: [false, true])
    func legacyInputsDoNotPreventSavingFinishedOutput(upscaled: Bool) async throws {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("legacy-new-source")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let target = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let print = try legacyPrint("legacy-new.png", upscaled: upscaled)
        source.prints = [print]
        source.mediaAnswer = Data([1, 2, 3])
        source.retainedInventories[print.filename] = .init(availability: .unavailableLegacy, members: [])
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? target : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts, syncSession: LibrarySyncSession(defaults: UserDefaults(suiteName: UUID().uuidString)!))
        await library.saveLocally([LibraryEntry(host: remote, print: print)])
        #expect(target.importedNames == [print.filename])
        #expect(target.importedMedia == [Data([1, 2, 3])])
        #expect(target.retainedTransfers.isEmpty)
        #expect(library.localSaveFailures.isEmpty)
        #expect(library.localSaveReport.contains("Copied 1 print"))
        #expect(library.localSaveReport.contains("Original inputs are unavailable on the source machine for 1 legacy print; its finished file is saved."))
        #expect(library.localSaveNotices.count == 1)
        #expect(library.localSaveIssueKeys[library.localSaveNotices[0]] != nil)
        library.syncIssueAcknowledgment = true
        #expect(!library.syncBannerReport.contains("Original inputs are unavailable"))
        #expect(library.localSaveReport.contains("Original inputs are unavailable"))
        #expect(!library.localSaveReport.contains("not copied"))
        #expect(!library.localSaveReport.contains("issues need attention"))
    }

    @Test(arguments: [false, true])
    func existingLegacyOutputsStaySyncedAcrossRepeatedCachedSync(upscaled: Bool) async throws {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("legacy-existing-source")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let target = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let print = try legacyPrint("legacy-existing.png", upscaled: upscaled)
        source.prints = [print]
        target.prints = [print]
        source.mediaAnswer = Data([1, 2, 3])
        target.mediaAnswers[print.filename] = Data([1, 2, 3])
        source.retainedInventories[print.filename] = .init(availability: .unavailableLegacy, members: [])
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? target : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAllLocally()
        #expect(source.callCount("mediaFile") == 1)
        let outputReads = source.callCount("mediaFile")
        for _ in 0..<2 {
            #expect(target.importedNames.isEmpty)
            #expect(library.localSaveFailures.isEmpty)
            #expect(library.localSaveReport.contains("1 was already here"))
            #expect(library.localSaveReport.contains("Original inputs are unavailable"))
            #expect(!library.localSaveReport.contains("not copied"))
            #expect(!library.localSaveReport.contains("issues need attention"))
            await library.syncAllLocally()
            #expect(source.callCount("mediaFile") == outputReads)
        }
        #expect(target.importedNames.isEmpty)
        #expect(library.localSaveFailures.isEmpty)
        #expect(library.localSaveReport.contains("1 was already here"))
    }

    @Test func unsupportedSourceTransferEndpointStillRefusesDisclosedInputs() async throws {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("unsupported-source-transfer")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let target = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let print = try legacyPrint("unsupported-source.png", upscaled: false)
        source.prints = [print]
        source.mediaAnswer = Data([1, 2, 3])
        source.plantedErrors["retainedMediaTransferOffer"] = MoldClientError.http(status: 404, code: nil, message: "Not found")
        source.retainedInventories[print.filename] = .init(availability: .unavailableLegacy, members: [])
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? target : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.saveLocally([LibraryEntry(host: remote, print: print)])
        #expect(target.callCount("importPrint") == 0)
        #expect(source.callCount("mediaFile") == 0)
        #expect(!library.localSaveFailures.isEmpty)
        #expect(!library.localSaveReport.contains("Original inputs are unavailable"))
    }

    @Test(arguments: [RetainedSourceMedia.Availability.unavailableAuth, .unavailableMissingOrCorrupt, .unknown])
    func unverifiedMissingInputsRemainFailures(availability: RetainedSourceMedia.Availability) async throws {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("unverified-source")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let target = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let print = try legacyPrint("unverified.png", upscaled: false)
        source.prints = [print]
        source.mediaAnswer = Data([1, 2, 3])
        source.retainedInventories[print.filename] = .init(availability: availability, members: [])
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? target : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAllLocally()
        #expect(target.importedNames.isEmpty)
        #expect(source.callCount("mediaFile") == 0)
        #expect(library.localSaveFailures.count == 1)
        #expect(library.localSaveReport.contains("1 were not copied"))
        #expect(!library.localSaveReport.contains("Original inputs are unavailable"))
    }

    @Test func saveLocallyRetainsSourceMediaBeforeReportingSuccess() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("retained-remote")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let target = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        target.capabilityBlock = try! MoldJSON.decoder.decode(Capabilities.self, from: Data(#"{"retained_media_transfer":{"protocol_version":1}}"#.utf8))
        let print = versionedPrint("retained.png", prompt: "fox")
        source.prints = [print]
        source.mediaAnswer = Data([1, 2, 3])
        source.retainedTransferOffers[print.filename] = retainedOffer(metadata: print.metadata)
        source.retainedMemberBytes["source"] = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? target : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.saveLocally([LibraryEntry(host: remote, print: print)])
        #expect(target.retainedTransfers.count == 1)
        #expect(target.retainedTransfers.first?.0 == print.filename)
        #expect(target.retainedTransfers.first?.1.members.first?.role == "source_image")
        #expect(library.localSaveFailures.isEmpty)
    }


    @Test func copiedH3ReferencesRemainReusableWithoutTheOriginalHost() async throws {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("h3-source")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let target = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        target.capabilityBlock = try MoldJSON.decoder.decode(Capabilities.self, from: Data(#"{"retained_media_transfer":{"protocol_version":1}}"#.utf8))
        let members = (0..<4).map { index in
            let bytes = Data([UInt8(index)])
            source.retainedMemberBytes["original-\(index)"] = bytes
            return RetainedSourceMedia.TransferMember(memberId: "original-\(index)", role: "references", position: "item:\(index)", sizeBytes: bytes.count,
                sha256: SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined())
        }
        let references: [[String: Any]] = members.map {
            ["kind": "image", "mime_type": "image/png", "sha256": $0.sha256, "width": 32, "height": 32]
        }
        let metadata: [String: Any] = ["prompt": "fixture", "model": "minimax-h3-ref2va:fixture", "version": "fixture", "references": references]
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: JSONSerialization.data(withJSONObject:
            ["filename": "h3-copy.mp4", "timestamp": 1000, "size_bytes": 3, "media_version": "1000:3", "metadata": metadata]))
        source.prints = [print]
        source.mediaAnswer = Data([1, 2, 3])
        let offer = retainedOffer(metadata: print.metadata)
        source.retainedTransferOffers[print.filename] = .init(archiveIdentitySha256: offer.archiveIdentitySha256,
            members: members, outputSha256: offer.outputSha256, outputSizeBytes: offer.outputSizeBytes, metadata: print.metadata)
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? target : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAllLocally()
        #expect(library.localSaveFailures.isEmpty)
        #expect(target.retainedTransfers.count == 1)
        #expect(target.retainedTransfers.first?.1.members.map(\.position) == members.map(\.position))
        hosts.hosts = [local]
        let sourceCalls = source.calls.count
        let store = ReuseStore(hosts: hosts)
        let draft = RenderDraft(reusing: print.metadata)
        store.arm(draft)
        await store.probe([.init(host: local.id, filename: print.filename)], fence: store.currentFence, disclosing: print.metadata)
        #expect(store.referenceRefusal(for: draft) == nil)
        let authority = try #require(store.take(for: draft))
        #expect(authority.origin == local.id)
        #expect(authority.members.map(\.memberId) == (0..<4).map { "destination-\($0)" })
        target.retainedSession = try MoldJSON.decoder.decode(RetainedSourceMedia.ReuseSession.self, from: Data(#"{"instance_id":"destination","expires_at":100,"request_sha256":"digest","session_handle":"fixture-session"}"#.utf8))
        var request = GenerateRequest(prompt: "changed", model: "minimax-h3-ref2va:fixture", width: 32, height: 32, steps: 4, guidance: 0)
        request.references = try #require(draft.media.generationReferences.canonicalReferences)
        let admission = try await RetainedSourceMedia.hydrated(.init(requests: [request]), filename: authority.filename,
            members: authority.members, sameHost: true, origin: target, target: target)
        #expect(admission.retainedMediaSession == "fixture-session")
        #expect(target.retainedSessionRequests.first?.members == authority.members.map(\.memberId))
        #expect(source.calls.count == sourceCalls)
    }

    @Test func cachedSyncRepairsMissingSourcesWithoutDownloadingOutputAgain() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("cached-retained-remote")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let target = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        target.capabilityBlock = try! MoldJSON.decoder.decode(Capabilities.self, from: Data(#"{"retained_media_transfer":{"protocol_version":1}}"#.utf8))
        let print = versionedPrint("cached-retained.png", prompt: "fox")
        source.prints = [print]
        source.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? target : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAllLocally()
        target.prints = [print]
        target.mediaAnswers[print.filename] = Data([1, 2, 3])
        await library.syncAllLocally()
        let outputReads = source.callCount("mediaFile")
        source.retainedTransferOffers[print.filename] = retainedOffer(metadata: print.metadata)
        source.retainedMemberBytes["source"] = Data([1, 2, 3])
        await library.syncAllLocally()
        #expect(source.callCount("mediaFile") == outputReads)
        #expect(target.retainedTransfers.count == 1)
        #expect(library.localSaveFailures.isEmpty)
    }

    @Test func failedSourceTransferLeavesCopyIncompleteAndRepairable() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("failed-retained-remote")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let target = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        target.capabilityBlock = try! MoldJSON.decoder.decode(Capabilities.self, from: Data(#"{"retained_media_transfer":{"protocol_version":1}}"#.utf8))
        let print = versionedPrint("failed-retained.png", prompt: "fox")
        source.prints = [print]
        source.mediaAnswer = Data([1, 2, 3])
        source.retainedTransferOffers[print.filename] = retainedOffer(metadata: print.metadata)
        source.retainedMemberBytes["source"] = Data([1, 2, 3])
        target.plantedErrors["importRetainedMedia"] = MoldClientError.http(status: 503, code: nil, message: "Source copy interrupted")
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? target : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAllLocally()
        #expect(target.importedNames == [print.filename])
        #expect(library.localSaveFailures.contains { $0.contains("Source copy interrupted") })
        target.prints = [print]
        target.mediaAnswers[print.filename] = Data([1, 2, 3])
        target.plantedErrors.removeValue(forKey: "importRetainedMedia")
        await library.syncAllLocally()
        #expect(target.importedNames == [print.filename])
        #expect(target.retainedTransfers.count == 1)
        #expect(library.localSaveFailures.isEmpty)
    }

    /// Where a "relaunch" reads back what the first launch wrote.
    private func relaunchDirectory() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "mold-sync-relaunch-\(UUID().uuidString)")
    }

    private func host(_ name: String) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func versionedPrint(_ name: String, prompt: String? = nil) -> GalleryPrint {
        let json = try! JSONSerialization.data(withJSONObject: [
            "filename": name, "timestamp": 1000, "size_bytes": 3,
            "media_version": "1000:3", "metadata": ["prompt": prompt.map { $0 as Any } ?? NSNull()],
        ])
        return try! MoldJSON.decoder.decode(GalleryPrint.self, from: json)
    }

    @Test func savingAFilteredRemoteSelectionImportsOnlyPicturesIntoThisMac() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("remote")
        let localBackend = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let remoteBackend = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { host in
            host.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        let picture = LibraryEntry(host: remote, print: FakeFixtures.print("remote.png", prompt: "a fox"))
        remoteBackend.prints = [picture.print]
        let localRow = LibraryEntry(host: local, print: FakeFixtures.print("mine.png"))
        let mesh = LibraryEntry(host: remote, print: FakeFixtures.print("mesh.glb"))

        await library.saveLocally([picture, localRow, mesh])

        #expect(remoteBackend.callCount("mediaFile") == 1)
        #expect(remoteBackend.callCount("gallery") == 1)
        #expect(localBackend.importedNames == ["remote.png"])
        #expect(localBackend.importedMedia.first == Data([1, 2, 3]))
        #expect(localBackend.importedItems.first?.originalMetadata?.prompt == "a fox")
        #expect(localBackend.callCount("gallery") == 3)
        #expect(library.localSaveReport.contains("Skipped 2"))
    }

    @Test func anUnavailableLocalEngineExplainsWhyNothingWasSaved() async {
        let remote = host("remote")
        let remoteBackend = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let hosts = HostStore(hosts: [remote]) { _ in remoteBackend }
        let library = LibraryStore(hosts: hosts)
        library.localSaveTask = Task {}
        let picture = LibraryEntry(host: remote, print: FakeFixtures.print("remote.png"))

        await library.saveLocally([picture])

        #expect(remoteBackend.callCount("mediaFile") == 0)
        #expect(library.localSaveAlertPresented)
        #expect(library.localSaveTask == nil)
        #expect(library.localSaveReport.contains("Start This Mac’s engine"))
    }

    @Test func remoteCollectionIsCreatedAndFiledOnlyOnThisMac() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("remote")
        let localBackend = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let remoteBackend = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        remoteBackend.collectionRows = [Collection(id: "remote-id", name: "Night Sky",
                                                    slug: "night-sky")]
        localBackend.collectionCreateResponses["Night Sky"] = Collection(id: "local-id", name: "Night Sky", slug: "night-sky")
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        var mutable = GalleryPrint.Mutable(FakeFixtures.print("star.png"))
        mutable.collections = ["remote-id"]
        let print = mutable.build()
        remoteBackend.prints = [print]

        await library.saveLocally([LibraryEntry(host: remote, print: print)])

        #expect(localBackend.importedNames == ["star.png"])
        #expect(localBackend.mutationRequests.count == 1)
        #expect(localBackend.mutationRequests.first?.filenames == ["star.png"])
        #expect(localBackend.mutationRequests.first?.addToCollection?.name == "Night Sky")
        #expect(remoteBackend.callCount("mutate") == 0)
        #expect(localBackend.callCount("createCollection") == 1)
    }

    @Test func unavailableSourceCollectionsDoNotBlockPictureCopies() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("remote")
        let localBackend = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let remoteBackend = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        remoteBackend.plantedErrors["collections"] = MoldClientError.malformedResponse
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        var mutable = GalleryPrint.Mutable(FakeFixtures.print("star.png"))
        mutable.collections = ["remote-id"]
        let print = mutable.build()
        let plain = FakeFixtures.print("plain.png")
        remoteBackend.prints = [print, plain]

        await library.saveLocally([LibraryEntry(host: remote, print: print),
                                   LibraryEntry(host: remote, print: plain)])

        #expect(Set(localBackend.importedNames) == ["star.png", "plain.png"])
        #expect(localBackend.mutationRequests.isEmpty)
        #expect(library.localSaveFailures.contains { $0.contains("Collections on") })
    }

    @Test func stoppedSaveDoesNotStartAnotherTransfer() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("remote")
        let localBackend = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let remoteBackend = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let print = FakeFixtures.print("star.png")
        remoteBackend.prints = [print]
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        library.localSaveStopRequested = true
        await library.saveLocally([LibraryEntry(host: remote, print: print)])
        #expect(localBackend.importedNames.isEmpty)
        #expect(library.localSaveFailures.contains { $0.contains("stopped") })
    }

    @Test func aFailedImportReportsOneFailureAndContinuesTheBatch() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("remote")
        let localBackend = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let remoteBackend = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { host in
            host.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        let collision = LibraryEntry(host: remote, print: FakeFixtures.print("taken.png"))
        let available = LibraryEntry(host: remote, print: FakeFixtures.print("free.png"))
        remoteBackend.prints = [collision.print, available.print]
        localBackend.importFailures = ["taken.png"]

        await library.saveLocally([collision, available])

        #expect(localBackend.importedNames == ["free.png"])
        #expect(library.localSaveReport.contains("Copied 1 print to"))
        #expect(library.localSaveReport.contains("1 were not copied"))
        #expect(library.localSaveFailures.contains { $0.contains("taken.png") })
        #expect(library.localSaveAlertPresented)
        #expect(localBackend.callCount("gallery") == 3)
    }

    @Test func syncAllIgnoresSelectionAndCopiesEveryMediaKindAndEmptyCollections() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("remote")
        let localBackend = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let remoteBackend = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        let print = try! MoldJSON.decoder.decode(GalleryPrint.self, from: Data(#"""
            {
            "filename":"star.png","timestamp":1000,
            "metadata":{"prompt":"star","model":"flux","seed":1},
            "collections":["remote-night"],"title":"A Star","favorite":true,"tags":["night"]
            }
            """#.utf8))
        remoteBackend.prints = [print, FakeFixtures.print("movie.mp4"),
                                FakeFixtures.print("shape.glb"), FakeFixtures.print("sound.wav")]
        remoteBackend.collectionRows = [
            Collection(id: "remote-night", name: "Night Sky", slug: "night-sky"),
            Collection(id: "remote-empty", name: "Empty Shelf", slug: "empty-shelf"),
        ]
        localBackend.collectionCreateResponses = [
            "Night Sky": Collection(id: "local-night", name: "Night Sky", slug: "night-sky"),
            "Empty Shelf": Collection(id: "local-empty", name: "Empty Shelf", slug: "empty-shelf"),
        ]
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)

        await library.syncAllLocally()

        #expect(Set(localBackend.importedNames) == ["star.png", "movie.mp4", "shape.glb", "sound.wav"])
        #expect(localBackend.callCount("createCollection") == 2)
        #expect(localBackend.mutationRequests.contains { $0.addToCollection?.name == "Night Sky"
            && $0.filenames == ["star.png"] })
        #expect(localBackend.mutationRequests.contains { $0.titles.contains {
            $0.filename == "star.png" && $0.title == "A Star" } })
        #expect(localBackend.mutationRequests.contains { $0.favorite == true
            && $0.filenames == ["star.png"] })
        #expect(localBackend.mutationRequests.contains { $0.addTags == ["night"]
            && $0.filenames == ["star.png"] })
        #expect(library.localSaveReport.contains("Created 2 collections"))
    }

    @Test func syncAllKeepsSameNamedPrintsFromDifferentHostsAndRerunsWithoutDuplicates() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let first = MoldHost(id: UUID(),
                             name: "first", baseURL: URL(string: "http://first")!)
        let second = MoldHost(id: UUID(),
                              name: "second", baseURL: URL(string: "http://second")!)
        let localBackend = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let firstBackend = FakeBackend(host: first, noRetainedMedia: true, syncOrganization: true)
        let secondBackend = FakeBackend(host: second, noRetainedMedia: true, syncOrganization: true)
        firstBackend.prints = [versionedPrint("same.png")]
        secondBackend.prints = [versionedPrint("same.png")]
        firstBackend.mediaAnswer = Data([1, 2, 3])
        secondBackend.mediaAnswer = Data([4, 5, 6])
        let hosts = HostStore(hosts: [local, first, second]) { machine in
            if machine.id == local.id { return localBackend }
            return machine.id == first.id ? firstBackend : secondBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)

        await library.syncAllLocally()
        let copied = localBackend.importedNames
        #expect(copied.count == 2)
        #expect(Set(copied).count == 2)
        #expect(copied.contains("same.png"))
        for (name, bytes) in zip(localBackend.importedNames, localBackend.importedMedia) {
            localBackend.mediaAnswers[name] = bytes
            localBackend.prints.append(versionedPrint(name))
        }

        await library.syncAllLocally()

        #expect(localBackend.importedNames == copied)
        #expect(library.localSaveReport.contains("2 were already here"))
        let remoteReads = firstBackend.callCount("mediaFile") + secondBackend.callCount("mediaFile")
        await library.syncAllLocally()
        #expect(firstBackend.callCount("mediaFile") + secondBackend.callCount("mediaFile") == remoteReads)
    }

    @Test func syncAllKeepsDifferentRecipesEvenWhenMediaBytesMatch() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("different-recipe")
        let localBackend = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let remoteBackend = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let name = "same.png"
        localBackend.prints = [versionedPrint(name, prompt: "local recipe")]
        localBackend.mediaAnswers[name] = Data([1, 2, 3])
        remoteBackend.prints = [versionedPrint(name, prompt: "remote recipe")]
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())

        await LibraryStore(hosts: hosts).syncAllLocally()

        #expect(localBackend.importedNames.count == 1)
        #expect(localBackend.importedNames.first != name)
    }

    @Test func syncAllBoundsCollisionNamesToFileSystemLimit() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("long-names")
        let localBackend = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let remoteBackend = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let name = String(repeating: "🌟", count: 59) + ".png"
        localBackend.prints = [versionedPrint(name, prompt: "local")]
        localBackend.mediaAnswers[name] = Data([1, 2, 3])
        remoteBackend.prints = [versionedPrint(name, prompt: "remote")]
        remoteBackend.mediaAnswer = Data([4, 5, 6])
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())

        await LibraryStore(hosts: hosts).syncAllLocally()

        #expect(localBackend.importedNames.count == 1)
        #expect(localBackend.importedNames[0].utf8.count <= 255)
        #expect(localBackend.importedNames[0].hasSuffix(".png"))
    }

    @Test func syncAllReportsAnUnavailableRemoteHost() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("unavailable")
        let localBackend = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let remoteBackend = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        remoteBackend.plantedErrors["gallery"] = MoldClientError.malformedResponse
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)

        await library.syncAllLocally()

        #expect(library.localSaveFailures.contains { $0.contains("unavailable") })
        #expect(library.localSaveReport.contains("issues need attention"))
    }

    @Test func failedOrganizationIsRetriedAfterMediaWasAlreadyPresent() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("retry-organization")
        let localBackend = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let remoteBackend = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        var mutable = GalleryPrint.Mutable(versionedPrint("same.png", prompt: "recipe"))
        mutable.title = "Remote title"
        remoteBackend.prints = [mutable.build()]
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        localBackend.plantedErrors["mutate"] = MoldClientError.malformedResponse
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)

        await library.syncAllLocally()
        #expect(localBackend.importedNames == ["same.png"])
        #expect(library.localSaveFailures.contains { $0.contains("organization could not be verified") })
        let organizationKey = LocalSyncRecord.key(hostID: remote.id, filename: "same.png")
        #expect(library.syncRecords.records(hosts.instanceID(of: local.id))[organizationKey]?.organizationIntent != nil,
            "upgrading the output receipt must preserve pending organization")

        localBackend.prints = [versionedPrint("same.png", prompt: "recipe")]
        localBackend.mediaAnswers["same.png"] = Data([1, 2, 3])
        localBackend.plantedErrors.removeValue(forKey: "mutate")
        await library.syncAllLocally()
        #expect(localBackend.mutationRequests.contains { $0.titles.contains {
            $0.filename == "same.png" && $0.title == "Remote title" } })
    }

    @Test func anAlreadyLocalPrintKeepsItsOwnTitleOnSyncAll() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("existing-local-title")
        let localBackend = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let remoteBackend = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        var localMutable = GalleryPrint.Mutable(versionedPrint("same.png", prompt: "recipe"))
        localMutable.title = "My title"
        localBackend.prints = [localMutable.build()]
        localBackend.mediaAnswers["same.png"] = Data([1, 2, 3])
        var remoteMutable = GalleryPrint.Mutable(versionedPrint("same.png", prompt: "recipe"))
        remoteMutable.title = "Remote title"
        remoteBackend.prints = [remoteMutable.build()]
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())

        await LibraryStore(hosts: hosts).syncAllLocally()

        #expect(localBackend.importedNames.isEmpty)
        #expect(localBackend.mutationRequests.allSatisfy { $0.titles.isEmpty })
    }

    @Test func uncertainImportResponseKeepsPendingOrganizationForRetry() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("uncertain-import")
        let localBackend = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let remoteBackend = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        var mutable = GalleryPrint.Mutable(versionedPrint("clip.mp4", prompt: "recipe"))
        mutable.title = "Remote clip"
        remoteBackend.prints = [mutable.build()]
        remoteBackend.mediaAnswer = Data([1, 2, 3])
        localBackend.importFailures = ["clip.mp4"]
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)

        await library.syncAllLocally()
        #expect(library.localSaveFailures.contains { $0.contains("clip.mp4") })

        // Simulate a server that committed before its response was lost.
        localBackend.prints = [versionedPrint("clip.mp4", prompt: "recipe")]
        localBackend.mediaAnswers["clip.mp4"] = Data([1, 2, 3])
        localBackend.importFailures = []
        await library.syncAllLocally()
        #expect(localBackend.importedNames.isEmpty)
        #expect(localBackend.mutationRequests.contains { $0.titles.contains {
            $0.filename == "clip.mp4" && $0.title == "Remote clip" } })
    }

    @Test func impossibleStagingSizeReportsFailureWithoutDownloading() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("huge-gallery-row")
        let localBackend = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let remoteBackend = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let row = try! JSONSerialization.data(withJSONObject: [
            "filename": "huge.mp4", "timestamp": 1000,
            "size_bytes": Int.max, "metadata": ["prompt": "clip"],
        ])
        remoteBackend.prints = [try! MoldJSON.decoder.decode(GalleryPrint.self, from: row)]
        let hosts = HostStore(hosts: [local, remote]) { machine in
            machine.id == local.id ? localBackend : remoteBackend
        }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)

        await library.syncAllLocally()

        #expect(remoteBackend.callCount("mediaFile") == 0)
        #expect(library.localSaveFailures.contains { $0.contains("enough free space") })
    }

    @Test func remoteTrashDoesNotSendALocalCopyToTrash() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("remote")
        let localBackend = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let remoteBackend = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let hosts = HostStore(hosts: [local, remote]) { host in
            host.id == local.id ? localBackend : remoteBackend
        }
        let library = LibraryStore(hosts: hosts)
        let remoteRow = LibraryEntry(host: remote, print: FakeFixtures.print("same.png"))
        let localRow = LibraryEntry(host: local, print: FakeFixtures.print("same.png"))
        library.perHost[remote.id] = [remoteRow]
        library.perHost[local.id] = [localRow]
        library.rebuild()

        await library.moveToTrash([remoteRow])

        #expect(remoteBackend.callCount("trash") == 1)
        #expect(localBackend.callCount("trash") == 0)
        #expect(library.items.map(\.id) == [localRow.id])
    }
    @Test func syncDoesNotRecreateDestinationTrash() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("trashed-sync-source")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let destination = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let print = versionedPrint("intentionally-trashed.png", prompt: "fox")
        source.prints = [print]
        source.mediaAnswer = Data([1, 2, 3])
        destination.trashedRows = [print]
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? destination : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAllLocally()
        #expect(destination.importedNames.isEmpty)
        #expect(source.callCount("mediaFile") == 0)
        #expect(library.localSaveReport.contains("kept removed"))
    }

    @Test func syncDoesNotRecreateRecordedCopyRemovedFromDestination() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("removed-sync-source")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let destination = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let print = versionedPrint("intentionally-deleted.png", prompt: "fox")
        source.prints = [print]
        destination.prints = [print]
        source.mediaAnswer = Data([1, 2, 3])
        destination.mediaAnswers[print.filename] = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? destination : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAllLocally()
        destination.prints = []
        await library.syncAllLocally()
        #expect(destination.importedNames.isEmpty)
        #expect(library.localSaveReport.contains("kept removed"))
    }

    @Test func replacedSourceRouteCannotPairOldListingWithNewBackend() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("original-source-route")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let destination = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        source.prints = [versionedPrint("old-route.png", prompt: "fox")]
        source.delays["gallery"] = .milliseconds(100)
        source.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? destination : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        let run = Task { await library.syncAllLocally() }
        await settle { source.callCount("gallery") > 0 }
        var replacement = remote
        replacement.baseURL = URL(string: "http://replacement-source")!
        hosts.hosts = [local, replacement]
        await run.value
        #expect(destination.importedNames.isEmpty)
        #expect(source.callCount("mediaFile") == 0)
    }

    @Test func newerShowWinsDuringSyncCollectionCreation() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("visibility-race-source")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let destination = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let slug = "sync-race-\(UUID().uuidString)"
        source.collectionRows = [Collection(id: "remote-hidden", name: slug, slug: slug, hidden: true)]
        destination.collectionCreateResponses[slug] = Collection(id: "local-hidden", name: slug, slug: slug, hidden: false)
        destination.delays["createCollection"] = .milliseconds(100)
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? destination : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.refreshOrganization()
        let run = Task { await library.syncAllLocally() }
        await settle { destination.callCount("createCollection") > 0 }
        await library.setShelfHidden(library.shelf(slug: slug)!, hidden: false)
        await run.value
        #expect(source.collectionRows.first?.hidden == false)
        #expect(destination.collectionRows.first?.hidden != true)
    }

    @Test func recurringSyncKeepsLocalCollectionRemovalButExplicitSaveCanRestoreIt() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("membership-removal-source")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let destination = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let slug = "membership-\(UUID().uuidString)"
        source.collectionRows = [Collection(id: "source-shelf", name: slug, slug: slug)]
        destination.collectionRows = [Collection(id: "destination-shelf", name: slug, slug: slug)]
        var mutable = GalleryPrint.Mutable(versionedPrint("membership.png", prompt: "fox"))
        mutable.collections = ["source-shelf"]
        source.prints = [mutable.build()]
        source.mediaAnswer = Data([1, 2, 3])
        var localPrint = GalleryPrint.Mutable(versionedPrint("membership.png", prompt: "fox"))
        localPrint.collections = ["destination-shelf"]
        destination.prints = [localPrint.build()]
        destination.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? destination : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAllLocally()
        localPrint.collections = []
        destination.prints = [localPrint.build()]
        destination.mutationRequests.removeAll()
        await library.syncAllLocally()
        #expect(destination.mutationRequests.allSatisfy { $0.addToCollection == nil })
        await library.saveLocally([LibraryEntry(host: remote, print: source.prints[0])])
        #expect(destination.mutationRequests.contains { $0.addToCollection?.name == slug })
    }

    @Test func cachedCopyReconcilesNewOrganizationAndPreservesLocalRemoval() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("organization-origin")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let destination = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let original = versionedPrint("organization.png", prompt: "fox")
        source.prints = [original]; destination.prints = [original]
        source.mediaAnswer = Data([1, 2, 3]); destination.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? destination : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAllLocally()
        var changed = GalleryPrint.Mutable(original)
        changed.tags = ["added"]; changed.title = "Origin title"; changed.favorite = true
        source.prints = [changed.build()]
        await library.syncAllLocally()
        #expect(destination.prints[0].tagList == ["added"])
        #expect(destination.prints[0].title == "Origin title")
        #expect(destination.prints[0].isFavorite)
        let key = LocalSyncRecord.key(hostID: remote.id, filename: original.filename)
        #expect(library.syncRecords.records(hosts.instanceID(of: local.id))[key]?.organization?.ownedTags == ["added"])
        var edited = GalleryPrint.Mutable(destination.prints[0])
        edited.tags = ["mine"]; edited.title = "Destination title"; edited.favorite = false
        destination.prints = [edited.build()]
        await library.syncAllLocally()
        #expect(destination.prints[0].tagList == ["mine"])
        #expect(destination.prints[0].title == "Destination title")
        #expect(!destination.prints[0].isFavorite)
        #expect(destination.importedNames.isEmpty)
        #expect(library.localSaveFailures.isEmpty)
    }

    @Test func syncCopiesToChosenRemoteDestinationWithoutRecreatingVerifiedOrigin() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("chosen-destination")
        let source = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        let destination = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        source.prints = [versionedPrint("fanout.png", prompt: "fox")]
        source.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? source : destination }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        hosts.reachability[remote.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAll(to: remote)
        #expect(destination.importedNames == ["fanout.png"])
        #expect(source.importedNames.isEmpty)
        await library.syncAllLocally()
        #expect(source.importedNames.isEmpty)
        #expect(library.localSaveFailures.isEmpty)
    }

    @Test func hiddenTrashOriginReassertsPrivacyOnVerifiedBackupWithoutRecopyingOutput() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("privacy-trash-origin")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let destination = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        source.collectionRows = [Collection(id: "source-private", name: "Private", slug: "private", hidden: true)]
        destination.collectionRows = [Collection(id: "backup-private", name: "Private", slug: "private", hidden: true)]
        var original = GalleryPrint.Mutable(versionedPrint("private-backup.png", prompt: "fox"))
        original.collections = ["source-private"]
        source.prints = [original.build()]; source.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? destination : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAllLocally()
        #expect(destination.prints[0].collectionList == ["backup-private"])
        source.trashedRows = [FakeFixtures.trashState(original.build(), at: 2)]
        source.prints = []
        var unfiled = GalleryPrint.Mutable(destination.prints[0]); unfiled.collections = []
        destination.prints = [unfiled.build()]
        let downloads = source.callCount("mediaFile")
        await library.syncAllLocally()
        #expect(destination.prints[0].collectionList == ["backup-private"])
        #expect(source.callCount("mediaFile") == downloads)
        #expect(destination.importedNames.count == 1)
        #expect(library.localSaveFailures.isEmpty)
    }

    @Test func threeDestinationRingNeverCopiesForwardedOutputsBackToOrigin() async {
        let origin = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let second = host("ring-second"), third = host("ring-third")
        let machines = [origin, second, third]
        let backends = Dictionary(uniqueKeysWithValues: machines.map {
            ($0.id, FakeBackend(host: $0, noRetainedMedia: true, syncOrganization: true))
        })
        let source = backends[origin.id]!
        source.prints = [versionedPrint("ring.png", prompt: "fox")]
        source.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: machines) { backends[$0.id]! }
        for machine in machines { hosts.reachability[machine.id] = .up(FakeFixtures.serverStatus()) }
        let library = LibraryStore(hosts: hosts)
        await library.syncAll(to: second)
        await library.syncAll(to: third)
        await library.syncAll(to: origin)
        #expect(source.importedNames.isEmpty)
        #expect(backends[second.id]?.importedNames == ["ring.png"])
        #expect(backends[third.id]?.importedNames == ["ring.png"])
        #expect(library.localSaveFailures.isEmpty)
    }

    @Test func changedSourceRecipeBeforeImportDoesNotPublishAPartialCopy() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("changed-before-copy")
        let source = FakeBackend(host: remote, noRetainedMedia: true)
        let destination = FakeBackend(host: local, noRetainedMedia: true)
        let original = versionedPrint("changed.png", prompt: "original")
        let changed = versionedPrint("changed.png", prompt: "changed")
        source.prints = [original]; source.mediaAnswer = Data([1, 2, 3])
        var reads = 0
        source.retainedTransferOfferResponder = { _ in
            reads += 1
            return .init(archiveIdentitySha256: String(repeating: "a", count: 64), members: [],
                outputSha256: "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81", outputSizeBytes: 3,
                metadata: reads == 1 ? original.metadata : changed.metadata)
        }
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? destination : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        await library.syncAllLocally()
        #expect(destination.importedNames.isEmpty)
        #expect(source.callCount("retainedSourceMediaBytes") == 0)
        #expect(library.localSaveFailures.contains { $0.contains("changed while copying") })
    }

    @Test func receiptBeforeOrganizationStillInitializesAfterRestartedRun() async {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("receipt-organization-origin")
        let source = FakeBackend(host: remote, noRetainedMedia: true, syncOrganization: true)
        let destination = FakeBackend(host: local, noRetainedMedia: true, syncOrganization: true)
        var original = GalleryPrint.Mutable(versionedPrint("receipt-org.png", prompt: "fox"))
        original.title = "Origin title"; original.tags = ["origin"]; original.favorite = true
        source.prints = [original.build()]; source.mediaAnswer = Data([1, 2, 3])
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? destination : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        // The import answered, then retained-media repair failed before organization.
        source.plantedErrors["retainedMediaTransferOffer"] = MoldClientError.unreachable("interrupted")
        var calls = 0
        source.retainedTransferOfferResponder = { _ in
            calls += 1
            if calls > 2 { throw MoldClientError.unreachable("interrupted") }
            return .init(archiveIdentitySha256: String(repeating: "a", count: 64), members: [],
                outputSha256: "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81", outputSizeBytes: 3, metadata: original.build().metadata)
        }
        source.plantedErrors.removeValue(forKey: "retainedMediaTransferOffer")
        await library.syncAllLocally()
        #expect(destination.importedNames == ["receipt-org.png"])
        source.retainedTransferOfferResponder = nil
        await library.syncAllLocally()
        #expect(destination.importedNames.count == 1)
        #expect(destination.prints.first?.title == "Origin title")
        #expect(destination.prints.first?.tagList == ["origin"])
        #expect(destination.prints.first?.isFavorite == true)
        #expect(library.localSaveFailures.isEmpty)
    }

    @Test func explicitForwardCopyReadsFreshOriginalAndHiddenMembershipFromAnotherCopy() async {
        let origin = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let backup = host("explicit-privacy-backup"), destination = host("explicit-privacy-destination")
        let machines = [origin, backup, destination]
        let backends = Dictionary(uniqueKeysWithValues: machines.map {
            ($0.id, FakeBackend(host: $0, noRetainedMedia: true, syncOrganization: true))
        })
        let original = versionedPrint("explicit-privacy.png", prompt: "fox")
        backends[origin.id]!.prints = [original]
        for backend in backends.values { backend.mediaAnswer = Data([1, 2, 3]) }
        let hosts = HostStore(hosts: machines) { backends[$0.id]! }
        for machine in machines { hosts.reachability[machine.id] = .up(FakeFixtures.serverStatus()) }
        let library = LibraryStore(hosts: hosts)
        await library.syncAll(to: backup)
        // The backup's last verified origin snapshot is stale, but the
        // original and privacy membership are both authoritatively readable.
        var changed = GalleryPrint.Mutable(original); changed.title = "Current origin"; changed.tags = ["current"]
        backends[origin.id]!.prints = [changed.build()]
        backends[backup.id]!.collectionRows = [Collection(id: "backup-private", name: "Private", slug: "private", hidden: true)]
        var filed = GalleryPrint.Mutable(backends[backup.id]!.prints[0]); filed.collections = ["backup-private"]
        backends[backup.id]!.prints = [filed.build()]
        backends[destination.id]!.collectionCreateResponses["Private"] = Collection(id: "dest-private", name: "Private", slug: "private", hidden: true)
        await library.copy([LibraryEntry(host: backup, print: filed.build())], to: destination)
        #expect(backends[destination.id]!.prints.first?.title == "Current origin")
        #expect(backends[destination.id]!.prints.first?.tagList == ["current"])
        #expect(backends[destination.id]!.prints.first?.collectionList == ["dest-private"])
        #expect(library.localSaveFailures.isEmpty)
    }

    @Test func explicitOriginalCopyReadsHiddenMembershipFromAnotherVerifiedHost() async {
        let origin = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let backup = host("hidden-evidence-backup"), destination = host("hidden-evidence-destination")
        let machines = [origin, backup, destination]
        let backends = Dictionary(uniqueKeysWithValues: machines.map {
            ($0.id, FakeBackend(host: $0, noRetainedMedia: true, syncOrganization: true))
        })
        let original = versionedPrint("hidden-evidence.png", prompt: "fox")
        backends[origin.id]!.prints = [original]
        for backend in backends.values { backend.mediaAnswer = Data([1, 2, 3]) }
        let hosts = HostStore(hosts: machines) { backends[$0.id]! }
        for machine in machines { hosts.reachability[machine.id] = .up(FakeFixtures.serverStatus()) }
        let library = LibraryStore(hosts: hosts)
        await library.syncAll(to: backup)
        backends[backup.id]!.collectionRows = [Collection(id: "backup-private", name: "Private", slug: "private", hidden: true)]
        var filed = GalleryPrint.Mutable(backends[backup.id]!.prints[0]); filed.collections = ["backup-private"]
        backends[backup.id]!.prints = [filed.build()]
        backends[destination.id]!.collectionCreateResponses["Private"] = Collection(id: "dest-private", name: "Private", slug: "private", hidden: true)
        await library.copy([LibraryEntry(host: origin, print: original)], to: destination)
        #expect(backends[destination.id]!.prints.first?.collectionList == ["dest-private"])
        #expect(library.localSaveFailures.isEmpty)
    }

    @Test func automaticRunReassertsHiddenMembershipChangedOnlyOnForwardedCopy() async {
        let origin = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let backup = host("delta-private-backup"), destination = host("delta-private-destination")
        let machines = [origin, backup, destination]
        let backends = Dictionary(uniqueKeysWithValues: machines.map {
            ($0.id, FakeBackend(host: $0, noRetainedMedia: true, syncOrganization: true))
        })
        let original = versionedPrint("delta-private.png", prompt: "fox")
        backends[origin.id]!.prints = [original]
        for backend in backends.values {
            backend.mediaAnswer = Data([1, 2, 3]); backend.syncCheckpoint = [original.filename: "v1"]
            backend.collectionRows = [Collection(id: "private", name: "Private", slug: "private", hidden: true)]
        }
        let hosts = HostStore(hosts: machines) { backends[$0.id]! }
        for machine in machines { hosts.reachability[machine.id] = .up(FakeFixtures.serverStatus()) }
        let library = LibraryStore(hosts: hosts)
        await library.syncAll(to: backup)
        await library.syncAll(to: destination)
        var filed = GalleryPrint.Mutable(backends[backup.id]!.prints[0]); filed.collections = ["private"]
        backends[backup.id]!.prints = [filed.build()]
        await library.syncAll(to: destination, fullPlan: false)
        #expect(backends[destination.id]!.prints.first?.collectionList == ["private"])
        #expect(backends[destination.id]!.importedNames.count == 1)
        #expect(library.localSaveFailures.isEmpty)
        await library.syncAll(to: origin, fullPlan: false)
        #expect(backends[origin.id]!.prints.first?.collectionList == ["private"])
        #expect(backends[origin.id]!.importedNames.isEmpty)
        #expect(library.syncRecords.verifiedOrigin(of: PrintID(host: origin.id, filename: original.filename), authority: library.syncAuthority) == nil)
    }

    @Test func syncBacksUpSoleForwardedCopyWhileRootIsOfflineWithoutPromotingBackupEdits() async {
        let origin = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let backup = host("sole-forwarded-backup"), destination = host("sole-forwarded-destination")
        let machines = [origin, backup, destination]
        let backends = Dictionary(uniqueKeysWithValues: machines.map {
            ($0.id, FakeBackend(host: $0, noRetainedMedia: true, syncOrganization: true))
        })
        var original = GalleryPrint.Mutable(versionedPrint("sole-forwarded.png", prompt: "fox"))
        original.title = "Original title"; original.tags = ["original"]
        backends[origin.id]!.prints = [original.build()]
        for backend in backends.values { backend.mediaAnswer = Data([1, 2, 3]) }
        let hosts = HostStore(hosts: machines) { backends[$0.id]! }
        for machine in machines { hosts.reachability[machine.id] = .up(FakeFixtures.serverStatus()) }
        let library = LibraryStore(hosts: hosts)
        await library.syncAll(to: backup)
        var edited = GalleryPrint.Mutable(backends[backup.id]!.prints[0])
        edited.title = "Backup-only edit"; edited.tags = ["original", "backup-only"]
        backends[backup.id]!.prints = [edited.build()]
        backends[origin.id]!.plantedErrors["gallery"] = MoldClientError.unreachable("original offline")
        await library.syncAll(to: destination)
        #expect(backends[destination.id]!.importedNames == ["sole-forwarded.png"])
        #expect(backends[destination.id]!.prints.first?.title == "Original title")
        #expect(backends[destination.id]!.prints.first?.tagList == ["original"])
        let copiedID = PrintID(host: destination.id, filename: "sole-forwarded.png")
        #expect(library.syncRecords.verifiedOrigin(of: copiedID, authority: library.syncAuthority)
            == LocalSyncRecord.key(hostID: origin.id, filename: "sole-forwarded.png"))
        backends[destination.id]!.prints = []
        await library.syncAll(to: destination)
        #expect(backends[destination.id]!.importedNames.count == 1,
            "a deliberately removed verified forwarded backup must stay removed")
    }

}
