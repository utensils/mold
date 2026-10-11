import AppKit
import Foundation
import MoldClient
import Testing

@testable import Mold

@MainActor
struct LibraryAttachmentTests {
    @Test func typedImageReferencesAreOfferedByCapabilityAndRespectTheirLimit() throws {
        var json = try #require(JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(FakeFixtures.recipe())) as? [String: Any])
        json["capabilities"] = ["generation_references": ["mode": "adjustable", "required": true,
            "kinds": ["image", "video", "audio"], "max_count": 12, "max_images": 9,
            "max_videos": 3, "max_audios": 3, "min_duration_ms": 2000,
            "max_duration_ms": 15000, "max_video_duration_ms": 15000,
            "max_audio_duration_ms": 15000, "max_inline_bytes": 33554432, "requires_visual": true]]
        let recipe = try MoldJSON.decoder.decode(GenerationRecipe.self, from: JSONSerialization.data(withJSONObject: json))
        let model = FakeFixtures.model("arbitrary-capable-model")
        var media = DraftMedia()
        #expect(LibraryAttachmentOffer.resolve(recipe: recipe, model: model, media: media).canAddReference)
        media.generationReferences = (0..<9).map { _ in
            GenerationReference(kind: "image", media: .init(authority: "inline", data: "AA=="), mimeType: "image/png")
        }
        #expect(!LibraryAttachmentOffer.resolve(recipe: recipe, model: model, media: media).canAddReference)
    }

    private func entry(_ filename: String, _ format: String, trashed: UInt64? = nil) throws
        -> (MoldHost, LibraryEntry) {
        let host = MoldHost(name: "workstation", baseURL: URL(string: "http://w")!)
        let trash = trashed.map { ",\"trashed_at\":\($0)" } ?? ""
        let data = Data("""
        {"filename":"\(filename)","metadata":{"prompt":"p","model":"m","seed":1,
        "steps":4,"guidance":1,"width":8,"height":8,"version":"1"},
        "timestamp":1,"format":"\(format)"\(trash)}
        """.utf8)
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: data)
        return (host, LibraryEntry(host: host, print: print))
    }

    private func stores() -> (GenerateController, ReuseStore, MoldHost) {
        let host = MoldHost(name: "workstation", baseURL: URL(string: "http://w")!)
        let backend = FakeBackend(host: host)
        let hosts = HostStore(hosts: [host]) { _ in backend }
        return (GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts)),
                ReuseStore(hosts: hosts), host)
    }

    @Test func fenceRejectsEveryStateThatCanRetargetAnAttachment() {
        let (controller, reuse, host) = stores()
        controller.hostID = host.id
        controller.modelName = "flux"
        let fence = DraftAttachmentFence(controller: controller, reuse: reuse)
        #expect(fence.permits(controller, reuse: reuse))

        controller.draft.prompt = "moved"
        #expect(fence.permits(controller, reuse: reuse)) // scalar edits remain live
        controller.draft.media.sourceImage = "replacement"
        #expect(!fence.permits(controller, reuse: reuse))
        controller.draft.media.sourceImage = nil
        controller.draft.prompt = ""
        controller.recipeID = "edit"
        #expect(!fence.permits(controller, reuse: reuse))
        controller.recipeID = nil
        controller.machineChoice = host.id
        #expect(!fence.permits(controller, reuse: reuse))
        controller.machineChoice = nil
        reuse.clear()
        #expect(!fence.permits(controller, reuse: reuse))
    }

    @Test func openingLibraryCancelsEarlierImportBeforeCapturingItsDestination() async {
        let (controller, _, _) = stores()
        var release: CheckedContinuation<Void, Never>?
        var earlier: Task<Void, Never>? = Task { @MainActor in
            await withCheckedContinuation { release = $0 }
            guard !Task.isCancelled else { return }
            controller.draft.media.editImages.append("old file completion")
        }
        while release == nil { await Task.yield() }
        let oldTask = earlier
        let librarySession = PictureImportSession.begin(controller: controller, replacing: &earlier)
        #expect(earlier == nil)
        #expect(oldTask?.isCancelled == true)
        release?.resume()
        await oldTask?.value
        #expect(controller.draft.media.editImages.isEmpty)
        #expect(librarySession.isCurrent)
        controller.draft.media.editImages.append("new library choice")
        librarySession.advance()
        #expect(librarySession.isCurrent)
    }

    @Test func pictureChooserSessionAdvancesOnlyItsOwnAcceptedInputs() {
        let (controller, _, _) = stores()
        let session = PictureImportSession(controller: controller)
        controller.draft.prompt = "keep this live edit"
        #expect(session.isCurrent)
        controller.draft.media.editImages.append("first accepted picture")
        #expect(!session.isCurrent)
        session.advance()
        #expect(session.isCurrent)
        controller.draft.media.editImages.append("unrelated replacement")
        #expect(!session.isCurrent)
        let next = PictureImportSession(controller: controller)
        controller.modelName = "different model"
        #expect(!next.isCurrent)
    }

    @Test func statusOnlyHostCanAttachWithoutAnSSEAuthorityAnnouncement() throws {
        let (source, raster) = try entry("still.png", "png")
        let target = MoldHost(name: "target", baseURL: URL(string: "http://target")!)
        let hosts = HostStore(hosts: [source, target]) { FakeBackend(host: $0) }
        hosts.reachability[source.id] = .up(FakeFixtures.serverStatus(instanceId: "source-status"))
        hosts.reachability[target.id] = .up(FakeFixtures.serverStatus(instanceId: "target-status"))
        let library = LibraryStore(hosts: hosts)
        library.perHost[source.id] = [raster]
        library.rebuild()
        #expect(hosts.instanceIDs[source.id] == nil)
        let routes = LibraryAttachmentRoutes.capture(source: source.id, destination: target, hosts: hosts)
        #expect(routes.sourceInstance == "source-status")
        #expect(routes.destinationInstance == "target-status")
        #expect(PictureSource.isCurrent(raster, origin: source, instance: routes.sourceInstance,
                                       hosts: hosts, library: library))
        hosts.reachability[source.id] = .up(FakeFixtures.serverStatus(instanceId: "restarted"))
        #expect(!PictureSource.isCurrent(raster, origin: source, instance: routes.sourceInstance,
                                        hosts: hosts, library: library))
    }

    @Test func attachmentRoutesRejectCredentialAndInstanceChanges() {
        let source = MoldHost(name: "source", baseURL: URL(string: "http://source")!)
        let target = MoldHost(name: "target", baseURL: URL(string: "http://target")!)
        let initial = LibraryAttachmentRoutes(source: source, sourceInstance: "source-1",
                                              destination: target, destinationInstance: "target-1")
        #expect(initial != LibraryAttachmentRoutes(source: source, sourceInstance: "source-2",
                                                    destination: target, destinationInstance: "target-1"))
        #expect(initial != LibraryAttachmentRoutes(source: source, sourceInstance: "source-1",
                                                    destination: target, destinationInstance: "target-2"))
        #expect(initial != LibraryAttachmentRoutes(source: nil, sourceInstance: "source-1",
                                                    destination: target, destinationInstance: "target-1"))
        var rekeyed = source
        rekeyed.apiKey = "replacement"
        #expect(initial != LibraryAttachmentRoutes(source: rekeyed, sourceInstance: "source-1",
                                                    destination: target, destinationInstance: "target-1"))
    }

    @Test func asyncCompletionRejectsStaleUIAndHostState() async throws {
        let (_, raster) = try entry("still.png", "png")
        let cursor = LibraryCursor.Selection(
            items: [raster.id], anchor: raster.id, lead: raster.id)
        let transaction = LibraryAttachmentTransaction(
            version: 4, cursor: cursor, viewed: nil, scope: .all, destination: .library)
        await Task.yield() // the authenticated media read completed here

        func permitted(
            destination: Destination = .library,
            cursor current: LibraryCursor.Selection? = nil,
            live: LibraryEntry? = raster, sourceUp: Bool = true
        ) -> Bool {
            transaction.permits(
                version: 4, cursor: current ?? cursor, viewed: nil, scope: .all,
                destination: destination, draftIsCurrent: true,
                liveEntry: live, sourceIsUp: sourceUp)
        }

        #expect(permitted())
        #expect(!permitted(destination: .queue))
        #expect(!permitted(cursor: .empty))
        #expect(!permitted(live: nil))
        #expect(!permitted(sourceUp: false))
    }

    @Test func libraryFirstRestoreRunsBeforeAnyAttachmentAdoptsAModel() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString, directoryHint: .isDirectory)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = DraftStore(directory: directory)
        var saved = RenderDraft()
        saved.prompt = "the prompt that must survive"
        store.save(DraftDescriptor(
            saved, model: "saved-model", family: "flux", recipeID: nil))
        let drafts = DraftPersistence(store: store, delay: .zero)
        let (controller, _, _) = stores()

        drafts.restore(into: controller)

        #expect(controller.draft.prompt == saved.prompt)
        #expect(drafts.restoredModel == "saved-model")
        #expect(controller.modelFamily == "flux")
    }

    @Test func onlyTheRestoredModelKeepsTheAuthoredDraftOnFirstAdoption() {
        #expect(AttachmentModelAdoption.keepsDraft(
            restoredModel: "flux-dev:q8", adopting: "flux-dev:q8"))
        #expect(!AttachmentModelAdoption.keepsDraft(
            restoredModel: nil, adopting: "flux-dev:q8"))
        #expect(!AttachmentModelAdoption.keepsDraft(
            restoredModel: "missing-model", adopting: "flux-dev:q8"))
    }

    @Test func onlyALiveRasterCanEnterAPictureWell() throws {
        #expect(try entry("still.png", "png").1.isAttachableRaster)
        #expect(try entry("qwen.png", "PNG").1.isAttachableRaster)
        #expect(try entry("qwen.png", "image/png").1.isAttachableRaster)
        #expect(try entry("photo.heic", "heic").1.isAttachableRaster)
        #expect(try entry("photo.tiff", "tiff").1.isAttachableRaster)
        #expect(try !entry("vector.svg", "svg").1.isAttachableRaster)
        #expect(try !entry("sound.wav", "wav").1.isAttachableRaster)
        #expect(try !entry("clip.mp4", "mp4").1.isAttachableRaster)
        #expect(try !entry("gone.png", "png", trashed: 1).1.isAttachableRaster)
    }

    @Test func sharedActionRouterDeliversTheExactRaster() throws {
        let (host, raster) = try entry("still.png", "png")
        let backend = FakeBackend(host: host)
        let hosts = HostStore(hosts: [host]) { _ in backend }
        var delivered: PrintID?
        let actions = LibraryActions(
            hosts: hosts, library: LibraryStore(hosts: hosts),
            useAsSource: { delivered = $0.id })

        actions.perform(.useAsSourceImage, on: [raster], scope: .all)

        #expect(delivered == raster.id)

        let (_, audio) = try entry("sound.wav", "wav")
        delivered = nil
        actions.perform(.useAsSourceImage, on: [audio], scope: .all)
        #expect(delivered == nil)
    }

    @Test func sourceAttachmentKeepsOriginalBytesNamePixelsAndShapeIntent() throws {
        let rep = try #require(NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: 8, pixelsHigh: 4,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
        let data = try #require(rep.representation(using: .png, properties: [:]))
        let picked = ImportedPicture(
            encoded: data.base64EncodedString(), name: "source.png", data: data)
        var draft = RenderDraft()

        DraftPictureAttachment.useAsSource(picked, in: &draft, recipe: nil)

        #expect(draft.media.sourceImage == picked.encoded)
        #expect(draft.media.sourceImageOriginal == picked.encoded)
        #expect(draft.media.sourceImageName == "source.png")
        #expect(draft.media.sourceImagePixels == SourcePixels(width: 8, height: 4))
        #expect(draft.canvasIntent == .source)
    }
}
