import AppKit
import Foundation
import MoldClient
import Testing

@testable import Mold

@MainActor
struct LibraryAttachmentTests {
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
        #expect(!fence.permits(controller, reuse: reuse))
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
