import Foundation
import Testing
@testable import MoldClient

struct DraftInputSnapshotTests {
    private func temporary() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    @Test func everyLocalRoleAndParkedBytesRoundTripWithoutScopedAuthority() throws {
        var media = DraftMedia()
        media.sourceImage = "AQID"; media.sourceImageOriginal = "BAUG"
        media.sourceImageName = "fitted.png"; media.sourceImageOriginalName = "original.png"
        media.sourceImagePixels = .init(width: 10, height: 20)
        media.editImages = ["BwgJ"]
        media.maskImage = "CgsM"
        media.identity = .init(photos: [.init(encoded: "DQ4P", name: "face.png")])
        media.control = .init(image: "EBES", name: "control.png", model: "control")
        media.audioFile = "ExQV"; media.audioFileName = "sound.wav"
        media.sourceVideo = "FhcY"; media.sourceVideoName = "reference.mp4"
        media.extendVideo = "GRob"; media.extendVideoName = "extend.mp4"; media.extendOverlapFrames = 17
        media.loras = [.init(path: "adapter", scale: 0.5, name: "Style")]
        media.keyframes = [.init(frame: 4, image: "HB0e", name: "key.png")]
        media.boundaryKeyframes = ["wan-pair": [.init(frame: 0, image: "HyAh", name: "first.png")]]
        media.parked.sourceImage = "IiMk"; media.parked.maskImage = "JSYn"
        media.parked.audioFile = "KCkq"; media.parked.keyframes = [.init(frame: 8, image: "Kywt")]
        media.generationReferences = [.init(kind: "image", media: .init(authority: "inline", data: "Li8w"), mimeType: "image/png")]
        media.lastExclusiveWrite = .references
        media.sourceFit = .padFit
        let snapshot = DraftInputSnapshot(media)
        let encoded = try MoldJSON.localEncoder.encode(snapshot)
        let decoded = try MoldJSON.localDecoder.decode(DraftInputSnapshot.self, from: encoded)
        var restored = DraftMedia()
        try decoded.apply(to: &restored)
        #expect(restored == media)
        var scoped = media
        scoped.generationReferences[0].media = .init(authority: "upload", data: "PRIVATE", handle: "LEASE", path: "/secret/path")
        scoped.parked.generationReferences = scoped.generationReferences
        let safe = DraftInputSnapshot(scoped)
        let text = String(decoding: try MoldJSON.localEncoder.encode(safe), as: UTF8.self)
        #expect(!text.contains("LEASE")); #expect(!text.contains("/secret/path")); #expect(!text.contains("PRIVATE"))
        #expect(safe.active.generationReferences[0].media.authority == "descriptor")
        #expect(safe.parked.generationReferences[0].media.authority == "descriptor")
    }

    @Test func boundaryProtocolsAndManualPortraitCanvasSurviveReadoption() throws {
        let h3 = try MoldJSON.decoder.decode(RecipeCapabilities.self, from: Data(#"{"boundary_frames":{"mode":"adjustable","wire":"h3-endpoints","min_frames":9,"first_required":true,"last_required":false}}"#.utf8))
        let interpolation = try MoldJSON.decoder.decode(RecipeCapabilities.self, from: Data(#"{}"#.utf8))
        var draft = RenderDraft()
        draft.width = 704; draft.height = 1280; draft.canvasIntent = .manual
        draft.media.adoptedReferenceCapabilities = h3
        draft.media.sourceImage = "AQID"; draft.media.sourceImagePixels = .init(width: 1024, height: 1024)
        draft.media.keyframes = [.init(frame: 120, image: "BAUG")]
        draft.media.boundaryKeyframes["interpolation"] = [.init(frame: 40, image: "BwgJ")]
        let descriptor = DraftDescriptor(draft, model: "h3", family: "h3", recipeID: nil)
        var restored = RenderDraft()
        descriptor.apply(to: &restored)
        try DraftInputSnapshot(draft.media).apply(to: &restored.media)
        BoundaryFramePolicy.reconcile(media: &restored.media, capabilities: h3)
        restored.media.adoptedReferenceCapabilities = h3
        #expect(restored.media.keyframes == draft.media.keyframes)
        #expect(restored.media.sourceImage == "AQID")
        #expect(restored.width == 704 && restored.height == 1280)
        #expect(restored.media.sourceImagePixels == .init(width: 1024, height: 1024))
        BoundaryFramePolicy.reconcile(media: &restored.media, capabilities: interpolation)
        #expect(restored.media.keyframes.map(\.frame) == [40])
        #expect(restored.media.keyframes.map(\.image) == ["BwgJ"])
    }

    @Test func snapshotFailurePreservesPreviousDocumentAndCachedCorruptionIsRepaired() throws {
        let directory = try temporary()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = DraftStore(directory: directory)
        var draft = RenderDraft(); draft.media.sourceImage = "AQID"
        let descriptor = DraftDescriptor(draft, model: "model", family: nil, recipeID: nil)
        let inputs = DraftInputSnapshot(draft.media)
        #expect(store.save(descriptor, inputs: inputs))
        let saved = try #require(store.load())
        let digest = try #require(saved.localInputsSHA256)
        let file = store.inputsDirectory.appending(path: "\(digest).json")
        let mode = try FileManager.default.attributesOfItem(atPath: file.path)[.posixPermissions] as? NSNumber
        #expect(mode?.intValue == 0o600)
        try Data("broken".utf8).write(to: file)
        #expect(throws: (any Error).self) { try store.loadInputs(for: saved) }
        #expect(store.save(descriptor, inputs: inputs))
        #expect(try store.loadInputs(for: try #require(store.load())) == inputs)
        let prior = try Data(contentsOf: store.url)
        try FileManager.default.removeItem(at: store.inputsDirectory)
        try Data().write(to: store.inputsDirectory)
        draft.media.sourceImage = "BAUG"
        #expect(!store.save(DraftDescriptor(draft, model: "changed", family: nil, recipeID: nil), inputs: DraftInputSnapshot(draft.media)))
        #expect(try Data(contentsOf: store.url) == prior)
    }

    @Test func missingOversizedAndWrongDigestInputsNeverRestoreAsEmpty() throws {
        let directory = try temporary()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = DraftStore(directory: directory)
        let descriptor = DraftDescriptor(RenderDraft(), model: nil, family: nil, recipeID: nil)
        #expect(store.save(descriptor, inputs: DraftInputSnapshot(DraftMedia())))
        let saved = try #require(store.load())
        let file = store.inputsDirectory.appending(path: "\(try #require(saved.localInputsSHA256)).json")
        try FileManager.default.removeItem(at: file)
        #expect(throws: (any Error).self) { try store.loadInputs(for: saved) }
        FileManager.default.createFile(atPath: file.path, contents: nil)
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: UInt64(DraftStore.maximumInputSnapshotBytes) + 1)
        try handle.close()
        #expect(throws: (any Error).self) { try store.loadInputs(for: saved) }
        var traversal = saved; traversal.localInputsSHA256 = "../elsewhere"
        #expect(throws: (any Error).self) { try store.loadInputs(for: traversal) }
    }

    @Test func quitReservationPreventsOlderWritesAndAttachmentOnlyChangesPersist() throws {
        let directory = try temporary()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = DraftStore(directory: directory)
        let descriptor = DraftDescriptor(RenderDraft(), model: "model", family: nil, recipeID: nil)
        var old = DraftMedia(); old.sourceImage = "AQID"
        var current = old; current.sourceImage = "BAUG"
        let older = store.reserveWrite(), newest = store.reserveWrite()
        #expect(store.save(descriptor, inputs: DraftInputSnapshot(current), revision: newest))
        #expect(!store.save(descriptor, inputs: DraftInputSnapshot(old), revision: older))
        #expect(try store.loadInputs(for: try #require(store.load())) == DraftInputSnapshot(current))
        #expect(DraftInputSnapshot(old) != DraftInputSnapshot(current))
        store.clear()
        #expect(store.load() == nil)
        #expect(!FileManager.default.fileExists(atPath: store.inputsDirectory.path))
    }
}
