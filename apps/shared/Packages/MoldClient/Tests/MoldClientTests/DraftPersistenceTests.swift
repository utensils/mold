import Foundation
import Testing

@testable import MoldClient

/// The draft across launches. **Fails today**: nothing persisted it, so every
/// launch opened on an empty pane.
struct DraftPersistenceTests {
    private func temporary() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "mold-draft-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// A draft holding EVERY byte-bearing root this app has. The persisted
    /// document must contain none of them.
    private func loadedDraft() -> RenderDraft {
        var draft = RenderDraft()
        draft.prompt = "a tin robot"
        draft.media.sourceImage = "SOURCEBYTES"
        draft.media.sourceImageName = "s.png"
        draft.media.sourceImageOriginal = "ORIGINALBYTES"
        draft.media.sourceImageOriginalName = "s.png"
        draft.media.editImages = ["REFBYTES"]
        draft.media.maskImage = "MASKBYTES"
        draft.media.identity = IdentityConditioning(
            photos: [IdentityPhoto(encoded: "FACEBYTES", name: "face.png")])
        draft.media.control = ControlConditioning(
            image: "CONTROLBYTES", model: "controlnet-canny-sd15:fp16", scale: 1)
        draft.media.keyframes = [KeyframeCondition(frame: 1, image: "KEYBYTES", name: "k.png")]
        draft.media.extendVideo = "EXTENDBYTES"
        draft.media.audioFile = "AUDIOBYTES"
        draft.media.sourceVideo = "VIDEOBYTES"
        draft.media.parked.sourceImage = "PARKEDBYTES"
        return draft
    }

    /// The whole point of a DESCRIPTOR. Every one of these sentinels is a
    /// base64 payload somewhere in the draft; not one may reach the disk.
    @Test func noByteBearingRootReachesTheFile() throws {
        let store = DraftStore(directory: temporary())
        store.save(DraftDescriptor(loadedDraft(), model: "m", family: "f",
                                   recipeID: nil))
        let written = try String(contentsOf: store.url, encoding: .utf8)
        let sentinels = ["SOURCEBYTES", "ORIGINALBYTES", "REFBYTES", "MASKBYTES", "FACEBYTES",
                         "CONTROLBYTES", "KEYBYTES", "EXTENDBYTES", "AUDIOBYTES", "VIDEOBYTES",
                         "PARKEDBYTES", "s.png", "face.png", "k.png"]
        #expect(sentinels.count > 0)
        for sentinel in sentinels {
            #expect(written.contains(sentinel) == false, "\(sentinel) reached the draft file")
        }
    }

    /// A restore puts back what was asked for and leaves the MEDIA alone --
    /// so a live session's staged picture is not cleared by a descriptor that
    /// knows nothing about it.
    @Test func everythingButTheMediaRoundTrips() throws {
        var draft = loadedDraft()
        draft.title = "Robots"
        draft.tags = ["metal"]
        draft.steps = 33
        draft.canvasIntent = .manual
        draft.media.sourceFit = .padFit
        draft.advanced.scheduler = "uni-pc"
        draft.advanced.cfgPlus = true
        draft.advanced.stgBlocks = "3, 7"
        draft.advanced.skipStep = 2

        let store = DraftStore(directory: temporary())
        store.save(DraftDescriptor(draft, model: "flux-dev:q4", family: "flux",
                                   recipeID: "auto"))
        let restored = try #require(store.load())
        #expect(restored.model == "flux-dev:q4")

        var fresh = RenderDraft()
        fresh.media.sourceImage = "SOMETHINGELSE"
        restored.apply(to: &fresh)
        #expect(fresh.title == "Robots")
        #expect(fresh.tags == ["metal"])
        #expect(fresh.steps == 33)
        #expect(fresh.canvasIntent == .manual)
        #expect(fresh.media.sourceFit == .padFit)
        #expect(fresh.advanced.scheduler == "uni-pc")
        #expect(fresh.advanced.cfgPlus)
        #expect(fresh.advanced.stgBlocks == "3, 7")
        #expect(fresh.advanced.skipStep == 2)
        // The media it was handed is untouched.
        #expect(fresh.media.sourceImage == "SOMETHINGELSE")
    }

    /// A version this build does not write is discarded WHOLE. A half-restored
    /// draft is worse than an empty one: nothing on screen says which half is
    /// real.
    @Test func anotherVersionIsDiscardedAndParked() throws {
        let directory = temporary()
        let store = DraftStore(directory: directory)
        var descriptor = DraftDescriptor(RenderDraft(), model: "m", family: nil,
                                         recipeID: nil)
        descriptor.version = DraftDescriptor.currentVersion + 1
        store.save(descriptor)

        #expect(store.load() == nil)
        // Parked beside itself rather than clobbered.
        #expect(FileManager.default.fileExists(
            atPath: store.url.appendingPathExtension("corrupt").path(percentEncoded: false)))
        #expect(FileManager.default.fileExists(atPath: store.url.path(percentEncoded: false))
            == false)
    }

    /// A document that will not parse is parked, not fatal -- and a SECOND
    /// failure does not overwrite the first copy, which is the one worth
    /// keeping.
    @Test func aCorruptDocumentIsParkedOnceAndTheNextLaunchIsEmpty() throws {
        let store = DraftStore(directory: temporary())
        try FileManager.default.createDirectory(
            at: store.url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("not json at all".utf8).write(to: store.url)

        #expect(store.load() == nil)
        let parked = store.url.appendingPathExtension("corrupt")
        #expect(try String(contentsOf: parked, encoding: .utf8) == "not json at all")

        try Data("also not json".utf8).write(to: store.url)
        #expect(store.load() == nil)
        #expect(try String(contentsOf: parked, encoding: .utf8) == "not json at all")
    }

    /// The local coders have NO key strategy, because the wire pair are not
    /// inverses -- `recipeID` would be written `recipe_id` and read back as
    /// `recipeId`, and the field would be nil for ever (the `StoredHost`
    /// trap). This asserts the spelling on disk, which is the only place the
    /// mistake would show.
    @Test func theDocumentSpellsItsKeysExactlyAsTheTypeDoes() throws {
        let store = DraftStore(directory: temporary())
        store.save(DraftDescriptor(RenderDraft(), model: "m", family: "f",
                                   recipeID: "auto"))
        let written = try String(contentsOf: store.url, encoding: .utf8)
        #expect(written.contains("\"recipeID\""))
        #expect(written.contains("\"negativePrompt\""))
        #expect(written.contains("recipe_id") == false)
        #expect(written.contains("negative_prompt") == false)
    }

    @Test func legacyFalseAudioRemainsAnExplicitSavedPreference() throws {
        let descriptor = DraftDescriptor(RenderDraft(), model: "m", family: "ltx2",
                                         recipeID: "auto")
        var object = try #require(JSONSerialization.jsonObject(
            with: MoldJSON.localEncoder.encode(descriptor)) as? [String: Any])
        object.removeValue(forKey: "preferredAudio")
        object.removeValue(forKey: "hasAudioPreference")
        object["enableAudio"] = false
        let legacy = try MoldJSON.localDecoder.decode(
            DraftDescriptor.self, from: JSONSerialization.data(withJSONObject: object))

        var restored = RenderDraft()
        legacy.apply(to: &restored)
        #expect(restored.preferredAudio == false)
    }
}
