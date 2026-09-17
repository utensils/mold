import Foundation
import Testing

@testable import MoldClient

// What the Length control may offer (findings 01#3, 02#3).

private func temporal(_ name: String, _ id: String = "default") throws -> TemporalProfile {
    let set = try MoldJSON.decoder.decode(
        GenerationProfileSet.self, from: RepoFixtures.fixture(name))
    return try #require(set.recipe(named: id)?.temporal)
}

/// **Fails today**: `maxDurationSeconds` is decoded and read nowhere, so the
/// slider ran to LTX-2's 120-fps grid maximum (601) at 24 fps, reported
/// "25.0s" against a 20-second budget, and every value from 489 up was a hard
/// 422 at submit.
@Test func ltx2sCeilingIsTheDurationCapAtTheChosenRate() throws {
    let ltx2 = try temporal("recipe-ltx2.json", "auto")
    #expect(ltx2.frames.max == 601)
    #expect(ltx2.maxDurationSeconds == 20)

    // `20 x 24 + 1 = 481`, exactly what admission narrows to
    // (`generation_profile.rs:1241-1251`).
    #expect(ltx2.durationCappedMaxFrames(fps: 24) == 481)
    // The ceiling MOVES with the rate -- that is the whole reason the server
    // advertises the 120-fps figure.
    #expect(ltx2.durationCappedMaxFrames(fps: 120) == 601)
    #expect(ltx2.durationCappedMaxFrames(fps: 16) == 321)

    let bounds = ltx2.lengthBounds(fps: 24, family: "ltx2", model: "ltx2-2.3-13b:q8",
                                  sourceImage: .optional)
    #expect(bounds.max == 481)
    // Nothing is refused here, so there is nothing to explain.
    #expect(bounds.note == nil)
    #expect((bounds.max - 1) % ltx2.frames.step == 0)
}

@Test func aRecipeWithNoDurationBudgetKeepsItsAdvertisedMaximum() throws {
    let wan = try temporal("recipe-wan.json")
    #expect(wan.maxDurationSeconds == nil)
    #expect(wan.durationCappedMaxFrames(fps: 16) == 257)
}

/// **Fails today**: `frames.max` for wan is `MAX_FRAMES_GLOBAL`, a memory
/// guard -- so the slider offered 257 frames of a checkpoint that renders its
/// trained clip and then repeats it.
@Test func aTextOnlyWanTierStopsAtItsOwnClip() throws {
    let wan = try temporal("recipe-wan.json")
    let bounds = wan.lengthBounds(fps: 16, family: "wan", model: "wan22-t2v-a14b:q8",
                                 sourceImage: .unsupported)
    // The tier's own recorded default (73) over the A14B floor (53).
    #expect(bounds.max == 73)
    let note = try #require(bounds.note)
    #expect(note.contains("wan22-t2v-a14b:q8"))
    #expect(note.contains("73"))
}

/// An image-to-video tier CAN be continued, so nothing is capped and nothing
/// is explained.
@Test func anImageToVideoWanTierKeepsTheAdvertisedCeiling() throws {
    let wan = try temporal("recipe-wan.json")
    let bounds = wan.lengthBounds(fps: 16, family: "wan", model: "wan22-ti2v-5b:turbo",
                                 sourceImage: .optional)
    #expect(bounds.max == 257)
    #expect(bounds.note == nil)
}

@Test func snappingDownNeverRoundsBackPastACeiling() throws {
    let wan = try temporal("recipe-wan.json")
    // Nearest would give 121; a ceiling has to hold.
    #expect(wan.snapDown(123) == 121)
    #expect(wan.snapDown(121) == 121)
    #expect(wan.snapDown(120) == 117)
    for requested in stride(from: 1, through: 257, by: 7) {
        #expect((wan.snapDown(requested) - 1) % 4 == 0)
    }
}

/// The clip sizes are the fleet's, not this app's: pinned against the shared
/// cross-surface fixture rather than against a number typed twice.
@Test func theClipCeilingsAgreeWithTheSharedWanFixture() throws {
    guard let root = RepoFixtures.repoRoot else { return }
    let data = try Data(contentsOf: root.appending(
        path: "tests/fixtures/wan/surface-parity-v1.json"))
    let json = try #require(
        JSONSerialization.jsonObject(with: data) as? [String: Any])
    let chain = try #require(json["auto_chain"] as? [String: Any])
    let refusal = try #require(chain["text_only_refusal"] as? [String: Any])

    let refused = try #require(refusal["refused"] as? [[String: Any]])
    #expect(!refused.isEmpty)
    for row in refused {
        let model = try #require(row["model"] as? String)
        let tierDefault = row["tier_default_frames"] as? Int
        let expected = try #require(row["clip_frames"] as? Int)
        #expect(ClipLengthBounds.singleClipCeiling(
            family: "wan", model: model, sourceImage: .unsupported,
            tierDefault: tierDefault) == expected, "clip ceiling for \(model)")
    }

    // A tier that CAN be chained has no single-clip ceiling to key on at all.
    for row in try #require(refusal["chained"] as? [[String: Any]]) {
        let model = try #require(row["model"] as? String)
        #expect(ClipLengthBounds.singleClipCeiling(
            family: "wan", model: model, sourceImage: .optional,
            tierDefault: row["tier_default_frames"] as? Int) == nil, "\(model) is continuable")
    }
}

@Test func anotherFamilyIsNeverGivenWansClipRule() {
    #expect(ClipLengthBounds.singleClipCeiling(
        family: "ltx2", model: "ltx2-2.3-13b:q8", sourceImage: .unsupported,
        tierDefault: 121) == nil)
    #expect(ClipLengthBounds.singleClipCeiling(
        family: nil, model: nil, sourceImage: .unsupported, tierDefault: nil) == nil)
}
