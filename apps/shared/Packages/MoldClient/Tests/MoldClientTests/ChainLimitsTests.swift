import Foundation
import Testing

@testable import MoldClient

/// The HOST's own chain limits, which outrank every constant this app carries.
/// **Fails today**: the router read only its ported constants, so a machine
/// configured with a different clip size was contradicted by its own client.
struct ChainLimitsTests {
    private func limits(
        clipCap: Int = 97, recommended: Int = 97, maxStages: Int = 16,
        maxTotal: Int = 1_297, supportsSequence: Bool = true, reason: String? = nil
    ) -> ChainLimits {
        let json = """
        {"model": "ltx-2-19b:fp8", "frames_per_clip_cap": \(clipCap), "fps": 24,
         "frames_per_clip_recommended": \(recommended), "max_stages": \(maxStages),
         "max_total_frames": \(maxTotal), "fade_frames_max": 16,
         "transition_modes": ["smooth"], "quantization_family": "fp8",
         "supports_audio": true, "supports_sequence": \(supportsSequence),
         "sequence_unsupported_reason": \(reason.map { "\"\($0)\"" } ?? "null")}
        """
        return try! MoldJSON.decoder.decode(ChainLimits.self, from: Data(json.utf8))
    }

    /// The advertised clip size replaces the client constant, one for one --
    /// a host whose routing clip is 49 splits at 49, not at 97.
    @Test func theAdvertisedClipSizeIsWhatTheRouterSplitsOn() {
        let decision = ChainRouting.decide(
            frames: 200, family: "ltx2", model: "ltx-2-19b:fp8",
            limits: limits(clipCap: 49, recommended: 49))
        // 49 + ceil((200-49) / (49-17)) = 1 + 5 = 6.
        #expect(decision == .chain(clipFrames: 49, motionTail: 17, stageCount: 6))
    }

    /// And the advertised stage cap is the one that refuses.
    @Test func theAdvertisedStageCapIsWhatRefuses() {
        let decision = ChainRouting.decide(
            frames: 400, family: "ltx2", model: "ltx-2-19b:fp8",
            limits: limits(maxStages: 2, maxTotal: 10_000))
        guard case let .reject(reason) = decision else {
            Issue.record("a 400-frame render was not refused: \(decision)")
            return
        }
        #expect(reason.contains("(2 clips)"))
    }

    /// A total ceiling is a DIFFERENT number from the stage cap times the
    /// clip, and the host enforces it separately.
    @Test func theAdvertisedTotalCeilingRefusesOnItsOwn() {
        let decision = ChainRouting.decide(
            frames: 300, family: "ltx2", model: "ltx-2-19b:fp8", limits: limits(maxTotal: 250))
        #expect(decision == .reject(
            "Chained video supports at most 250 frames for this model. Reduce the frame count."))
    }

    /// A host that says this model has no sequence path is the authority, and
    /// its own sentence is the one shown.
    @Test func aHostThatRefusesSequencesIsTheAuthorityAndSaysWhy() {
        let decision = ChainRouting.decide(
            frames: 200, family: "ltx2", model: "ltx-2-19b:fp8",
            limits: limits(supportsSequence: false, reason: "This build has no stitcher."))
        #expect(decision == .reject("This build has no stitcher."))
        // And below the cap it is still one perfectly good denoise.
        #expect(ChainRouting.decide(
            frames: 90, family: "ltx2", model: "ltx-2-19b:fp8",
            limits: limits(supportsSequence: false)) == .single())
    }

    /// The contracts the limits block does NOT answer stay where they are: a
    /// wan tier that hands nothing across a seam is still refused by name,
    /// however generous the host's numbers are.
    @Test func advertisedLimitsNeverOverruleTheTextOnlyRefusal() {
        let decision = ChainRouting.decide(
            frames: 259, family: "wan", model: "wan21-t2v-1.3b:bf16",
            limits: limits(clipCap: 257, recommended: 121, maxTotal: 10_000),
            sourceImage: .unsupported)
        guard case let .reject(reason) = decision else {
            Issue.record("a text-only wan tier was chained: \(decision)")
            return
        }
        #expect(reason.hasPrefix("'wan21-t2v-1.3b:bf16' is text-to-video"))
    }
}
