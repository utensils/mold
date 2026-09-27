import Foundation
import Testing

@testable import MoldClient

/// Where a clip longer than one denoise goes.
/// **Fails today**: this app rendered one denoise per press and stopped the
/// Length slider at the clip size for every model.
struct ChainRoutingTests {
    /// The fixture every mold surface reads, so a drift in any one of them
    /// fails somebody's CI (`tests/fixtures/wan/surface-parity-v1.json`).
    private struct Parity: Decodable {
        struct AutoChain: Decodable { let textOnlyRefusal: Refusal }
        struct Refusal: Decodable {
            let template: String
            let totalFrames: Int
            let refused: [Row]
            let chained: [Row]
        }
        struct Row: Decodable {
            let model: String
            let sourceImage: SourceImageCapability
            let tierDefaultFrames: Int
            let clipFrames: Int
        }
        let autoChain: AutoChain
    }

    private func parity() throws -> Parity.Refusal {
        let root = try #require(RepoFixtures.repoRoot, "the mold checkout")
        let data = try Data(contentsOf: root.appending(
            path: "tests/fixtures/wan/surface-parity-v1.json"))
        return try MoldJSON.decoder.decode(Parity.self, from: data).autoChain.textOnlyRefusal
    }

    /// A one-shot may never silently become a context-free chain. The sentence
    /// is the fixture's template rendered byte for byte -- not "contains", not
    /// a paraphrase -- because every door a person can come through is
    /// supposed to say the SAME thing.
    @Test func aTextOnlyWanTierIsRefusedByNameWithTheFixturesOwnSentence() throws {
        let refusal = try parity()
        #expect(refusal.refused.count > 0, "the fixture listed no refused tiers")
        for row in refusal.refused {
            let decision = ChainRouting.decide(
                frames: refusal.totalFrames, family: "wan", model: row.model,
                sourceImage: row.sourceImage, tierDefault: row.tierDefaultFrames)
            let expected = refusal.template
                .replacingOccurrences(of: "{model}", with: row.model)
                .replacingOccurrences(of: "{total_frames}", with: "\(refusal.totalFrames)")
                .replacingOccurrences(of: "{clip_frames}", with: "\(row.clipFrames)")
            #expect(decision == .reject(expected), "\(row.model)")
        }
    }

    /// The other half of the same fixture: an image-conditioned tier is
    /// CHAINED, at the clip size the fixture names.
    @Test func anImageConditionedWanTierChainsAtTheFixturesClipSize() throws {
        let refusal = try parity()
        #expect(refusal.chained.count > 0, "the fixture listed no chainable tiers")
        for row in refusal.chained {
            let decision = ChainRouting.decide(
                frames: refusal.totalFrames, family: "wan", model: row.model,
                sourceImage: row.sourceImage, tierDefault: row.tierDefaultFrames)
            guard case let .chain(clipFrames, motionTail, _) = decision else {
                Issue.record("\(row.model) was not chained: \(decision)")
                continue
            }
            #expect(clipFrames == row.clipFrames, "\(row.model)")
            // Wan's seam re-renders exactly the one frame it was seeded with.
            #expect(motionTail == ChainRouting.wanHandoffDuplicatedFrames, "\(row.model)")
        }
    }

    /// Legacy LTX-Video takes the OTHER honest route: it stays one denoise up
    /// to its engine ceiling, and is refused by its own name past it.
    @Test func legacyLtxVideoStaysOneDenoiseUpToItsCeiling() {
        let single = ChainRouting.decide(frames: 257, family: "ltx-video",
                                         model: "ltx-video-0.9.6:bf16", advertisedMaxFrames: 257)
        #expect(single == .single())
        let over = ChainRouting.decide(frames: 300, family: "ltx-video",
                                       model: "ltx-video-0.9.6:bf16", advertisedMaxFrames: 257)
        guard case let .reject(reason) = over else {
            Issue.record("legacy LTX-Video was not refused: \(over)")
            return
        }
        #expect(reason.contains("does not support chained video generation"))
        #expect(reason.contains("257"))
    }

    /// Below the clip size nothing changes: one denoise, and no chain.
    @Test func aClipThatFitsInOneDenoiseIsNotAChain() {
        #expect(ChainRouting.decide(frames: 97, family: "ltx2", model: "ltx-2-19b:fp8")
            == .single())
        #expect(ChainRouting.decide(frames: nil, family: "ltx2", model: "ltx-2-19b:fp8")
            == .single())
    }

    /// The stage arithmetic: the first clip emits `clipFrames`, each
    /// continuation contributes the clip minus its trimmed motion tail.
    @Test func ltx2SplitsOnTheMotionTailAndStopsAtSixteenClips() {
        // 97 + ceil((249-97) / (97-17)) = 97 + ceil(152/80) = 1 + 2 = 3.
        #expect(ChainRouting.decide(frames: 249, family: "ltx-2", model: "ltx-2-19b:fp8")
            == .chain(clipFrames: 97, motionTail: 17, stageCount: 3))
        let tooLong = ChainRouting.decide(frames: 5_000, family: "ltx2", model: "ltx-2-19b:fp8")
        guard case let .reject(reason) = tooLong else {
            Issue.record("a 5000-frame chain was not refused: \(tooLong)")
            return
        }
        #expect(reason.contains("at most \(97 + 15 * 80) frames (16 clips)"))
    }
}
