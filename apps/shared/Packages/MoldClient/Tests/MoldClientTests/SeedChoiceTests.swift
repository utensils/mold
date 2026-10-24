import Testing
@testable import MoldClient

struct SeedChoiceTests {
    @Test func choosingFixedWithoutASeedChoosesOneOnce() {
        var draft = RenderDraft()
        SeedChoice.select(.fixed, in: &draft, newSeed: 42)
        #expect(draft.locksSeed)
        #expect(draft.seed == 42)
        #expect(RenderRequest.one(draft, model: "fixture").seed == 42)
        SeedChoice.select(.fixed, in: &draft, newSeed: 99)
        #expect(draft.seed == 42)
    }

    @Test func randomModeOmitsTheRememberedSeedAndFixedRestoresIt() {
        var draft = RenderDraft()
        draft.seed = UInt64.max
        draft.locksSeed = true
        SeedChoice.select(.random, in: &draft, newSeed: 42)
        #expect(!draft.locksSeed)
        #expect(draft.seed == UInt64.max)
        #expect(SeedChoice.Mode.resolve(draft) == .random)
        #expect(RenderRequest.one(draft, model: "fixture").seed == nil)
        SeedChoice.select(.fixed, in: &draft, newSeed: 42)
        #expect(draft.seed == UInt64.max)
        #expect(RenderRequest.one(draft, model: "fixture").seed == UInt64.max)
    }

    @Test func newSeedChangesTheFixedStartingSeedWithoutChangingBatchSemantics() {
        var draft = RenderDraft()
        draft.seed = 12
        SeedChoice.pickNewSeed(in: &draft, newSeed: 99)
        #expect(SeedChoice.Mode.resolve(draft) == .fixed)
        #expect(draft.seed == 99)
        #expect(RenderRequest.batch(draft, model: "fixture", copies: 3, randomBase: 7).map(\.seed) == [99, 100, 101])
        SeedChoice.select(.random, in: &draft, newSeed: 42)
        #expect(RenderRequest.batch(draft, model: "fixture", copies: 3, randomBase: 7).map(\.seed) == [7, 8, 9])
    }

    @Test func anOldLockedDraftWithoutASeedDoesNotClaimToBeFixed() {
        var draft = RenderDraft()
        draft.locksSeed = true
        #expect(SeedChoice.Mode.resolve(draft) == .random)
        #expect(RenderRequest.one(draft, model: "fixture").seed == nil)
    }

    @Test func typedSeedsAcceptTheFullUnsignedRangeOnly() {
        #expect(SeedChoice.parse("0") == 0)
        #expect(SeedChoice.parse("18446744073709551615") == UInt64.max)
        for bad in ["", " 1", "1 ", "+1", "-1", "1.5", "18446744073709551616", "1,000", "１"] {
            #expect(SeedChoice.parse(bad) == nil, "\(bad)")
        }
    }

    @Test func emptyOrMidEditTextKeepsFixedAndTheLastValidSeed() {
        var draft = RenderDraft()
        draft.seed = 77
        draft.locksSeed = true
        #expect(!SeedChoice.edit("", in: &draft))
        #expect(SeedChoice.Mode.resolve(draft) == .fixed)
        #expect(draft.seed == 77)
        #expect(!SeedChoice.edit("12a", in: &draft))
        #expect(draft.seed == 77 && draft.locksSeed)
        #expect(SeedChoice.edit("18446744073709551615", in: &draft))
        #expect(draft.seed == UInt64.max)
        #expect(SeedChoice.text(for: draft) == "18446744073709551615")
    }

    @Test func theWordsAreSharedByBothApps() {
        #expect(SeedChoice.Mode.allCases.map(\.title) == ["Random", "Fixed"])
        #expect(SeedChoice.help.contains("In a batch, each result uses the next seed number"))
    }
}
