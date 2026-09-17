import Foundation
import MoldClient
import Testing

@testable import Mold

/// The Clip group's pure gates for S6a -- `VideoOnlyPolicy`, audio's MP4
/// pin, pipeline propagation, and the recipe picker -- tested the way
/// `RefineTests` tests `RefineGroup`'s: no view needed, just the pure
/// functions a body switches on.
@MainActor
struct ClipTests {
    // MARK: - VideoOnlyPolicy, ported verbatim against `videoOnly.test.ts`

    @Test func videoOnlyBlockedByEachOfItsFourConflicts() {
        let clear = VideoOnlyPolicy.Inputs()
        #expect(VideoOnlyPolicy.blockedReason(clear) == nil)
        #expect(VideoOnlyPolicy.requestValue(enabled: true, clear) == true)
        #expect(VideoOnlyPolicy.requestValue(enabled: false, clear) == nil)

        var audioEnabled = clear
        audioEnabled.audioEnabled = true
        var audioOnlyPipeline = clear
        audioOnlyPipeline.audioOnlyPipeline = true
        var hasConditioningAudio = clear
        hasConditioningAudio.hasConditioningAudio = true
        var isExtend = clear
        isExtend.isExtend = true

        for inputs in [audioEnabled, audioOnlyPipeline, hasConditioningAudio, isExtend] {
            #expect(VideoOnlyPolicy.blockedReason(inputs) != nil)
            #expect(VideoOnlyPolicy.requestValue(enabled: true, inputs) == nil)
        }

        // The pipeline explanation wins over the audio toggle's, sentence
        // for sentence against `videoOnly.test.ts`'s
        // "prefers the pipeline explanation over the audio toggle".
        var both = clear
        both.audioEnabled = true
        both.audioOnlyPipeline = true
        #expect(VideoOnlyPolicy.blockedReason(both)?.contains("Text-to-audio") == true)
    }

    // MARK: - Generate audio pins the format

    @Test func generatingAudioPinsTheFormatWhereTheRecipeAsksForMp4() throws {
        let auto = try Self.recipe("recipe-ltx2.json", "auto")
        let draft = RenderDraft().enablingAudio(true, capabilities: auto.capabilities)
        #expect(draft.enableAudio == true)
        #expect(draft.outputFormat == "mp4")
    }

    @Test func doesNotWhereItDoesNot() throws {
        let wan = try Self.recipe("recipe-wan.json")
        let draft = RenderDraft().enablingAudio(true, capabilities: wan.capabilities)
        #expect(draft.enableAudio == true)
        #expect(draft.outputFormat == nil)
    }

    // MARK: - Pipeline propagation

    @Test func theAutoRecipeSendsNoPipelineAndTheOthersSendTheirOwn() throws {
        let set = try Self.profile("recipe-ltx2.json")
        let auto = try #require(set.recipe(named: "auto"))
        let draft = RenderDraft().adopting(auto, isNewModel: true)
        #expect(draft.pipeline == nil)
        #expect(draft.request(model: "ltx-2.5-22b-dev:bf16").pipeline == nil)

        for id in ["t2a", "ic-lora"] {
            let recipe = try #require(set.recipe(named: id))
            let switched = draft.adopting(recipe, isNewModel: false)
            #expect(switched.pipeline == id)
            #expect(switched.request(model: "ltx-2.5-22b-dev:bf16").pipeline == id)
        }
    }

    // MARK: - Switching recipe re-reads every control

    @Test func switchingRecipeReReadsEveryControl() throws {
        let set = try Self.profile("recipe-ltx2.json")
        let auto = try #require(set.recipe(named: "auto"))
        let t2a = try #require(set.recipe(named: "t2a"))

        var draft = RenderDraft().adopting(auto, isNewModel: true)
        draft.steps = 50
        draft.outputFormat = "gif" // survives on `auto`, not on `t2a`

        let switched = draft.adopting(t2a, isNewModel: false)
        #expect(switched.steps == t2a.steps.clamp(50))
        // `t2a`'s own bounds, not a value smuggled in from a different
        // recipe -- its `formats` is `["wav"]` alone, so the stale "gif"
        // must not survive to be sent.
        #expect(switched.outputFormat == nil)
        #expect(switched.request(model: "ltx-2.5-22b-dev:bf16").outputFormat == nil)
    }

    // MARK: - Recipe picker

    @Test func theRecipePickerIsHiddenForAModelWithOneRecipe() throws {
        let wan = try Self.profile("recipe-wan.json")
        #expect(RecipePicker.resolve(recipes: wan.recipes, selected: wan.defaultRecipe) == .hidden)
    }

    @Test func theRecipePickerListsEveryRecipeByItsOwnLabel() throws {
        let ltx2 = try Self.profile("recipe-ltx2.json")
        guard case let .menu(options, selectedID)
            = RecipePicker.resolve(recipes: ltx2.recipes, selected: ltx2.defaultRecipe)
        else {
            Issue.record("expected .menu")
            return
        }
        #expect(options.map(\.id) == ["auto", "ic-lora", "t2a"])
        #expect(options.map(\.label) == ["Auto", "Ic Lora", "T2a"])
        #expect(selectedID == "auto")
    }

    // MARK: - Fixture loading

    /// `recipe-ltx2.json` (trimmed to `auto`, `ic-lora`, `t2a`) and
    /// `recipe-wan.json`, both captured read-only from plato -- same files
    /// `RecipeCapabilityTests` reads in MoldClientTests. This bundle cannot
    /// see MoldClient's own `RepoFixtures` (`@testable import Mold`, not
    /// MoldClient), so this loads by a path relative to this file, the same
    /// way `RefineTests.loadPlatoModels()` does.
    private static func profile(_ name: String) throws -> GenerationProfileSet {
        let fixtures = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests/MoldTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // apps/macos
            .appending(path: "Packages/MoldClient/Tests/MoldClientTests/Fixtures")
        let data = try Data(contentsOf: fixtures.appending(path: name))
        return try MoldJSON.decoder.decode(GenerationProfileSet.self, from: data)
    }

    private static func recipe(_ name: String, _ id: String = "default") throws -> GenerationRecipe {
        let set = try profile(name)
        return try #require(set.recipe(named: id))
    }
}
