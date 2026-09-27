import Foundation
import Testing

@testable import MoldClient

/// The native request boundary must obey the same advertised recipe contract
/// as the browser clients. These tests use the MiniMax H3 facts captured from
/// a live host on 2026-09-18: temporal, audio-capable, source-required, and
/// `supports_strength: false` with strength fixed to 1.
struct GenerationContractParityTests {
    private func recipe(
        strength: Bool?, audio: Bool?, pipeline: String? = nil,
        audioRequiresMP4: Bool = true
    ) throws -> GenerationRecipe {
        let strengthField = strength.map { "\"supports_strength\":\($0)," } ?? ""
        let pipelineField = pipeline.map { "\"request_selector\":{\"pipeline\":\"\($0)\"}," }
            ?? ""
        let audioField = audio.map { "\"supports_audio\":\($0)," } ?? ""
        return try MoldJSON.decoder.decode(GenerationRecipe.self, from: Data(#"""
        {
          "id":"default","label":"Default",
          \#(pipelineField)
          "defaults":{"width":768,"height":768,"steps":5,"guidance":0,
                      "frames":124,"fps":24},
          "resolution":{"domain":"dynamic","alignment":16,"min_width":256,
                        "min_height":256},
          "steps":{"default":5,"min":5,"max":5,"step":1,"mode":"fixed"},
          "guidance":{"default":0,"min":0,"max":0,"step":0.1,"mode":"fixed"},
          "temporal":{
            "frames":{"default":124,"min":107,"max":345,"step":17,
                      "recommended":[124],"mode":"adjustable"},
            "frame_offset":5,"fps":{"mode":"fixed","value":24}
          },
          "capabilities":{
            \#(strengthField)
            \#(audioField)"source_image":"required",
            "output":{"default_format":"mp4","formats":["mp4","gif","apng","webp"],
                      "audio_requires_mp4":\#(audioRequiresMP4)}
          }
        }
        """#.utf8))
    }

    private func sourceDraft(_ recipe: GenerationRecipe) -> RenderDraft {
        var draft = RenderDraft().adopting(recipe, isNewModel: true)
        draft.media.sourceImage = "SOURCE"
        draft.media.lastExclusiveWrite = .source
        return draft
    }

    @Test func advertisedNoStrengthPinsEveryRequestPathToOne() throws {
        var draft = sourceDraft(try recipe(strength: false, audio: true))
        draft.strength = 0.42

        #expect(RenderRequest.one(draft, model: "minimax-h3-fl2va:test").strength == 1)
        #expect(RenderRequest.batch(
            draft, model: "minimax-h3-fl2va:test", copies: 3, randomBase: 9
        ).allSatisfy { $0.strength == 1 })
        #expect(RenderRequest.placement(
            draft, model: "minimax-h3-fl2va:test").strength == 1)
        // Capability reconciliation never destroys the value waiting for a
        // model that exposes the slider again.
        #expect(draft.strength == 0.42)
    }

    @Test func ref2VAWithoutASourceStillCarriesFixedStrengthOne() throws {
        var draft = RenderDraft().adopting(
            try recipe(strength: false, audio: true), isNewModel: true)
        draft.media.editImages = ["REFERENCE"]
        draft.media.lastExclusiveWrite = .references
        let request = RenderRequest.one(draft, model: "minimax-h3-ref2va:test")
        #expect(request.sourceImage == nil)
        #expect(request.strength == 1)
    }

    @Test func oldAndCapableHostsKeepTheAuthoredStrength() throws {
        for advertised in [Bool?.none, Bool?.some(true)] {
            var draft = sourceDraft(try recipe(strength: advertised, audio: false))
            draft.strength = 0.42
            #expect(RenderRequest.one(draft, model: "m").strength == 0.42)
        }
    }

    @Test func capableStrengthSurvivesSourceAddedByRetainedHydration() throws {
        var draft = RenderDraft().adopting(
            try recipe(strength: true, audio: false), isNewModel: true)
        draft.strength = 0.42
        #expect(draft.media.sourceImage == nil)
        #expect(RenderRequest.one(draft, model: "m").strength == 0.42)
        #expect(RenderRequest.batch(
            draft, model: "m", copies: 2, randomBase: 7
        ).allSatisfy { $0.strength == 0.42 })
    }

    @Test func freshCapableVideoDefaultsToAudioAndMp4() throws {
        var draft = RenderDraft()
        draft = draft.adopting(
            try recipe(strength: false, audio: true), isNewModel: true, family: "ltx2")

        #expect(draft.enableAudio)
        #expect(draft.preferredAudio == nil)
        #expect(draft.outputFormat == "mp4")
        #expect(RenderRequest.one(draft, model: "m").enableAudio == true)
    }

    @Test func choosingANonMP4ContainerTurnsOptionalAudioOff() throws {
        let capable = try recipe(strength: true, audio: true)
        for format in ["gif", "apng", "webp"] {
            let draft = RenderDraft().adopting(
                capable, isNewModel: true, family: "ltx2")
                .selectingOutputFormat(format, output: capable.capabilities.output)

            #expect(draft.outputFormat == format)
            #expect(draft.preferredAudio == false)
            #expect(!draft.enableAudio)
            let request = RenderRequest.one(draft, model: "m")
            #expect(request.outputFormat == format)
            #expect(request.enableAudio == false)
        }
    }

    @Test func reusedGIFWithoutAudioMetadataStaysGIFAndSilent() throws {
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data("""
        {"prompt":"p","model":"m","steps":5,"guidance":0,
         "width":768,"height":768,"output_format":"gif"}
        """.utf8))
        let capable = try recipe(strength: true, audio: true)
        let draft = RenderDraft(reusing: metadata)
            .adopting(capable, isNewModel: false, family: "ltx2")

        #expect(draft.outputFormat == "gif")
        #expect(draft.preferredAudio == false)
        #expect(!draft.enableAudio)
        #expect(RenderRequest.one(draft, model: "m").enableAudio == false)
    }

    @Test func explicitOffSurvivesAnUnsupportedRecipeAndReachesTheWire() throws {
        let capable = try recipe(strength: true, audio: true)
        let unsupported = try recipe(strength: true, audio: false)
        var draft = RenderDraft().adopting(capable, isNewModel: true, family: "ltx2")
            .enablingAudio(false, capabilities: capable.capabilities)
        #expect(draft.preferredAudio == false)
        #expect(RenderRequest.one(draft, model: "m").enableAudio == false)

        draft = draft.adopting(unsupported, isNewModel: true)
        #expect(!draft.enableAudio)
        #expect(draft.preferredAudio == false)
        #expect(RenderRequest.one(draft, model: "m").enableAudio == nil)

        draft = draft.adopting(capable, isNewModel: true, family: "ltx2")
        #expect(!draft.enableAudio)
        #expect(RenderRequest.one(draft, model: "m").enableAudio == false)
    }

    @Test func audioOnlyIsFixedOnAndOmitsInvalidAudioChoices() throws {
        let audioOnly = try recipe(strength: true, audio: true, pipeline: "t2a",
                                   audioRequiresMP4: false)
        let draft = RenderDraft().adopting(audioOnly, isNewModel: true)
            .enablingAudio(false, capabilities: audioOnly.capabilities)
        #expect(draft.enableAudio)
        #expect(draft.preferredAudio == false)
        #expect(draft.requiresAudio)
        #expect(RenderRequest.one(draft, model: "m").enableAudio == nil)
    }

    @Test func h3IsFixedOnAndNeverSendsFalseOrVideoOnly() throws {
        let h3 = try recipe(strength: false, audio: true)
        var draft = RenderDraft().adopting(
            h3, isNewModel: true, family: "minimax-h3")
            .enablingAudio(false, capabilities: h3.capabilities)
        draft.videoOnly = true // stale/manual state must not escape.

        #expect(draft.preferredAudio == false)
        #expect(draft.enableAudio)
        #expect(draft.requiresAudio)
        #expect(!draft.offersAudioControl)
        let request = RenderRequest.one(draft, model: "minimax-h3-fl2va:test")
        #expect(request.enableAudio == nil)
        #expect(request.videoOnly == nil)
    }

    @Test func checkpointAudioVetoWinsOverRecipeSupport() throws {
        let capable = try recipe(strength: true, audio: true)
        var draft = RenderDraft().adopting(
            capable, isNewModel: true, family: "ltx2", modelSupportsAudio: false)
        draft.videoOnly = true

        #expect(!draft.supportsAudio)
        #expect(draft.audioUnavailableForModel)
        #expect(!draft.enableAudio)
        #expect(RenderRequest.one(draft, model: "m").enableAudio == false)
        #expect(RenderRequest.one(draft, model: "m").videoOnly == nil)
    }

    @Test func ltxRecipeVetoSendsFalseAndOlderRecipeDefaultsOn() throws {
        let unavailable = RenderDraft().adopting(
            try recipe(strength: true, audio: false), isNewModel: true, family: "ltx2")
        #expect(unavailable.audioUnavailableForModel)
        #expect(RenderRequest.one(unavailable, model: "m").enableAudio == false)

        let older = RenderDraft().adopting(
            try recipe(strength: true, audio: nil), isNewModel: true, family: "ltx2")
        #expect(older.enableAudio)
        #expect(RenderRequest.one(older, model: "m").enableAudio == true)
    }

    @Test func unavailableCheckpointParksPreferenceUntilCapabilityReturns() throws {
        let capable = try recipe(strength: true, audio: true)
        var unavailable = RenderDraft()
        unavailable.preferredAudio = true
        unavailable = unavailable.adopting(
            capable, isNewModel: true, family: "ltx2", modelSupportsAudio: false)
            .selectingOutputFormat("gif", output: capable.capabilities.output)

        #expect(unavailable.preferredAudio == true)
        #expect(!unavailable.enableAudio)
        #expect(RenderRequest.one(unavailable, model: "m").enableAudio == false)

        let restored = unavailable.adopting(
            capable, isNewModel: false, family: "ltx2", modelSupportsAudio: true)
        #expect(restored.outputFormat == "gif")
        #expect(restored.preferredAudio == false)
        #expect(RenderRequest.one(restored, model: "m").enableAudio == false)

        let mp4 = unavailable.selectingOutputFormat(
            "mp4", output: capable.capabilities.output)
            .adopting(capable, isNewModel: false, family: "ltx2", modelSupportsAudio: true)
        #expect(mp4.preferredAudio == true)
        #expect(mp4.enableAudio)
        #expect(RenderRequest.one(mp4, model: "m").enableAudio == true)
    }

    @Test func modelRowDecodesCheckpointAudioVetoAndOlderAbsence() throws {
        func model(_ field: String) throws -> Model {
            let json = "{\"name\":\"m\",\"family\":\"ltx2\",\"description\":\"M\""
                + field + "}"
            return try MoldJSON.decoder.decode(Model.self, from: Data(json.utf8))
        }
        #expect(try model(",\"supports_audio\":false").supportsAudio == false)
        #expect(try model("").supportsAudio == nil)
    }

    @Test func persistedPreferenceStaysParkedWhileAudioIsUnsupported() throws {
        let capable = try recipe(strength: true, audio: true)
        let unsupported = try recipe(strength: true, audio: false)
        let chosen = RenderDraft().adopting(capable, isNewModel: true)
            .enablingAudio(true, capabilities: capable.capabilities)
            .adopting(unsupported, isNewModel: true)
        let descriptor = DraftDescriptor(chosen, model: "m", family: "f", recipeID: "default")
        var restored = RenderDraft()
        descriptor.apply(to: &restored)
        restored = restored.adopting(capable, isNewModel: true)
        #expect(restored.preferredAudio == true)
        #expect(restored.enableAudio)
    }

    @Test func descriptorCodingPreservesUntouchedAndExplicitAudioChoices() throws {
        let capable = try recipe(strength: true, audio: true)
        for preference in [Bool?.none, Bool?.some(false), Bool?.some(true)] {
            var original = RenderDraft().adopting(capable, isNewModel: true)
            original.preferredAudio = preference
            let encoded = try MoldJSON.localEncoder.encode(DraftDescriptor(
                original, model: "m", family: "f", recipeID: "default"))
            let descriptor = try MoldJSON.localDecoder.decode(DraftDescriptor.self, from: encoded)
            var restored = RenderDraft()
            descriptor.apply(to: &restored)
            #expect(restored.preferredAudio == preference)
        }
    }

    @Test func reusedAbsentAudioDefaultsOnButRecordedOffStaysOff() throws {
        func metadata(_ audio: String) throws -> OutputMetadata {
            try MoldJSON.decoder.decode(OutputMetadata.self, from: Data("""
            {"prompt":"p","model":"m","steps":5,"guidance":0,
             "width":768,"height":768\(audio)}
            """.utf8))
        }
        let capable = try recipe(strength: true, audio: true)
        let untouched = RenderDraft(reusing: try metadata(""))
            .adopting(capable, isNewModel: false)
        let off = RenderDraft(reusing: try metadata(",\"enable_audio\":false"))
            .adopting(capable, isNewModel: false)
        #expect(untouched.enableAudio)
        #expect(untouched.preferredAudio == nil)
        #expect(!off.enableAudio)
        #expect(off.preferredAudio == false)
    }

    @Test func reusedLegacyVideoOnlyImpliesAudioWasExplicitlyOff() throws {
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data("""
        {"prompt":"p","model":"m","steps":5,"guidance":0,
         "width":768,"height":768,"video_only":true}
        """.utf8))
        let capable = try recipe(strength: true, audio: true)
        let draft = RenderDraft(reusing: metadata)
            .adopting(capable, isNewModel: false, family: "ltx2")
        #expect(draft.preferredAudio == false)
        #expect(!draft.enableAudio)
        #expect(draft.videoOnly)
        #expect(RenderRequest.one(draft, model: "m").videoOnly == true)
    }
}
