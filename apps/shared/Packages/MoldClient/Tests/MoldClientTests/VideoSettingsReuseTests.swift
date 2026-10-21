import Foundation
import Testing
@testable import MoldClient

struct VideoSettingsReuseTests {
    private func metadata(authoring: String, sourceFrames: String = "0", extra: String = "") throws -> OutputMetadata {
        let text = #"{"prompt":"literal\\new\r\nline","model":"video-model","seed":18446744073709551615,"steps":3,"guidance":0,"width":768,"height":768,"frames":720,"fps":24,"enable_audio":false,"spatial_upscale":"2x","temporal_upscale":"2x","video_job":{"id":"parent","source_frames":""# + sourceFrames + #"","total_frames":"720","authoring":"# + authoring + "}" + extra + "}"
        return try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(text.utf8))
    }

    @Test func parentReuseRestoresRequestedTimeRatherThanClampedDeliveredFrames() throws {
        let source = try metadata(authoring: #"{"duration_ms":"30001","boundary_frames":107}"#)
        let reuse = try VideoSettingsReuse(metadata: source)
        #expect(reuse.duration?.draft.seconds == "30.001")
        #expect(reuse.draft.frames == 107)
        #expect(reuse.draft.prompt == "literal\\new\r\nline")
        #expect(reuse.draft.seed == UInt64.max)
        #expect(reuse.draft.locksSeed)
        #expect(reuse.draft.preferredAudio == false)
        #expect(reuse.draft.spatialUpscale == "2x")
        #expect(reuse.draft.temporalUpscale == "2x")
        #expect(reuse.duration?.draft.source == nil)
    }

    @Test func scenesRemainLiteralOrderedAndSumToTheAuthoredTime() throws {
        let source = try metadata(authoring: #"{"duration_ms":"6001","boundary_frames":107,"scenes":[{"prompt":"first\\nline\r\nreal","duration_ms":"3000"},{"prompt":"second scene","duration_ms":"3001"}]}"#)
        let reuse = try VideoSettingsReuse(metadata: source)
        let authoring = try #require(reuse.duration)
        #expect(authoring.draft.scenesEnabled)
        #expect(authoring.draft.scenes.map(\.prompt) == ["first\\nline\r\nreal", "second scene"])
        #expect(authoring.draft.scenes.map(\.seconds) == ["3", "3.001"])
        #expect(authoring.draft.seconds == "6.001")
        #expect(authoring.draft.source == nil)
    }

    @Test func extendedMovieReuseOnlyRestoresAddedFootageAndDisclosesNoPrefix() throws {
        let reuse = try VideoSettingsReuse(metadata: metadata(authoring: #"{"duration_ms":"6000","boundary_frames":107}"#, sourceFrames: "576"))
        #expect(reuse.duration?.draft.seconds == "6")
        #expect(reuse.duration?.draft.source == nil)
        #expect(reuse.notice?.contains("added footage") == true)
    }

    @Test func missingAuthoringOrUnsupportedControlsCannotSilentlyBecomeAShortClip() throws {
        let old = try metadata(authoring: "null")
        #expect(throws: (any Error).self) { try VideoSettingsReuse(metadata: old) }
        let controlled = try metadata(authoring: #"{"duration_ms":"6000","boundary_frames":107}"#, extra: #", "ic_lora_control":"original-control""#)
        #expect(throws: (any Error).self) { try VideoSettingsReuse(metadata: controlled) }
    }
    @Test func invalidPublishedBoundaryIsRefusedEvenWhenAnInputLocatorExists() throws {
        let source = try metadata(authoring: #"{"duration_ms":"6000","boundary_frames":4294967296,"input_frames":107}"#)
        #expect(throws: (any Error).self) { try VideoSettingsReuse(metadata: source) }
    }

}
