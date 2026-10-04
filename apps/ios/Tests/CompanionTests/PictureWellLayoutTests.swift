import Foundation
import MoldClient
import Testing

@testable import MoldCompanion

@MainActor
struct PictureWellLayoutTests {
    @Test func ordinarySourceRecipesDoNotRequireABoundaryProtocol() throws {
        let caps = try MoldJSON.decoder.decode(RecipeCapabilities.self, from: Data("{}".utf8))
        #expect(PictureWells.showsSourceWell(capabilities: caps, mode: .single))
        #expect(PictureWells.showsSourceWell(capabilities: caps, mode: .singleAndReferences))
        #expect(PictureWells.showsSourceWell(capabilities: caps, mode: .singleOrReferences))
        #expect(!PictureWells.showsSourceWell(capabilities: nil, mode: .single))
        #expect(!PictureWells.showsSourceWell(capabilities: caps, mode: .references))
    }

    @Test func boundaryAndUnsupportedRecipesKeepDedicatedLayouts() throws {
        let boundary = try MoldJSON.decoder.decode(RecipeCapabilities.self, from: Data(#"{"boundary_frames":{"mode":"adjustable","first_required":true,"last_required":true,"min_frames":2,"wire":"wan-pair"}}"#.utf8))
        #expect(!PictureWells.showsSourceWell(capabilities: boundary, mode: .single))
        let unsupported = try MoldJSON.decoder.decode(RecipeCapabilities.self, from: Data(#"{"source_image":"unsupported"}"#.utf8))
        #expect(!PictureWells.showsSourceWell(capabilities: unsupported, mode: .single))
    }

    @Test func namedAndTypedReferencesKeepDedicatedWells() throws {
        let named = try MoldJSON.decoder.decode(RecipeCapabilities.self, from: Data(#"{"mesh":{"named_views":{"mode":"adjustable","roles":["front"],"min_count":1,"max_count":1}}}"#.utf8))
        #expect(!PictureWells.showsSourceWell(capabilities: named, mode: .single))
        let typed = try MoldJSON.decoder.decode(RecipeCapabilities.self, from: Data(#"{"generation_references":{"mode":"adjustable","required":true,"kinds":["image"],"max_count":12,"max_images":9,"max_videos":3,"max_audios":3,"min_duration_ms":2000,"max_duration_ms":15000,"max_video_duration_ms":15000,"max_audio_duration_ms":15000,"max_inline_bytes":33554432,"requires_visual":true}}"#.utf8))
        #expect(!PictureWells.showsSourceWell(capabilities: typed, mode: .single))
    }

}
