import Foundation
import Testing

@testable import MoldClient

// `Fixtures/recipe-outputs.json` is the five deduplicated `output` blocks
// sampled across every recipe workstation advertises: `delivery_reason` is real and
// is the only thing that explains a one-entry `formats` list.

private func outputs() throws -> [String: OutputCapabilities] {
    try MoldJSON.decoder.decode(
        [String: OutputCapabilities].self, from: RepoFixtures.fixture("recipe-outputs.json"))
}

@Test func aRecipeWithOneFormatSaysWhy() throws {
    let blocks = try outputs()

    #expect(blocks["image"]?.deliveryReason == nil)
    #expect(blocks["image"]?.isFixed == false)

    #expect(blocks["video"]?.deliveryReason == nil)
    #expect(blocks["video"]?.isFixed == false)

    #expect(blocks["ltx2"]?.deliveryReason == "Audio-enabled video delivery requires MP4.")
    #expect(blocks["ltx2"]?.audioRequiresMp4 == true)

    #expect(blocks["audio"]?.deliveryReason == "Audio-only delivery uses WAV.")
    #expect(blocks["audio"]?.isFixed == true)

    let mesh = try #require(blocks["mesh"])
    #expect(mesh.isFixed == true)
    #expect(mesh.deliveryReason?.hasPrefix("3-D delivery uses binary glTF") == true)
}
