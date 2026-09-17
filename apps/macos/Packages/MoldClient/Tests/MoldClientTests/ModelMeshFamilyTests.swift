import Foundation
import Testing

@testable import MoldClient

private func model(family: String) -> Model {
    Model(name: "x", family: family, description: "d", sizeGb: nil, isLoaded: nil,
          downloaded: nil, hfRepo: nil, displayName: nil,
          remainingDownloadBytes: nil, generationProfile: nil, diskUsageBytes: nil,
          kind: nil, modality: nil, nsfw: nil, runtimeAvailable: nil,
          runtimeUnavailableReason: nil)
}

/// **Fails today**: `Model.swift` lists `hunyuan3d-paint` in
/// `auxiliaryFamilies` but not `hunyuan3d`, so a shape checkpoint is a plain
/// generator and IS offered in the style picker. The request it builds is
/// structurally valid but carries no `mesh` block, and nothing on this Mac can
/// draw the GLB it returns -- several GPU minutes for a blank canvas.
/// `hunyuan3d` makes something; it just does not make a PICTURE.
@Test func aMeshFamilyIsAMakerButNotAPictureMaker() {
    #expect(model(family: "hunyuan3d").isMeshMaker)
    #expect(model(family: "hunyuan3d").isGenerator)
    #expect(!model(family: "hunyuan3d").isPictureMaker)
}

/// The distinction is the whole point: `isGenerator` still answers "is this a
/// standalone model at all", which is what separates it from a ControlNet or
/// an upscaler.
@Test func everyOtherGeneratorIsStillAPictureMaker() {
    for family in ["flux", "wan", "ltx2", "sdxl"] {
        #expect(model(family: family).isPictureMaker, "\(family)")
        #expect(!model(family: family).isMeshMaker, "\(family)")
    }
    for family in ["qwen3-expand", "upscaler", "controlnet", "hunyuan3d-paint"] {
        #expect(!model(family: family).isPictureMaker, "\(family)")
        #expect(!model(family: family).isMeshMaker, "\(family)")
    }
}

/// `hunyuan3d` is NOT an auxiliary family on the Rust side -- it is a family
/// that generates, and putting it in `auxiliaryFamilies` would break the
/// contract test that reads `manifest.rs`. There is no Rust constant naming
/// the mesh families, so this set is the honest place for it.
@Test func theMeshFamilySetStaysOutOfTheRustPinnedSets() {
    #expect(Model.meshFamilies.isDisjoint(with: Model.auxiliaryFamilies))
    #expect(Model.meshFamilies.isDisjoint(with: Model.utilityFamilies))
    #expect(Model.meshFamilies.isDisjoint(with: Model.upscalerFamilies))
}
