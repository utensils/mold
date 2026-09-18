import Foundation
import MoldClient
import Testing

@testable import Mold

/// The inspector's groups, tested on their PURE gates -- no view needed.
/// `OutputGroup.Row.resolve`, `UpscaleRow.resolve` and
/// `FileUnderGroup.isShown` are exactly what a `GenerateInspector` asks
/// before drawing anything, extracted so a test can ask the same question.
@MainActor
struct GenerateInspectorTests {
    private func output(_ json: String) -> OutputCapabilities {
        try! MoldJSON.decoder.decode(OutputCapabilities.self, from: Data(json.utf8))
    }

    @Test func aHostThatCannotOrganizeHasNoFileUnderGroup() {
        #expect(FileUnderGroup.isShown(capabilities: FakeFixtures.capabilities(organize: false)) == false)
        #expect(FileUnderGroup.isShown(capabilities: nil) == false)
        #expect(FileUnderGroup.isShown(capabilities: FakeFixtures.capabilities(organize: true)) == true)
    }

    /// The mesh block from the five deduplicated `output` shapes workstation
    /// advertises (`recipe-outputs.json`): one format and a real reason.
    @Test func aRecipeWithOneFormatShowsItsReasonInsteadOfAPicker() {
        let mesh = output("""
        {"formats": ["glb"], "default_format": "glb",
         "delivery_reason": "3-D delivery uses binary glTF; OBJ, OBJ+PBR ZIP, STL and PLY are available as gallery exports."}
        """)
        #expect(OutputGroup.Row.resolve(output: mesh) == .fixed(
            name: "GLB",
            reason: "3-D delivery uses binary glTF; OBJ, OBJ+PBR ZIP, STL and PLY are available as gallery exports."
        ))
    }

    @Test func aRecipeWithNoOutputBlockShowsNoFormatRow() {
        #expect(OutputGroup.Row.resolve(output: nil) == .hidden)
    }

    @Test func aMultiFormatRecipeOffersAPickerSeededFromTheDefault() {
        let image = output(#"{"formats": ["png", "jpeg", "webp"], "default_format": "png"}"#)
        #expect(OutputGroup.Row.resolve(output: image) == .picker(
            formats: ["png", "jpeg", "webp"], defaultFormat: "png"
        ))
    }

    @Test func aMachineWithNoUpscalerInstalledOffersNoUpscaleRow() {
        let generatorsOnly = [FakeFixtures.model("flux-dev:q4", downloaded: true)]
        #expect(UpscaleRow.resolve(models: generatorsOnly).isEmpty)

        let notDownloaded = [FakeFixtures.model("real-esrgan-x4plus:fp16", family: "upscaler")]
        #expect(UpscaleRow.resolve(models: notDownloaded).isEmpty)

        let ready = [FakeFixtures.model("real-esrgan-x4plus:fp16", family: "upscaler", downloaded: true)]
        #expect(UpscaleRow.resolve(models: ready).map(\.name) == ["real-esrgan-x4plus:fp16"])
    }

    @Test func aCollectionIsAlwaysNamedNeverIdentified() {
        var draft = RenderDraft()
        draft.collectionName = "Smurf Village"
        let request = RenderRequest.one(draft, model: "m")
        #expect(request.collection == .named("Smurf Village"))
    }

    @Test func aWhitespaceOnlyTitleIsNoTitle() {
        var draft = RenderDraft()
        draft.title = "   "
        #expect(RenderRequest.one(draft, model: "m").title == nil)
    }

    /// Fact 1 in the M4 design: an absent `source_image` block means the
    /// recipe reads a still (every installed still model on workstation), not "no
    /// source path at all" -- `PromptPanel` used to read the raw optional
    /// backwards and hid the well on every one of them.
    @Test func theSourceWellAppearsOnAStillModelWhoseRecipeOmitsTheBlock() {
        let recipe = FakeFixtures.recipe()
        #expect(PromptPanel.showsSourceWell(for: recipe))
    }
}
