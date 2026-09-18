import Foundation
import MoldClient
import Testing

@testable import Mold

/// The Refine group's pure gates -- the mask row and the ControlNet rows --
/// tested the way `GenerateInspectorTests` tests `OutputGroup.Row` and
/// `UpscaleRow`: no view needed, just the `resolve`/`maskRow` functions a
/// `RefineGroup` body switches on.
@MainActor
struct RefineTests {
    private func capabilities(_ json: String) -> RecipeCapabilities {
        try! MoldJSON.decoder.decode(RecipeCapabilities.self, from: Data(json.utf8))
    }

    /// `AdapterControl`'s memberwise init is internal to `MoldClient` -- only
    /// `@testable import` sees it, and this bundle is `@testable import Mold`,
    /// not MoldClient. Decoding is how every other app test builds a
    /// MoldClient wire type it does not own (see `GenerateInspectorTests.output(_:)`).
    private func adapterControl(_ json: String) -> AdapterControl {
        try! MoldJSON.decoder.decode(AdapterControl.self, from: Data(json.utf8))
    }

    // MARK: - Mask row

    /// Four cases: no recipe, a recipe that doesn't advertise a mask at all,
    /// a mask-capable recipe with no source picture yet, and one with a
    /// source. `readsSourceImage` gates it too -- a mask over a checkpoint
    /// that never reads a still is meaningless whatever `mask` says.
    @Test func theMaskRowNeedsBothTheCapabilityAndASource() {
        #expect(RefineGroup.maskRow(capabilities: nil, hasSource: true) == .hidden)

        let noMask = capabilities(#"{"mask": {"mode": "hidden", "required": false}}"#)
        #expect(RefineGroup.maskRow(capabilities: noMask, hasSource: true) == .hidden)

        let videoOnly = capabilities(#"""
        {"mask": {"mode": "adjustable", "required": false}, "source_image": "unsupported"}
        """#)
        #expect(RefineGroup.maskRow(capabilities: videoOnly, hasSource: true) == .hidden)

        let maskCapable = capabilities(#"{"mask": {"mode": "adjustable", "required": false}}"#)
        #expect(RefineGroup.maskRow(capabilities: maskCapable, hasSource: false) == .needsSource)
        #expect(RefineGroup.maskRow(capabilities: maskCapable, hasSource: true) == .ready)
    }

    @Test func clearingTheSourceClearsTheMaskFromTheRequest() {
        var draft = RenderDraft()
        draft.media.sourceImage = "SRC"
        draft.media.maskImage = "MASK"
        draft.media.sourceImage = nil // what `SourceImageWell`'s clear button does
        let request = RenderRequest.one(draft, model: "sd15:fp16")
        #expect(request.sourceImage == nil)
        #expect(request.maskImage == nil)
    }

    // MARK: - ControlNet rows

    @Test func aRecipeThatAdvertisesControlnetWithNothingInstalledOffersToGetOne() {
        let control = adapterControl(#"{"mode": "adjustable", "max_count": 1, "reason": null}"#)
        let generatorsOnly = [FakeFixtures.model("sd15:fp16", downloaded: true)]
        guard case let .needsAdapter(reason) = ControlNetRow.resolve(control: control, models: generatorsOnly) else {
            Issue.record("expected .needsAdapter")
            return
        }
        #expect(reason == "No ControlNet adapter is installed on this machine.")

        // The recipe's own reason, when it has one, wins over the fallback.
        let reasoned = adapterControl(#"{"mode": "adjustable", "max_count": 1, "reason": "ControlNet needs SD1.5."}"#)
        guard case let .needsAdapter(customReason) = ControlNetRow.resolve(control: reasoned, models: []) else {
            Issue.record("expected .needsAdapter")
            return
        }
        #expect(customReason == "ControlNet needs SD1.5.")

        #expect(ControlNetRow.resolve(control: nil, models: generatorsOnly) == .hidden)
    }

    @Test func anInstalledAdapterIsOfferedAsReady() {
        let control = adapterControl(#"{"mode": "adjustable", "max_count": 1, "reason": null}"#)
        let installed = [
            FakeFixtures.model("controlnet-canny-sd15:fp16", family: "controlnet", downloaded: true),
        ]
        guard case let .ready(models) = ControlNetRow.resolve(control: control, models: installed) else {
            Issue.record("expected .ready")
            return
        }
        #expect(models.map(\.name) == ["controlnet-canny-sd15:fp16"])
    }

    @Test func aControlModelWithoutAPictureSendsNeither() {
        var draft = RenderDraft()
        draft.media.control = ControlConditioning(model: "controlnet-canny-sd15:fp16")
        let request = RenderRequest.one(draft, model: "sd15:fp16")
        #expect(request.controlModel == nil)
        #expect(request.controlImage == nil)
    }

    @Test func aControlPictureWithoutAModelSendsNeither() {
        var draft = RenderDraft()
        draft.media.control = ControlConditioning(image: "CTRL")
        let request = RenderRequest.one(draft, model: "sd15:fp16")
        #expect(request.controlModel == nil)
        #expect(request.controlImage == nil)
    }

    @Test func aCompleteControlConditioningSendsBoth() {
        var draft = RenderDraft()
        draft.media.control = ControlConditioning(image: "CTRL", model: "controlnet-canny-sd15:fp16")
        let request = RenderRequest.one(draft, model: "sd15:fp16")
        #expect(request.controlImage == "CTRL")
        #expect(request.controlModel == "controlnet-canny-sd15:fp16")
    }

    /// `validation.rs:3090`-area refuses a negative scale outright; the app
    /// clamps at the floor instead of letting a drag land there.
    @Test func controlScaleBelowZeroIsClampedBeforeItIsSent() {
        var draft = RenderDraft()
        draft.media.control = ControlConditioning(image: "CTRL", model: "m", scale: -1)
        let request = RenderRequest.one(draft, model: "sd15:fp16")
        #expect(request.controlScale == 0)
    }

    /// The family filter, against a real capture from workstation (fact 3 in the
    /// M4 design): three ControlNet manifests, none installed, and one
    /// installed upscaler that must never show up in the adapter picker.
    @Test func anInstalledUpscalerIsNotOfferedAsAControlModel() throws {
        let models = try Self.loadWorkstationModels()
        let control = adapterControl(#"{"mode": "adjustable", "max_count": 1, "reason": null}"#)
        guard case .needsAdapter = ControlNetRow.resolve(control: control, models: models) else {
            Issue.record("expected .needsAdapter -- every controlnet manifest on workstation is downloaded:false")
            return
        }
    }

    /// `models-workstation.json` is trimmed from a live `GET /api/models` capture
    /// (fact 3 in the M4 design). The app test bundle cannot see MoldClient's
    /// own `RepoFixtures`/resource bundle, so this loads the same file
    /// `FakeFixtures.configListing` does -- by a path relative to this file.
    private static func loadWorkstationModels() throws -> [Model] {
        let fixtures = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests/MoldTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // apps/macos
            .appending(path: "Packages/MoldClient/Tests/MoldClientTests/Fixtures")
        let data = try Data(contentsOf: fixtures.appending(path: "models-workstation.json"))
        return try MoldJSON.decoder.decode([Model].self, from: data)
    }
}
