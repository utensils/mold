import Foundation
import MoldClient
import Testing
@testable import Mold

@MainActor
struct SourceMediaReuseTests {
    @Test func compatibleRecipesAreAvailableWithoutAnActiveGenerateModel() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let profile = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("../shared/Packages/MoldClient/Tests/MoldClientTests/Fixtures/recipe-ltx2.json")))
        var payload: [String: Any] = ["name": "ltx2", "family": "ltx2", "description": "Test model", "downloaded": true, "generation_profile": profile]
        let model = try MoldJSON.decoder.decode(Model.self, from: JSONSerialization.data(withJSONObject: payload))
        let image = RetainedSourceMedia.Member(memberId: "original", role: "source_image", displayName: "source", sizeBytes: 1)
        let offers = SourceMediaReuseOffer.offers(model: model, member: image, draft: RenderDraft())
        #expect(!offers.isEmpty)
        #expect(offers.allSatisfy { !$0.choices.isEmpty })
        payload["runtime_ready"] = false
        let unavailable = try MoldJSON.decoder.decode(Model.self, from: JSONSerialization.data(withJSONObject: payload))
        #expect(SourceMediaReuseOffer.offers(model: unavailable, member: image, draft: RenderDraft()).isEmpty)
    }

    @Test func unknownSourceAndUninstalledModelOfferNoRoles() {
        let model = FakeFixtures.model("flux", downloaded: false)
        let image = RetainedSourceMedia.Member(memberId: "original", role: "source_image", displayName: "source", sizeBytes: 1)
        #expect(SourceMediaReuseOffer.offers(model: model, member: image, draft: RenderDraft()).isEmpty)
    }
    @Test func menusKeepModelChooserAvailableAndDisableMutationsDuringImport() {
        let menu = SourceMediaReuseAction.menu(choices: [], canOpen: true, knownKind: true, busy: false)
        #expect(menu.compactMap(\.kind) == [.open, .chooseModel])
        let busy = SourceMediaReuseAction.menu(choices: [.firstFrame, .lastFrame], canOpen: true, knownKind: true, busy: true)
        #expect(busy.filter { $0.kind != nil && $0.kind != .open }.allSatisfy { $0.isDisabled })
        #expect(SourceMediaReuseAction.menu(choices: [], canOpen: false, knownKind: false, busy: false).isEmpty)
    }

    @Test func choosingNonDefaultRecipeDoesNotClearPromptThroughDefault() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        var profile = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("../shared/Packages/MoldClient/Tests/MoldClientTests/Fixtures/recipe-ltx2.json"))) as? [String: Any])
        var recipes = try #require(profile["recipes"] as? [[String: Any]])
        var caps = try #require(recipes[0]["capabilities"] as? [String: Any])
        caps["prompt"] = ["mode": "ignored"]
        recipes[0]["capabilities"] = caps
        profile["recipes"] = recipes
        let model = try MoldJSON.decoder.decode(Model.self, from: JSONSerialization.data(withJSONObject:
            ["name": "ltx2", "family": "ltx2", "description": "Test model", "downloaded": true, "generation_profile": profile]))
        let selected = try #require(model.generationProfile?.recipe(named: "ic-lora"))
        let host = MoldHost(name: "Test", baseURL: URL(string: "http://localhost:19999")!)
        let fake = FakeBackend(host: host)
        let hosts = HostStore(hosts: [host]) { _ in fake }
        let controller = GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts))
        controller.draft.prompt = "Keep my authored prompt"
        controller.adopt(model: model, on: host.id, keepingDraft: true, selectedRecipe: selected)
        #expect(controller.draft.prompt == "Keep my authored prompt")
        #expect(controller.recipeID == selected.id)
        #expect(controller.machineChoice == host.id)
        #expect(controller.draft.media == RenderDraft().adopting(selected, isNewModel: false, for: model).media)
    }

}
