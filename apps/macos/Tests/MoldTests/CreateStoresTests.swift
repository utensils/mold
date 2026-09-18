import Foundation
import MoldClient
import Testing

@testable import Mold

/// Recent, and per-model defaults: two per-host stores that both report
/// through the one failure funnel, and the adoption step that puts a
/// machine's stored numbers on top of a recipe's own -- but only on a NEW
/// model, and clamped exactly the way `adopting` clamps a carried-over draft.
@MainActor
struct CreateStoresTests {
    private func machine(_ name: String = "workstation") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    /// `FakeFixtures.model` has no `generation_profile`, so `defaultRecipe`
    /// would be nil and adoption would never reach `applying` at all -- this
    /// wraps a real recipe into a model the way the wire actually carries
    /// one, by re-encoding the recipe into the `generation_profile` block.
    private func model(_ name: String, recipe: GenerationRecipe) -> Model {
        let recipeJSON = String(data: try! MoldJSON.encoder.encode(recipe), encoding: .utf8)!
        let json = """
        {"name": "\(name)", "family": "flux", "description": "\(name) — fake", "size_gb": null,
         "generation_profile": {"schema_version": 1, "profile_id": "p", "profile_hash": "h",
           "default_recipe_id": "\(recipe.id)", "recipes": [\(recipeJSON)]}}
        """
        return try! MoldJSON.decoder.decode(Model.self, from: Data(json.utf8))
    }

    // MARK: - PromptHistoryStore

    @Test func aMachineWithNoMetadataDbSaysSoRatherThanReportingAFailure() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.plantedErrors["history"] =
            MoldClientError.http(status: 503, code: "HISTORY_UNAVAILABLE", message: "no metadata db")
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = PromptHistoryStore(hosts: hosts)

        await store.refresh(on: workstation.id)

        #expect(store.unavailable.contains(workstation.id))
        #expect(hosts.failures.isEmpty)
    }

    /// The `QueueStore` lesson, applied before it can be made again: a
    /// machine that cannot answer keeps the rows it last showed.
    @Test func aMachineThatRefusesItsHistoryIsReportedAndKeepsWhatItShowed() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.historyRows = [HistoryEntry(prompt: "a cat", model: "flux-dev:q8", usedAt: 1)]
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = PromptHistoryStore(hosts: hosts)
        await store.refresh(on: workstation.id)
        #expect(store.entries(on: workstation.id).count == 1)

        backend.refuses = ["history"]
        await store.refresh(on: workstation.id)

        #expect(store.entries(on: workstation.id).count == 1)
        #expect(hosts.failures.contains { $0.host == workstation.id && $0.verb == "list what it was last asked for" })
    }

    @Test func anEmptyHistoryIsNotTheSameAsNoHistory() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.historyRows = []
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = PromptHistoryStore(hosts: hosts)
        #expect(store.hasLoaded(on: workstation.id) == false)

        await store.refresh(on: workstation.id)

        #expect(store.hasLoaded(on: workstation.id) == true)
        #expect(store.unavailable.isEmpty)
        #expect(store.entries(on: workstation.id).isEmpty)
    }

    // MARK: - Adoption

    /// **Fails today**, before `ModelDefaultsStore` (now `ConfigStore`) and
    /// `RenderDraft.applying` exist: with `config-workstation.json` planted -- every
    /// `models.flux-dev:q8.*`
    /// row present and `null` -- adopting `flux-dev:q8` must leave the
    /// recipe's own numbers standing, not something read out of an all-null
    /// listing.
    @Test func aModelNobodyConfiguredAdoptsTheRecipesOwnNumbers() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.configListing = FakeFixtures.configListing()
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let defaultsStore = ConfigStore(hosts: hosts)
        await defaultsStore.refresh(on: workstation.id)

        let recipe = FakeFixtures.recipe()
        let controller = GenerateController(hosts: hosts, defaults: defaultsStore)
        controller.select(model: model("flux-dev:q8", recipe: recipe), on: workstation.id)

        #expect(controller.draft.steps == recipe.defaults.steps)
        #expect(controller.draft.guidance == recipe.defaults.guidance)
    }

    @Test func aStoredDefaultBeatsTheRecipeOnANewModel() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.configListing = ConfigListing(entries: [
            ConfigEntry(key: "models.flux-dev:q8.default_steps", value: .number(12), source: "db"),
        ])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let defaultsStore = ConfigStore(hosts: hosts)
        await defaultsStore.refresh(on: workstation.id)

        let recipe = FakeFixtures.recipe()
        let controller = GenerateController(hosts: hosts, defaults: defaultsStore)
        controller.select(model: model("flux-dev:q8", recipe: recipe), on: workstation.id)

        #expect(controller.draft.steps == 12)
    }

    @Test func aStoredDefaultOutsideTheRecipesRangeIsClamped() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.configListing = ConfigListing(entries: [
            ConfigEntry(key: "models.flux-dev:q8.default_steps", value: .number(200), source: "db"),
        ])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let defaultsStore = ConfigStore(hosts: hosts)
        await defaultsStore.refresh(on: workstation.id)

        let recipe = FakeFixtures.recipe(stepsMax: 8)
        let controller = GenerateController(hosts: hosts, defaults: defaultsStore)
        controller.select(model: model("flux-dev:q8", recipe: recipe), on: workstation.id)

        #expect(controller.draft.steps == 8)
    }

    @Test func aReusedDraftIsNotOverwrittenByAMachinesDefaults() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.configListing = ConfigListing(entries: [
            ConfigEntry(key: "models.flux-dev:q8.default_steps", value: .number(12), source: "db"),
        ])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let defaultsStore = ConfigStore(hosts: hosts)
        await defaultsStore.refresh(on: workstation.id)

        let recipe = FakeFixtures.recipe()
        let controller = GenerateController(hosts: hosts, defaults: defaultsStore)
        // What reuse restored from a print's own provenance.
        controller.draft.steps = 55
        controller.adopt(model: model("flux-dev:q8", recipe: recipe), on: workstation.id, keepingDraft: true)

        #expect(controller.draft.steps == 55)
    }

    // MARK: - Saving and clearing

    @Test func savingWritesOnlyTheFieldsThisAppHasAControlFor() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.configListing = ConfigListing(entries: [])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let defaultsStore = ConfigStore(hosts: hosts)
        var draft = RenderDraft()
        draft.steps = 30
        draft.guidance = 4.5
        draft.width = 768
        draft.height = 512
        draft.negativePrompt = "blurry"

        await defaultsStore.save(draft, for: "flux-dev:q8", on: workstation.id)

        #expect(backend.configWrites.count == 5)
        #expect(!backend.configWrites.contains { $0.0.hasSuffix(".scheduler") })
        #expect(!backend.configWrites.contains { $0.0.hasSuffix(".lora") })
        #expect(!backend.configWrites.contains { $0.0.hasSuffix(".lora_scale") })
    }

    @Test func savingRereadsSoWhatIsShownIsWhatLanded() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.configListing = ConfigListing(entries: [])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let defaultsStore = ConfigStore(hosts: hosts)

        await defaultsStore.refresh(on: workstation.id)
        await defaultsStore.save(RenderDraft(), for: "flux-dev:q8", on: workstation.id)

        #expect(backend.callCount("config") == 2)
    }

    @Test func clearingDropsAllEightRows() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.configListing = ConfigListing(entries: [])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let defaultsStore = ConfigStore(hosts: hosts)

        await defaultsStore.clear(for: "flux-dev:q8", on: workstation.id)

        #expect(backend.configResets.count == 8)
    }
}
