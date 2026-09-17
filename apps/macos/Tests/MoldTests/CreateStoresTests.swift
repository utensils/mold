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
    private func machine(_ name: String = "plato") -> MoldHost {
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
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.plantedErrors["history"] =
            MoldClientError.http(status: 503, code: "HISTORY_UNAVAILABLE", message: "no metadata db")
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let store = PromptHistoryStore(hosts: hosts)

        await store.refresh(on: plato.id)

        #expect(store.unavailable.contains(plato.id))
        #expect(hosts.failures.isEmpty)
    }

    /// The `QueueStore` lesson, applied before it can be made again: a
    /// machine that cannot answer keeps the rows it last showed.
    @Test func aMachineThatRefusesItsHistoryIsReportedAndKeepsWhatItShowed() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.historyRows = [HistoryEntry(prompt: "a cat", model: "flux-dev:q8", usedAt: 1)]
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let store = PromptHistoryStore(hosts: hosts)
        await store.refresh(on: plato.id)
        #expect(store.entries(on: plato.id).count == 1)

        backend.refuses = ["history"]
        await store.refresh(on: plato.id)

        #expect(store.entries(on: plato.id).count == 1)
        #expect(hosts.failures.contains { $0.host == plato.id && $0.verb == "list what it was last asked for" })
    }

    @Test func anEmptyHistoryIsNotTheSameAsNoHistory() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.historyRows = []
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let store = PromptHistoryStore(hosts: hosts)
        #expect(store.hasLoaded(on: plato.id) == false)

        await store.refresh(on: plato.id)

        #expect(store.hasLoaded(on: plato.id) == true)
        #expect(store.unavailable.isEmpty)
        #expect(store.entries(on: plato.id).isEmpty)
    }

    // MARK: - Adoption

    /// **Fails today**, before `ModelDefaultsStore` and `RenderDraft.applying`
    /// exist: with `config-plato.json` planted -- every `models.flux-dev:q8.*`
    /// row present and `null` -- adopting `flux-dev:q8` must leave the
    /// recipe's own numbers standing, not something read out of an all-null
    /// listing.
    @Test func aModelNobodyConfiguredAdoptsTheRecipesOwnNumbers() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.configListing = FakeFixtures.configListing()
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let defaultsStore = ModelDefaultsStore(hosts: hosts)
        await defaultsStore.refresh(on: plato.id)

        let recipe = FakeFixtures.recipe()
        let controller = GenerateController(hosts: hosts, defaults: defaultsStore)
        controller.select(model: model("flux-dev:q8", recipe: recipe), on: plato.id)

        #expect(controller.draft.steps == recipe.defaults.steps)
        #expect(controller.draft.guidance == recipe.defaults.guidance)
    }

    @Test func aStoredDefaultBeatsTheRecipeOnANewModel() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.configListing = ConfigListing(entries: [
            ConfigEntry(key: "models.flux-dev:q8.default_steps", value: .number(12), source: "db"),
        ])
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let defaultsStore = ModelDefaultsStore(hosts: hosts)
        await defaultsStore.refresh(on: plato.id)

        let recipe = FakeFixtures.recipe()
        let controller = GenerateController(hosts: hosts, defaults: defaultsStore)
        controller.select(model: model("flux-dev:q8", recipe: recipe), on: plato.id)

        #expect(controller.draft.steps == 12)
    }

    @Test func aStoredDefaultOutsideTheRecipesRangeIsClamped() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.configListing = ConfigListing(entries: [
            ConfigEntry(key: "models.flux-dev:q8.default_steps", value: .number(200), source: "db"),
        ])
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let defaultsStore = ModelDefaultsStore(hosts: hosts)
        await defaultsStore.refresh(on: plato.id)

        let recipe = FakeFixtures.recipe(stepsMax: 8)
        let controller = GenerateController(hosts: hosts, defaults: defaultsStore)
        controller.select(model: model("flux-dev:q8", recipe: recipe), on: plato.id)

        #expect(controller.draft.steps == 8)
    }

    @Test func aReusedDraftIsNotOverwrittenByAMachinesDefaults() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.configListing = ConfigListing(entries: [
            ConfigEntry(key: "models.flux-dev:q8.default_steps", value: .number(12), source: "db"),
        ])
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let defaultsStore = ModelDefaultsStore(hosts: hosts)
        await defaultsStore.refresh(on: plato.id)

        let recipe = FakeFixtures.recipe()
        let controller = GenerateController(hosts: hosts, defaults: defaultsStore)
        // What reuse restored from a print's own provenance.
        controller.draft.steps = 55
        controller.adopt(model: model("flux-dev:q8", recipe: recipe), on: plato.id, keepingDraft: true)

        #expect(controller.draft.steps == 55)
    }

    // MARK: - Saving and clearing

    @Test func savingWritesOnlyTheFieldsThisAppHasAControlFor() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.configListing = ConfigListing(entries: [])
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let defaultsStore = ModelDefaultsStore(hosts: hosts)
        var draft = RenderDraft()
        draft.steps = 30
        draft.guidance = 4.5
        draft.width = 768
        draft.height = 512
        draft.negativePrompt = "blurry"

        await defaultsStore.save(draft, for: "flux-dev:q8", on: plato.id)

        #expect(backend.configWrites.count == 5)
        #expect(!backend.configWrites.contains { $0.0.hasSuffix(".scheduler") })
        #expect(!backend.configWrites.contains { $0.0.hasSuffix(".lora") })
        #expect(!backend.configWrites.contains { $0.0.hasSuffix(".lora_scale") })
    }

    @Test func savingRereadsSoWhatIsShownIsWhatLanded() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.configListing = ConfigListing(entries: [])
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let defaultsStore = ModelDefaultsStore(hosts: hosts)

        await defaultsStore.refresh(on: plato.id)
        await defaultsStore.save(RenderDraft(), for: "flux-dev:q8", on: plato.id)

        #expect(backend.callCount("config") == 2)
    }

    @Test func clearingDropsAllEightRows() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.configListing = ConfigListing(entries: [])
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let defaultsStore = ModelDefaultsStore(hosts: hosts)

        await defaultsStore.clear(for: "flux-dev:q8", on: plato.id)

        #expect(backend.configResets.count == 8)
    }
}
