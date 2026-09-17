import Foundation
import MoldClient
import Testing

@testable import Mold

/// The prompt wand, in the controller: whether it is offered at all, what
/// happens to the draft when a rewrite is accepted or reverted, and the one
/// deliberate departure from the M1.5 failure funnel -- a failed rewrite is
/// never a machine-banner failure.
@MainActor
struct ExpansionTests {
    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func makeController(
        _ backend: FakeBackend, host: MoldHost,
        capabilities: Capabilities = FakeFixtures.expandCapabilities()
    ) -> GenerateController {
        let hosts = HostStore(hosts: [host]) { _ in backend }
        hosts.capabilities[host.id] = capabilities
        let controller = GenerateController(hosts: hosts)
        controller.modelName = "flux-dev:q4"
        controller.modelFamily = "flux"
        controller.hostID = host.id
        return controller
    }

    // MARK: - Advice, not a refusal

    /// `expand-ignored-hunyuan3d.json` (MoldClient fixtures) is the captured
    /// wire shape: `{"original": "a brass gear", "expanded": ["hunyuan3d
    /// reads no prompt: ..."]}, one entry despite `variations: 3`. This app
    /// test bundle cannot load MoldClient's fixture file directly, so the
    /// prefix its own test pins is reproduced here rather than the whole body.
    @Test func aFamilyThatReadsNoPromptIsAdvisedNotRefused() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.expandAnswer = ExpandResponse(
            original: "a brass gear",
            expanded: ["hunyuan3d reads no prompt: the image is the whole conditioning."])
        let controller = makeController(backend, host: plato)
        controller.draft.prompt = "a brass gear"

        await controller.expand(on: plato, backend: backend)

        guard case let .advised(text) = controller.expansion else {
            Issue.record("expected .advised, got \(controller.expansion)")
            return
        }
        #expect(text.hasPrefix("hunyuan3d reads no prompt"))
        #expect(controller.draft.prompt == "a brass gear")
    }

    // MARK: - Accepting and reverting

    @Test func acceptingARewriteKeepsTheOriginalAsTheRoot() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.expandAnswer = ExpandResponse(
            original: "a cat", expanded: [
                "a fluffy orange cat asleep in a sunbeam",
                "a cat curled by the fire", "a cat on a windowsill",
            ])
        let controller = makeController(backend, host: plato)
        controller.draft.prompt = "a cat"

        await controller.expand(on: plato, backend: backend)
        guard case let .offering(offer) = controller.expansion else {
            Issue.record("expected .offering, got \(controller.expansion)")
            return
        }
        controller.accept(offer.choices[0])

        #expect(controller.draft.prompt == "a fluffy orange cat asleep in a sunbeam")
        #expect(controller.draft.originalPrompt == "a cat")
        #expect(controller.draft.promptTransform?.operation == .expand)
        #expect(controller.draft.promptTransform?.rootPrompt == "a cat")
        #expect(controller.draft.promptTransform?.sourcePrompt == "a cat")
    }

    @Test func aSecondRewriteKeepsTheFirstOriginalAsTheRoot() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.expandAnswer = ExpandResponse(
            original: "a cat", expanded: [
                "a fluffy orange cat asleep in a sunbeam",
                "a cat curled by the fire", "a cat on a windowsill",
            ])
        let controller = makeController(backend, host: plato)
        controller.draft.prompt = "a cat"

        await controller.expand(on: plato, backend: backend)
        guard case let .offering(expandOffer) = controller.expansion else {
            Issue.record("expected .offering after expand"); return
        }
        controller.accept(expandOffer.choices[0])

        backend.remixAnswer = RemixResponse(
            sourcePrompt: "a fluffy orange cat asleep in a sunbeam", rootPrompt: "a cat",
            sourceKind: .current, task: .textToImage,
            variants: [RemixVariant(prompt: "a fluffy orange cat asleep in golden light", dimensions: [.lighting])])
        await controller.remix(on: plato, backend: backend)
        guard case let .offering(remixOffer) = controller.expansion else {
            Issue.record("expected .offering after remix"); return
        }
        controller.accept(remixOffer.choices[0])

        #expect(controller.draft.prompt == "a fluffy orange cat asleep in golden light")
        // The root is the earliest idea, unchanged by the second rewrite.
        #expect(controller.draft.originalPrompt == "a cat")
        #expect(controller.draft.promptTransform?.rootPrompt == "a cat")
        // The source is what actually went INTO this rewrite.
        #expect(controller.draft.promptTransform?.sourcePrompt == "a fluffy orange cat asleep in a sunbeam")
        #expect(controller.draft.promptTransform?.operation == .remix)
    }

    @Test func revertingPutsBackWhatWasTyped() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.expandAnswer = ExpandResponse(
            original: "a cat", expanded: [
                "a fluffy orange cat asleep in a sunbeam",
                "a cat curled by the fire", "a cat on a windowsill",
            ])
        let controller = makeController(backend, host: plato)
        controller.draft.prompt = "a cat"

        await controller.expand(on: plato, backend: backend)
        guard case let .offering(offer) = controller.expansion else {
            Issue.record("expected .offering"); return
        }
        controller.accept(offer.choices[0])
        #expect(controller.canRevertExpansion)

        controller.revertExpansion()

        #expect(controller.draft.prompt == "a cat")
        #expect(controller.draft.originalPrompt == nil)
        #expect(controller.draft.promptTransform == nil)
        #expect(!controller.canRevertExpansion)
    }

    // MARK: - What the machine has and hasn't said

    @Test func aMachineWithoutTheExpanderNamesTheModelToPull() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        let controller = makeController(
            backend, host: plato,
            capabilities: FakeFixtures.expandCapabilities(modelPresent: false, model: "qwen3-expand:q8"))
        controller.draft.prompt = "a cat"

        await controller.expand(on: plato, backend: backend)

        #expect(controller.expansion == .needsModel("qwen3-expand:q8"))
        // Decided from capabilities alone -- the route is never reached.
        #expect(backend.callCount("expand") == 0)
    }

    @Test func aMachineThatHasNotSaidWhetherItExpandsIsStillAsked() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.expandAnswer = ExpandResponse(original: "a cat", expanded: ["a cat, expanded"])
        // No `expand` key at all -- an older host, not one that said no.
        let controller = makeController(backend, host: plato, capabilities: FakeFixtures.capabilities(events: false))
        controller.draft.prompt = "a cat"

        let offer = controller.expansionOffer(for: FakeFixtures.recipe(), on: plato)
        guard case .wand = offer else {
            Issue.record("expected .wand, got \(offer)")
            return
        }

        await controller.expand(on: plato, backend: backend)
        #expect(backend.callCount("expand") == 1)
    }

    @Test func aMachineThatDoesNotRemixOffersNoRemix() {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        let controller = makeController(backend, host: plato, capabilities: FakeFixtures.expandCapabilities(remix: false))

        let offer = controller.expansionOffer(for: FakeFixtures.recipe(), on: plato)
        #expect(offer == .wand(canRemix: false))
    }

    // MARK: - A refusal is about the prompt, not the machine

    @Test func aRefusalIsShownWhereTheWandWasNotInTheMachineBanner() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.refuses = ["expand"]
        let controller = makeController(backend, host: plato)
        controller.draft.prompt = "a cat"

        await controller.expand(on: plato, backend: backend)

        guard case let .refused(message) = controller.expansion else {
            Issue.record("expected .refused, got \(controller.expansion)")
            return
        }
        #expect(!message.isEmpty)
        // Never routed through `HostStore.report` -- this is about the
        // prompt in front of you, not the machine's health.
        #expect(controller.hosts.failures.isEmpty)
    }

    // MARK: - The studio invariant

    @Test func neverArmsServerSideExpansion() async throws {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.expandAnswer = ExpandResponse(
            original: "a cat", expanded: [
                "a fluffy orange cat asleep in a sunbeam",
                "a cat curled by the fire", "a cat on a windowsill",
            ])
        let controller = makeController(backend, host: plato)
        controller.draft.prompt = "a cat"

        await controller.expand(on: plato, backend: backend)
        guard case let .offering(offer) = controller.expansion else {
            Issue.record("expected .offering"); return
        }
        controller.accept(offer.choices[0])

        let request = controller.draft.request(model: "flux-dev:q4")
        let encoded = try MoldJSON.encoder.encode(request)
        let object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(!object.keys.contains { $0.contains("expand") })
    }
}
