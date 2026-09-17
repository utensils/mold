import Foundation
import MoldClient
import Testing

@testable import Mold

/// The task a rewrite records, and the fence that refuses one whose box moved
/// while it was in flight (findings 01#13, 02#12, 02#13).
@MainActor
struct ExpansionFenceTests {
    private func machine() -> MoldHost {
        MoldHost(name: "plato", baseURL: URL(string: "http://plato")!)
    }

    private func makeController(_ backend: FakeBackend, host: MoldHost) -> GenerateController {
        let hosts = HostStore(hosts: [host]) { _ in backend }
        let controller = GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts))
        controller.modelName = "ltx2-2.3-13b:q8"
        controller.modelFamily = "ltx2"
        controller.hostID = host.id
        controller.draft.prompt = "a tin robot"
        return controller
    }

    /// **Fails today**: `GenerateController+Expand` hard-codes
    /// `task: .textToImage` into every accepted offer, and sends no task at
    /// all, so a clip's print records that its prompt was written for a still.
    @Test func expandSendsAndRecordsTheRealTask() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.expandAnswer = ExpandResponse(
            original: "a tin robot", expanded: ["a", "b", "c"])
        let controller = makeController(backend, host: plato)
        controller.draft.media.sourceImage = "SRC"

        await controller.expand(on: plato, backend: backend)

        #expect(backend.expandRequests.last?.task == .imageToVideo)
        guard case let .offering(offer) = controller.expansion else {
            Issue.record("expected .offering, got \(controller.expansion)")
            return
        }
        #expect(offer.task == .imageToVideo)

        controller.accept(offer.choices[0])
        #expect(controller.draft.promptTransform?.task == .imageToVideo)
    }

    /// **Fails today**: whatever comes back is installed, even if the model,
    /// the family, the prompt or the machine moved while it was in flight.
    @Test func aRewriteWhoseBoxMovedIsRefusedByName() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.expandAnswer = ExpandResponse(
            original: "a tin robot", expanded: ["a", "b", "c"])
        let controller = makeController(backend, host: plato)

        backend.holdsExpand = true
        let rewrite = Task { await controller.expand(on: plato, backend: backend) }
        await settle { backend.calls.contains("expand") }
        // Typed while the rewrite was in the air.
        controller.draft.prompt = "a tin robot in a field"
        backend.releaseExpand()
        await rewrite.value

        guard case let .refused(sentence) = controller.expansion else {
            Issue.record("expected .refused, got \(controller.expansion)")
            return
        }
        #expect(sentence.contains("prompt changed"))
        #expect(controller.draft.prompt == "a tin robot in a field")
    }

    @Test func aRewriteThatStillBelongsToTheBoxIsInstalled() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.expandAnswer = ExpandResponse(
            original: "a tin robot", expanded: ["a", "b", "c"])
        let controller = makeController(backend, host: plato)

        await controller.expand(on: plato, backend: backend)

        guard case .offering = controller.expansion else {
            Issue.record("expected .offering, got \(controller.expansion)")
            return
        }
    }

    @Test func theSnapshotNamesEachThingThatMoved() {
        let host = UUID()
        let asked = ExpansionSnapshot(
            prompt: "a", model: "m", family: "flux", task: .textToImage, host: host)
        #expect(asked.refusalIfStale(against: asked) == nil)

        let moved = ExpansionSnapshot(
            prompt: "a", model: "other", family: "flux", task: .textToImage, host: host)
        #expect(asked.staleReasons(against: moved).first?.contains("Style changed") == true)

        let elsewhere = ExpansionSnapshot(
            prompt: "a", model: "m", family: "flux", task: .textToImage, host: UUID())
        #expect(asked.staleReasons(against: elsewhere).first?.contains("machine changed") == true)
    }
}
