import Foundation
import MoldClient
import Testing

@testable import Mold

/// `LoraStore`'s per-(machine, model) asking, and the two pure gates that
/// decide what the Adapters and Identity groups draw -- no view needed for
/// either. See `GenerateInspectorTests` for the same idiom applied to
/// `OutputGroup` and `FileUnderGroup`.
@MainActor
struct AdaptersTests {
    private func machine(_ name: String = "workstation") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func lora(_ id: String, name: String? = nil, path: String? = nil,
                       trainedWords: [String] = []) -> LoraInfo {
        let words = trainedWords.map { "\"\($0)\"" }.joined(separator: ",")
        let json = """
        {"id": "\(id)", "name": "\(name ?? id)", "family": "z-image",
         "author": null, "path": "\(path ?? "/models/\(id).safetensors")",
         "trained_words": [\(words)], "size_bytes": null, "thumbnail_url": null,
         "added_at": 1}
        """
        return try! MoldJSON.decoder.decode(LoraInfo.self, from: Data(json.utf8))
    }

    // MARK: - LoraStore

    @Test func anAdapterListIsAskedPerModelNotPerMachine() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.loraRows = ["flux-dev:q8": [lora("a")], "z-image-turbo:q8": [lora("b")]]
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = LoraStore(hosts: hosts)

        await store.refresh(model: "flux-dev:q8", on: workstation.id)
        await store.refresh(model: "z-image-turbo:q8", on: workstation.id)

        #expect(backend.loraModelsRequested == ["flux-dev:q8", "z-image-turbo:q8"])
        #expect(store.rows(for: "flux-dev:q8", on: workstation.id)?.map(\.id) == ["a"])
        #expect(store.rows(for: "z-image-turbo:q8", on: workstation.id)?.map(\.id) == ["b"])
    }

    @Test func anUnknownModelReportsOnceAndDoesNotRetry() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.loraErrors["not-a-real-model"] =
            MoldClientError.http(status: 400, code: "UNKNOWN_MODEL", message: "unknown model")
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = LoraStore(hosts: hosts)

        await store.refresh(model: "not-a-real-model", on: workstation.id)
        await store.refresh(model: "not-a-real-model", on: workstation.id)

        #expect(backend.callCount("loras") == 1)
        #expect(store.rows(for: "not-a-real-model", on: workstation.id) == nil)
        #expect(hosts.failures.isEmpty)
    }

    // MARK: - AdaptersGroup.Rows

    @Test func theAddMenuOmitsAnAdapterAlreadyInTheStack() {
        let installed = [lora("a", path: "/models/a.safetensors"), lora("b", path: "/models/b.safetensors")]
        let chosen = [LoraChoice(path: "/models/a.safetensors", name: "a")]
        guard case let .rows(available, canAdd) = AdaptersGroup.Rows.resolve(
            installed: installed, chosen: chosen, maxCount: 4
        ) else {
            Issue.record("expected .rows")
            return
        }
        #expect(available.map(\.id) == ["b"])
        #expect(canAdd)
    }

    @Test func theAddMenuDisappearsAtTheRecipesOwnLimit() {
        let installed = [lora("a", path: "/models/a.safetensors"), lora("b", path: "/models/b.safetensors")]
        let chosen = [LoraChoice(path: "/models/a.safetensors", name: "a")]
        guard case let .rows(_, canAdd) = AdaptersGroup.Rows.resolve(
            installed: installed, chosen: chosen, maxCount: 1
        ) else {
            Issue.record("expected .rows")
            return
        }
        #expect(canAdd == false)
    }

    @Test func anEmptyAnswerIsASentenceNotAnEmptyPicker() {
        #expect(AdaptersGroup.Rows.resolve(installed: [], chosen: [], maxCount: 4) == .empty)
        #expect(AdaptersGroup.Rows.resolve(installed: nil, chosen: [], maxCount: 4) == .none)
    }

    // MARK: - Identity

    @Test func identityStartStepFollowsTheStepControlDown() {
        var draft = RenderDraft()
        draft.media.identity = IdentityConditioning(
            photos: [IdentityPhoto(encoded: "x", name: "a")], startStep: 12
        )
        draft.steps = 4
        #expect(draft.media.identity?.startStep == 3)
    }

    @Test func theIdentityGroupNeedsBothTheRecipeAndTheHost() {
        let qualified = FakeFixtures.recipe(supportsIdentity: true)
        let unqualified = FakeFixtures.recipe(supportsIdentity: false)
        let identityHost = FakeFixtures.capabilities(identity: true)
        let noIdentityHost = FakeFixtures.capabilities(identity: false)

        #expect(IdentityGroup.isShown(recipe: qualified, host: identityHost))
        #expect(IdentityGroup.isShown(recipe: qualified, host: noIdentityHost) == false)
        #expect(IdentityGroup.isShown(recipe: unqualified, host: identityHost) == false)
        #expect(IdentityGroup.isShown(recipe: unqualified, host: nil) == false)
    }
}
