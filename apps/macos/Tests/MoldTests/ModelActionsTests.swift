import Foundation
import MoldClient
import Testing

@testable import Mold

/// The one door for delete/load/unload/components/licence -- `ModelActions`
/// -- and the pure menu both the contextual menu and the Model menu draw
/// from (M5 S5).
@MainActor
struct ModelActionsTests {
    private func machine(_ name: String = "workstation") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func actions(
        hosts: HostStore, models: ModelStore, downloads: DownloadStore, licenses: LicenseStore,
        confirmDestruction: ((Destruction) -> Void)? = nil
    ) -> ModelActions {
        ModelActions(hosts: hosts, models: models, downloads: downloads, licenses: licenses,
                     confirmDestruction: confirmDestruction)
    }

    /// **Fails today**: there is no `ModelActions`.
    @Test func theMenuOffersUnloadOnlyForLoadedAndLoadOnlyForInstalled() {
        let model = FakeFixtures.model("flux-dev:q4", downloaded: true)
        func kinds(_ state: ModelInstallState) -> [ModelActions.Kind] {
            ModelActions.menu(for: model, installState: state, isBusy: false, isDownloading: false, licensed: false)
                .compactMap(\.kind)
        }
        #expect(!kinds(.available(nil)).contains(.load))
        #expect(!kinds(.needsRepair(10)).contains(.load))
        #expect(kinds(.installed).contains(.load))
        #expect(!kinds(.installed).contains(.unload))
        #expect(kinds(.loaded).contains(.unload))
        #expect(!kinds(.loaded).contains(.load))
    }

    /// **Fails today**: nothing marks Delete… destructive, so both menus draw
    /// it straight under Show Licence… with nothing between them -- a right
    /// click can land the destructive item under the cursor. Every other menu
    /// in this app puts its destructive item last and behind a divider, which
    /// is now `RowAction.rendered`'s rule rather than a per-item flag.
    @Test func deleteIsLastAndBehindADividerWhereverItIsDrawn() {
        let model = FakeFixtures.model("flux-dev:q4", downloaded: true)
        for state in [ModelInstallState.installed, .loaded, .needsRepair(10)] {
            let drawn = RowAction.rendered(ModelActions.menu(
                for: model, installState: state, isBusy: false, isDownloading: false, licensed: true))
            #expect(drawn.last?.kind == .delete)
            #expect(drawn.last?.isDestructive == true)
            #expect(drawn.dropLast().last?.isSeparator == true)
        }
        // A row with nothing installed has nothing to delete, so there is no
        // divider either.
        let available = RowAction.rendered(ModelActions.menu(
            for: model, installState: .available(nil), isBusy: false, isDownloading: false, licensed: false))
        #expect(!available.contains { $0.isSeparator })
    }

    @Test func showLicenceAppearsOnlyForAGatedModel() {
        let model = FakeFixtures.model("flux-dev:q4", downloaded: true)
        let gated = ModelActions.menu(for: model, installState: .installed, isBusy: false,
                                       isDownloading: false, licensed: true)
        let ungated = ModelActions.menu(for: model, installState: .installed, isBusy: false,
                                         isDownloading: false, licensed: false)
        #expect(gated.compactMap(\.kind).contains(.licence))
        #expect(!ungated.compactMap(\.kind).contains(.licence))
    }

    @Test func deleteAsksFirstAndTheFakeRecordsNothingUntilPerform() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.removalAnswers["flux-dev:q4"] = FakeFixtures.modelRemoval(removed: ["flux-dev:q4"], freedBytes: 500_000_000)
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let models = ModelStore(hosts: hosts)
        let licenses = LicenseStore(hosts: hosts)
        let downloads = DownloadStore(hosts: hosts, licenses: licenses)
        var captured: Destruction?
        let acts = actions(hosts: hosts, models: models, downloads: downloads, licenses: licenses,
                            confirmDestruction: { captured = $0 })
        let model = FakeFixtures.model("flux-dev:q4", downloaded: true)

        acts.delete(model, on: workstation.id)

        #expect(captured != nil)
        #expect(captured?.title.contains(model.headline) == true)
        #expect(fake.callCount("deleteModel") == 0)

        captured?.perform()
        // Settled on what the assertion reads. `deleteModel` records its call
        // and appends its argument in the same turn, so a count adds nothing
        // here -- but `perform()` starts an unstructured `Task`, and on a
        // main actor shared with three other suites it can wait a while for
        // its first turn.
        await settle(until: { !fake.deletedModels.isEmpty })
        #expect(fake.deletedModels == ["flux-dev:q4"])
        #expect(fake.callCount("deleteModel") == 1)
    }

    @Test func aRemovalThatKeptSomethingSaysWhatAndWhy() {
        let removal = FakeFixtures.modelRemoval(
            removed: ["flux-dev:q4"], kept: [(component: "t5 encoder", usedBy: ["flux-schnell:q8"])],
            freedBytes: 500_000_000)
        let sentence = ModelActions.removalSummary(headline: "FLUX.1 Dev Q4", removal: removal)

        #expect(sentence.hasPrefix("Removed FLUX.1 Dev Q4 and freed"))
        #expect(sentence.contains("1 shared file kept for"))
        #expect(sentence.contains("flux-schnell:q8"))
    }

    @Test func aRemovalThatKeptNothingNamesOnlyWhatWasFreed() {
        let removal = FakeFixtures.modelRemoval(removed: ["flux-dev:q4"], freedBytes: 500_000_000)
        let sentence = ModelActions.removalSummary(headline: "FLUX.1 Dev Q4", removal: removal)

        #expect(sentence == "Removed FLUX.1 Dev Q4 and freed 500 MB.")
    }

    @Test func repairInstallsTheComponentTheMachineNamedNotTheModelWhoseSheetIsOpen() {
        let model = FakeFixtures.model("flux-schnell:q8", downloaded: true)
        let missing = FakeFixtures.modelComponentRow(
            kind: "t5 encoder", name: "model.safetensors", present: false, repairModel: "t5-xxl:fp16")
        let untitled = FakeFixtures.modelComponentRow(kind: "vae", name: "vae.safetensors", present: false)

        #expect(ComponentsSheet.repairTarget(for: missing, model: model) == "t5-xxl:fp16")
        #expect(ComponentsSheet.repairTarget(for: untitled, model: model) == "flux-schnell:q8")
    }

    /// **Fails today**: `modelComponents(_:statuses:)` and
    /// `modelComponentRow` don't exist yet.
    @Test func theComponentsRowsNeverExpandOptions() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        let heavy = FakeFixtures.modelComponentRow(
            kind: "transformer", name: "model.safetensors", present: true, optionsCount: 103)
        fake.componentRows["flux-schnell:q8"] = FakeFixtures.modelComponents("flux-schnell:q8", statuses: [heavy])
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let models = ModelStore(hosts: hosts)
        let model = FakeFixtures.model("flux-schnell:q8", downloaded: true)

        let response = await models.components(of: model, on: workstation.id)

        // The sheet draws one row per `response.components` entry -- this is
        // that exact count, which options never inflate (design fact 4).
        #expect(response?.components.count == 1)
        #expect(response?.components.first?.options?.count == 103)
    }

    @Test func theMenuAndTheContextualMenuCallTheSameThing() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        let model = FakeFixtures.model("flux-dev:q4", downloaded: true)
        fake.modelRows = [model]
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let models = ModelStore(hosts: hosts)
        let licenses = LicenseStore(hosts: hosts)
        let downloads = DownloadStore(hosts: hosts, licenses: licenses)
        let acts = actions(hosts: hosts, models: models, downloads: downloads, licenses: licenses)

        // The row's contextual menu and the Model menu both call
        // `ModelActions.perform` with the picked item's kind -- there is no
        // second copy of "what Load means" for either surface to drift from.
        acts.perform(.load, on: model, host: workstation)
        await settle(until: { fake.callCount("loadModel") == 1 })
        acts.perform(.load, on: model, host: workstation)
        await settle(until: { fake.callCount("loadModel") == 2 })

        #expect(fake.loadedModels.map(\.model) == ["flux-dev:q4", "flux-dev:q4"])
    }
}
