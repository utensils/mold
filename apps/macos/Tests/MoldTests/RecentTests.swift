import Foundation
import MoldClient
import Testing

@testable import Mold

/// The wand's own visibility, and Recent: a per-machine list of what was
/// asked for, offered back as a starting point and never as a full restore.
@MainActor
struct RecentTests {
    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    // MARK: - Picking a row

    @Test func pickingARecentPromptChangesThePromptAndNothingElse() {
        var draft = RenderDraft()
        draft.prompt = "a cat"
        draft.steps = 40
        draft.width = 512
        draft.height = 512
        draft.seed = 12345
        let entry = HistoryEntry(prompt: "a brass orrery on a walnut desk", model: "flux2-dev:q8", usedAt: 1000)

        RecentGroup.pick(entry, into: &draft)

        #expect(draft.prompt == "a brass orrery on a walnut desk")
        #expect(draft.steps == 40)
        #expect(draft.width == 512)
        #expect(draft.height == 512)
        #expect(draft.seed == 12345)
    }

    // MARK: - What the section says, versus what the machine says

    @Test func aMachineWithNoHistoryFeatureSaysSoRatherThanLookingEmpty() {
        #expect(RecentGroup.Listing.resolve(hasLoaded: false, isUnavailable: false, entries: []) == .loading)
        #expect(RecentGroup.Listing.resolve(hasLoaded: true, isUnavailable: true, entries: []) == .unavailable)
        #expect(RecentGroup.Listing.resolve(hasLoaded: true, isUnavailable: false, entries: []) == .empty)
        let entry = HistoryEntry(prompt: "a cat", model: "flux-dev:q8", usedAt: 1)
        #expect(RecentGroup.Listing.resolve(hasLoaded: true, isUnavailable: false, entries: [entry]) == .rows([entry]))
    }

    // MARK: - Clearing asks first

    @Test func clearingAsksFirst() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.historyRows = [HistoryEntry(prompt: "a cat", model: "flux-dev:q8", usedAt: 1)]
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let history = PromptHistoryStore(hosts: hosts)
        await history.refresh(on: plato.id)

        let destruction = RecentGroup.clearDestruction(machine: plato.name) {
            Task { await history.clear(on: plato.id) }
        }

        #expect(destruction.title == "Clear the prompt history on plato?")
        #expect(destruction.verb == "Clear")
        #expect(backend.callCount("clearHistory") == 0)

        destruction.perform()
        await settle { backend.callCount("clearHistory") == 1 }
        #expect(backend.callCount("clearHistory") == 1)
    }

    // MARK: - Per machine

    @Test func aRecentRowFromAnotherMachineIsNotOfferedForThisOne() async {
        let plato = machine("plato")
        let hal = machine("hal9000")
        let platoBackend = FakeBackend(host: plato)
        platoBackend.historyRows = [HistoryEntry(prompt: "a brass gear", model: "flux-dev:q8", usedAt: 1)]
        let halBackend = FakeBackend(host: hal)
        halBackend.historyRows = [HistoryEntry(prompt: "a glowing orb", model: "flux2-dev:q8", usedAt: 2)]
        let hosts = HostStore(hosts: [plato, hal]) { host in
            host.id == plato.id ? platoBackend : halBackend
        }
        let history = PromptHistoryStore(hosts: hosts)

        await history.refresh(on: plato.id)
        await history.refresh(on: hal.id)

        #expect(history.entries(on: plato.id).map(\.prompt) == ["a brass gear"])
        #expect(history.entries(on: hal.id).map(\.prompt) == ["a glowing orb"])
    }

    // MARK: - The wand's own visibility

    @Test func theWandIsAbsentWhereTheRecipeReadsNoPrompt() {
        let visibility = PromptWand.Visibility.resolve(
            offer: .wand(canRemix: true), promptMode: .ignored, prompt: "a cat")
        #expect(visibility == .hidden)
    }

    @Test func holdingOptionOffersRemixOnlyWhereTheMachineRemixes() {
        #expect(PromptWand.wantsRemix(optionHeld: true, canRemix: true))
        #expect(!PromptWand.wantsRemix(optionHeld: true, canRemix: false))
        #expect(!PromptWand.wantsRemix(optionHeld: false, canRemix: true))
    }
}
