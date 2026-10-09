import Foundation
import Testing
@testable import MoldClient

@MainActor @Test func newerVisibilityIntentFencesOlderReplyAndPersistsPendingTargets() throws {
    let host = MoldHost(name: "remote", baseURL: URL(string: "http://remote:7680")!)
    let ledger = CollectionVisibilityLedger()
    let hide = ledger.set("hidden", hidden: true, hosts: [host])
    let show = ledger.set("hidden", hidden: false, hosts: [host])
    #expect(!ledger.isCurrent(slug: "hidden", revision: hide.revision, host: host))
    #expect(ledger.isCurrent(slug: "hidden", revision: show.revision, host: host))
    var replacement = host; replacement.baseURL = URL(string: "http://other:7680")!
    #expect(!ledger.isCurrent(slug: "hidden", revision: show.revision, host: replacement))
    let defaults = UserDefaults(suiteName: UUID().uuidString)!
    ledger.persist(to: defaults, key: "visibility")
    let restored = CollectionVisibilityLedger.load(from: defaults, key: "visibility")
    #expect(restored.desiredHidden(slug: "hidden", fallback: true) == false)
    #expect(restored.isCurrent(slug: "hidden", revision: show.revision, host: host))
    ledger.complete(slug: "hidden", revision: hide.revision)
    #expect(ledger.intents["hidden"] != nil)
    ledger.complete(slug: "hidden", revision: show.revision)
    #expect(ledger.intents.isEmpty)
}

@MainActor @Test func mixedHiddenReplicasRepairAndOfflineShowRetriesWithoutOverride() async {
    let one = MoldHost(name: "one", baseURL: URL(string: "http://one")!)
    let two = MoldHost(name: "two", baseURL: URL(string: "http://two")!)
    var collections: [UUID: [Collection]] = [
        one.id: [Collection(id: "1", name: "Hidden", slug: "hidden", hidden: true)],
        two.id: [Collection(id: "2", name: "Hidden", slug: "hidden", hidden: false)],
    ]
    var available: Set<UUID> = [one.id, two.id]
    var writes: [(UUID, Bool)] = []
    let ledger = CollectionVisibilityLedger()
    func reconcile() async {
        await ledger.reconcile(hosts: { [one, two] }, collections: { collections }, available: { available }) { host, collection, hidden in
            writes.append((host.id, hidden))
            collections[host.id] = [Collection(id: collection.id, name: collection.name, slug: collection.slug, hidden: hidden)]
            return true
        }
    }
    await reconcile()
    #expect(writes.count == 1 && writes[0].0 == two.id && writes[0].1)
    #expect(ledger.intents["hidden"] != nil) // confirm on a later inventory
    await reconcile()
    #expect(ledger.intents.isEmpty)
    ledger.set("hidden", hidden: false, hosts: [one, two])
    available = [one.id]
    await reconcile()
    #expect(ledger.desiredHidden(slug: "hidden", fallback: true) == false)
    #expect(ledger.intents["hidden"] != nil)
    available.insert(two.id)
    await reconcile()
    await reconcile()
    #expect(ledger.intents.isEmpty)
    #expect(collections.values.flatMap { $0 }.allSatisfy { $0.hidden == false })
}

@MainActor @Test func supersededInFlightHideCannotRetireNewShowIntent() async {
    let host = MoldHost(name: "remote", baseURL: URL(string: "http://remote")!)
    let slug = "hidden"
    let ledger = CollectionVisibilityLedger()
    ledger.set(slug, hidden: true, hosts: [host])
    var rows = [host.id: [Collection(id: "one", name: "Hidden", slug: slug, hidden: false)]]
    await ledger.reconcile(hosts: { [host] }, collections: { rows }, available: { [host.id] }) { _, _, _ in
        ledger.set(slug, hidden: false, hosts: [host])
        rows[host.id] = [Collection(id: "one", name: "Hidden", slug: slug, hidden: true)]
        return true
    }
    #expect(ledger.desiredHidden(slug: slug, fallback: true) == false)
    #expect(ledger.intents[slug] != nil)
    await ledger.reconcile(hosts: { [host] }, collections: { rows }, available: { [host.id] }) { _, _, hidden in
        #expect(hidden == false)
        rows[host.id] = [Collection(id: "one", name: "Hidden", slug: slug, hidden: hidden)]
        return true
    }
    #expect(rows[host.id]?.first?.hidden == false)
}
