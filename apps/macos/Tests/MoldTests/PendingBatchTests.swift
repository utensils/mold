import Foundation
import MoldClient
import Testing

@testable import Mold

/// `PendingBatch` records were written and never read -- `remember` fired on
/// every submit but nothing called `all()` back. These pin the round trip
/// through the suite the app actually uses (`AppStorageSuite`, not
/// `.standard`, so `MOLD_NATIVE_FRESH` isn't leaked into a real launch).
@MainActor
struct PendingBatchTests {
    private func scratch() -> UserDefaults {
        let name = "io.utensils.mold.native.tests.pendingbatch.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return defaults
    }

    @Test func aRememberedBatchIsReadBackByAll() {
        let defaults = scratch()
        let host = MoldHost(name: "plato", baseURL: URL(string: "http://plato")!)
        PendingBatch.remember("batch-1", host: host.id, in: defaults)
        #expect(PendingBatch.all(in: defaults)["batch-1"] == host.id.uuidString)
    }

    @Test func forgettingRemovesOnlyThatBatch() {
        let defaults = scratch()
        let host = MoldHost(name: "plato", baseURL: URL(string: "http://plato")!)
        PendingBatch.remember("batch-1", host: host.id, in: defaults)
        PendingBatch.remember("batch-2", host: host.id, in: defaults)
        PendingBatch.forget("batch-1", in: defaults)
        #expect(PendingBatch.all(in: defaults).keys.sorted() == ["batch-2"])
    }
}
