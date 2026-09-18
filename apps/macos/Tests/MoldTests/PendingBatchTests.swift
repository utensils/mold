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
        let host = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
        PendingBatch.remember("batch-1", host: host.id, in: defaults)
        #expect(PendingBatch.all(in: defaults)["batch-1"] == host.id.uuidString)
    }

    @Test func forgettingRemovesOnlyThatBatch() {
        let defaults = scratch()
        let host = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
        PendingBatch.remember("batch-1", host: host.id, in: defaults)
        PendingBatch.remember("batch-2", host: host.id, in: defaults)
        PendingBatch.forget("batch-1", in: defaults)
        #expect(PendingBatch.all(in: defaults).keys.sorted() == ["batch-2"])
    }

    /// UAT 2026-09-17 #10: five entries keyed on a removed machine survived
    /// both Remove and Reset. The reset keeps the key on purpose; the
    /// removal must clear its own machine's share.
    ///
    /// **Fails today**: nothing clears by machine.
    @Test func removingAMachineForgetsEverythingPendingOnIt() {
        let defaults = scratch()
        let gone = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
        let kept = MoldHost(name: "hal9000", baseURL: URL(string: "http://hal9000")!)
        PendingBatch.remember("batch-1", host: gone.id, in: defaults)
        PendingBatch.remember("batch-2", host: kept.id, in: defaults)
        PendingChain.remember("job-1", host: gone.id, in: defaults)
        PendingChain.remember("job-2", host: kept.id, in: defaults)

        PendingBatch.forgetAll(on: gone.id, in: defaults)
        PendingChain.forgetAll(on: gone.id, in: defaults)

        #expect(PendingBatch.all(in: defaults).keys.sorted() == ["batch-2"])
        #expect(PendingChain.all(in: defaults).keys.sorted() == ["job-2"])
    }
}
