import Foundation
import MoldClient
import MoldClientTesting
import Testing

@testable import MoldCompanion

/// Background refresh's ledger: a render whose machine was removed, or
/// that its machine no longer knows (it restarted), stops being waited for.
@MainActor
struct BackgroundReconcileTests {
    private func stores(_ fake: FakeBackend) async throws -> CompanionStores {
        fake.stub("status()", returning: try QueueStoreTests.decode(ServerStatus.self,
            #"{"version":"0.32.0","busy":false,"uptime_secs":1,"instance_id":"inst-1"}"#))
        fake.stub("capabilities()", returning: try QueueStoreTests.decode(Capabilities.self, "{}"))
        fake.stub("models()", returning: [Model]())
        let file = HostListFile(url: FileManager.default.temporaryDirectory.appending(path: "h-\(UUID()).json"))
        let stores = CompanionStores(list: file, credentials: HostStoreTests.MemoryCredentials(),
                                     makeBackend: { _ in fake })
        try stores.hosts.add(name: "workstation", address: "10.0.0.4", apiKey: nil, makeDefault: true)
        await stores.hosts.refreshAll()
        for batch in stores.generate.ledger.batches { stores.generate.ledger.remove(batch.clientBatchId) }
        return stores
    }

    @Test func aRenderOnARemovedMachineIsForgotten() async throws {
        let stores = try await stores(FakeBackend())
        let batch = ActiveBatch(id: "b1", clientBatchId: "c-\(UUID())", host: UUID(), prompt: "p", startedAt: .now)
        stores.generate.ledger.add(batch)
        await stores.reconcile(batch)
        #expect(!stores.generate.ledger.batches.contains(batch))
    }

    @Test func aRenderTheMachineNoLongerKnowsIsForgotten() async throws {
        let fake = FakeBackend()
        let stores = try await stores(fake)
        fake.stub("batchStatus(id:)", throwing: MoldClientError.http(status: 404, code: nil, message: nil))
        let batch = ActiveBatch(id: "b1", clientBatchId: "c-\(UUID())", host: stores.hosts.hosts[0].id,
                                prompt: "p", startedAt: .now)
        stores.generate.ledger.add(batch)
        await stores.reconcile(batch)
        #expect(!stores.generate.ledger.batches.contains(batch))
    }

    @Test func aDayOldRenderIsNotWaitedForAnyMore() async throws {
        let stores = try await stores(FakeBackend())
        let batch = ActiveBatch(id: "b1", clientBatchId: "c-\(UUID())", host: stores.hosts.hosts[0].id,
                                prompt: "p", startedAt: .now.addingTimeInterval(-25 * 60 * 60))
        stores.generate.ledger.add(batch)
        await stores.reconcile(batch)
        #expect(!stores.generate.ledger.batches.contains(batch))
    }
}
