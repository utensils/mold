import Foundation
import MoldClient
import Testing

@testable import Mold

/// `QueueHoldRow`'s own pure logic -- what a hold offers -- and
/// Pull-then-Retry's orchestration across `DownloadStore` and `QueueStore`
/// (design M6 S3, decision 12).
@MainActor
struct QueueHoldTests {
    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    // MARK: - What a hold offers

    /// **Fails today**: `QueueHoldRow` does not exist.
    @Test func aMissingModelHoldOffersPullThenRetry() {
        let hold = QueueHold.missingModel("z-image-turbo", sentence: "Mold can't find z-image-turbo.")
        #expect(QueueHoldRow.actions(for: hold) == [.pullThenRetry(model: "z-image-turbo")])
    }

    @Test func aRetryableProseHoldOffersTryAgain() {
        let hold = QueueHold.prose("GPU ran out of memory.", retryable: true)
        #expect(QueueHoldRow.actions(for: hold) == [.tryAgain])
    }

    @Test func anUnretryableHoldOffersNothing() {
        let hold = QueueHold.prose("Something needs repair on the host.", retryable: false)
        #expect(QueueHoldRow.actions(for: hold).isEmpty)
    }

    // MARK: - Pull-then-Retry

    /// **Fails today**: `DownloadStore.awaitSettlement` and
    /// `QueueHoldRow.pullThenRetry` do not exist.
    @Test func pullThenRetryRetriesOnlyAfterTheDownloadSettles() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.downloadTicket = FakeFixtures.downloadTicket("job-pull")
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        hosts.reachability[plato.id] = .up(FakeFixtures.serverStatus(instanceId: "run-1"))
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))
        let queue = QueueStore(hosts: hosts)
        let entry = FakeFixtures.queueEntry(
            "job-1", state: "held", batchId: "batch-1", clientBatchId: "client-1", model: "z-image-turbo")

        let orchestration = Task {
            await QueueHoldRow.pullThenRetry(
                "z-image-turbo", entry: entry, host: plato, downloads: downloads, queue: queue)
        }
        await settle { fake.callCount("downloadEvents") == 1 }
        #expect(fake.callCount("retryJob") == 0)

        fake.downloadStream?.yield(
            FakeFixtures.downloadEvent(type: "job_done", id: "job-pull", model: "z-image-turbo"))
        await orchestration.value

        #expect(fake.callCount("retryJob") == 1)
        #expect(fake.retriedAuthorities.first?.jobId == "job-1")
    }

    /// A cancelled or failed pull never retries -- the hold's own sentence
    /// stays as it was, and the download's own error is what the popover
    /// shows.
    @Test func aFailedPullLeavesTheRowHeld() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.downloadTicket = FakeFixtures.downloadTicket("job-pull")
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))
        let queue = QueueStore(hosts: hosts)
        let entry = FakeFixtures.queueEntry(
            "job-1", state: "held", batchId: "batch-1", clientBatchId: "client-1", model: "z-image-turbo")

        let orchestration = Task {
            await QueueHoldRow.pullThenRetry(
                "z-image-turbo", entry: entry, host: plato, downloads: downloads, queue: queue)
        }
        await settle { fake.callCount("downloadEvents") == 1 }

        fake.downloadStream?.yield(
            FakeFixtures.downloadEvent(type: "job_failed", id: "job-pull", model: "z-image-turbo", error: "disk full"))
        await orchestration.value

        #expect(fake.callCount("retryJob") == 0)
        #expect(downloads.finished[plato.id]?.first?.error == "disk full")
    }
}
