import Foundation
import MoldClient
import Testing

@testable import Mold

/// The downloads toolbar button's presence rule, its combined progress
/// figure, and the popover's row list -- all pure functions over the same
/// per-machine dictionaries `DownloadStore` already keeps, so none of this
/// needs a rendered view.
@MainActor
struct DownloadsPopoverTests {
    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func progress(
        model: String, bytesDone: Int64? = nil, bytesTotal: Int64? = nil
    ) -> DownloadStore.Progress {
        DownloadStore.Progress(model: model, bytesDone: bytesDone, bytesTotal: bytesTotal)
    }

    // MARK: - Button presence

    /// **Fails today**: `DownloadsButton` does not exist yet.
    @Test func theButtonIsAbsentOnAnIdleMachine() {
        #expect(DownloadsButton.isShown(active: [:], finished: []) == false)
    }

    @Test func presentWhileSomethingIsQueued() {
        let queued = ["job-1": progress(model: "flux-dev:q4")]
        #expect(DownloadsButton.isShown(active: queued, finished: []) == true)
    }

    @Test func presentWithOnlyAFinishedJobThisLaunch() {
        let finished = [DownloadJob(id: "job-1", model: "flux-dev:q4", status: .completed)]
        #expect(DownloadsButton.isShown(active: [:], finished: finished) == true)
    }

    // MARK: - Combined fraction

    /// **Fails today**: no combined-fraction function exists.
    @Test func theCombinedFractionIgnoresAJobThatReportsNoTotal() {
        let active = [
            "job-queued": progress(model: "sd15:fp16"),
            "job-active": progress(model: "flux-dev:q4", bytesDone: 50, bytesTotal: 100),
        ]
        #expect(DownloadsButton.combinedFraction(active: active) == 0.5)
    }

    @Test func theCombinedFractionIsNilWhenNoJobReportsATotal() {
        let active = ["job-queued": progress(model: "sd15:fp16")]
        #expect(DownloadsButton.combinedFraction(active: active) == nil)
    }

    // MARK: - Popover rows

    /// **Fails today**: `DownloadsPopover.Rows` does not exist yet.
    @Test func rowsResolveActiveBeforeFinished() {
        let active = ["job-active": progress(model: "flux-dev:q4", bytesDone: 10, bytesTotal: 20)]
        let finished = [DownloadJob(id: "job-done", model: "sd15:fp16", status: .completed)]

        let rows = DownloadsPopover.Rows.resolve(active: active, finished: finished)

        #expect(rows.map(\.id) == ["job-active", "job-done"])
        #expect(rows.map(\.isActive) == [true, false])
    }

    @Test func aQueuedRowWithNoTotalReadsAsWaiting() {
        let active = ["job-queued": progress(model: "sd15:fp16")]
        let rows = DownloadsPopover.Rows.resolve(active: active, finished: [])
        #expect(rows.first?.detailText == "Waiting")
    }

    /// A cancelled job never lands in `finished` at all -- `DownloadStore.
    /// cancel` drops it straight out of `active` -- so once a store has
    /// forgotten a job, `Rows.resolve` over its own dictionaries has nothing
    /// left to show for it. A job the stream reported failed DOES stay, with
    /// its own sentence.
    @Test func aCancelledRowLeavesTheListAndAFailedOneStaysWithItsReason() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.downloadTicket = FakeFixtures.downloadTicket("job-cancel")
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))
        await downloads.install("flux-dev:q4", on: plato)

        await downloads.cancel(jobID: "job-cancel", on: plato)
        downloads.apply(
            FakeFixtures.downloadEvent(type: "job_failed", id: "job-other", model: "sd15:fp16", error: "disk full"),
            on: plato.id)

        let rows = DownloadsPopover.Rows.resolve(
            active: downloads.active[plato.id] ?? [:], finished: downloads.finished[plato.id] ?? [])

        #expect(rows.map(\.id) == ["job-other"])
        #expect(rows.first?.detailText == "disk full")
        #expect(rows.first?.isFailed == true)
    }

    // MARK: - Clear

    @Test func clearEmptiesOnlyThatMachinesFinishedRows() {
        let plato = machine("plato")
        let hal = machine("hal9000")
        let fake = FakeBackend(host: plato)
        let hosts = HostStore(hosts: [plato, hal]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))
        downloads.apply(
            FakeFixtures.downloadEvent(type: "job_done", id: "job-plato", model: "flux-dev:q4"), on: plato.id)
        downloads.apply(
            FakeFixtures.downloadEvent(type: "job_done", id: "job-hal", model: "sd15:fp16"), on: hal.id)

        downloads.clearFinished(on: plato.id)

        #expect(DownloadsPopover.Rows.resolve(active: [:], finished: downloads.finished[plato.id] ?? []).isEmpty)
        #expect(
            DownloadsPopover.Rows.resolve(active: [:], finished: downloads.finished[hal.id] ?? []).map(\.id)
                == ["job-hal"])
    }
}
