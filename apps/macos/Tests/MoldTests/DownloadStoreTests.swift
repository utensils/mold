import Foundation
import MoldClient
import Testing

@testable import Mold

/// `DownloadStore`'s install routing, licence-refusal recovery, and adopting
/// a machine's whole download queue rather than only this app's own clicks.
@MainActor
struct DownloadStoreTests {
    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func refusal() -> LicenseRefusal {
        LicenseRefusal(
            id: "tencent-hunyuan3d-2.1", name: "Tencent Hunyuan3D 2.1", url: "https://example/LICENSE",
            canonical: "https://example/page", sha256: "abc123", summary: "Non-commercial research only.")
    }

    /// **Fails today**: `install` always calls `startDownload`, whatever the
    /// name looks like.
    @Test func aCatalogIdAndAManifestNameTakeDifferentDoors() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.downloadTicket = FakeFixtures.downloadTicket("job-manifest")
        fake.catalogInstallAnswer = FakeFixtures.catalogInstall(primary: "job-catalog")
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))

        await downloads.install("flux-dev:q4", on: plato)
        await downloads.install("cv:252914", on: plato)

        #expect(fake.startedDownloads == ["flux-dev:q4"])
        #expect(fake.catalogInstalls == ["cv:252914"])
    }

    @Test func aCompanionOnlyInstallIsStillWatched() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.catalogInstallAnswer = FakeFixtures.catalogInstall(
            primary: nil, companions: [("clip-l", "job-a"), ("t5", "job-b")])
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))

        await downloads.install("cv:252914", on: plato)

        #expect(Set((downloads.active[plato.id] ?? [:]).keys) == ["job-a", "job-b"])
    }

    /// **Fails today**: the 403 decodes to `.licenseRequired`, but `install`
    /// hands every error to `hosts.report`, so it lands in the banner as
    /// prose instead of on `pendingLicense`.
    @Test func aLicenceRefusalIsHeldForASheetAndNeverReachesTheBanner() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.plantedErrors["startDownload"] = MoldClientError.licenseRequired(refusal(), mismatch: false)
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))

        await downloads.install("hunyuan3d-2.1:fp16", on: plato)

        #expect(hosts.failures.isEmpty)
        #expect(downloads.pendingLicense?.refusal.id == "tencent-hunyuan3d-2.1")
    }

    @Test func acceptingAndRetryingStartsTheSameDownload() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.plantedErrors["startDownload"] = MoldClientError.licenseRequired(refusal(), mismatch: false)
        fake.downloadTicket = FakeFixtures.downloadTicket("job-1")
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let licenses = LicenseStore(hosts: hosts)
        let downloads = DownloadStore(hosts: hosts, licenses: licenses)
        await downloads.install("hunyuan3d-2.1:fp16", on: plato)
        guard let pending = downloads.pendingLicense else {
            Issue.record("expected a pending licence")
            return
        }

        fake.plantedErrors["startDownload"] = nil
        await downloads.accepted(pending)

        #expect(downloads.pendingLicense == nil)
        #expect(fake.acceptedLicenses.last?.map(\.id) == ["tencent-hunyuan3d-2.1"])
        // Two attempts reached the route -- the refused first, and the
        // retry -- and the one that got through carried the identical name.
        // (The fake only records a call's argument once it clears the
        // planted-error check, the same convention `deleteModel` and
        // `installCatalogEntry` already follow.)
        #expect(fake.callCount("startDownload") == 2)
        #expect(fake.startedDownloads == ["hunyuan3d-2.1:fp16"])
    }

    /// **Fails today**: nothing decodes a `snapshot` frame, and `active` only
    /// ever holds jobs this store's own `install` started.
    @Test func aSnapshotFrameShowsADownloadThisAppNeverStarted() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.downloadsListing = DownloadsListing(
            activeJobs: [DownloadJob(id: "job-mine", model: "flux-dev:q4", status: .active)])
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))

        await downloads.refresh(on: plato.id)
        await settle { fake.callCount("downloadEvents") == 1 }

        fake.downloadStream?.yield(FakeFixtures.downloadEvent(
            type: "snapshot",
            listing: DownloadsListing(activeJobs: [
                DownloadJob(id: "job-mine", model: "flux-dev:q4", status: .active),
                DownloadJob(id: "job-cli", model: "sd15:fp16", status: .active),
            ])))
        await settle { downloads.active[plato.id]?.count == 2 }

        #expect(Set((downloads.active[plato.id] ?? [:]).keys) == ["job-mine", "job-cli"])
    }

    @Test func aFinishedJobStaysInThePopoversListAndDoesNotHoldAStream() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.downloadTicket = FakeFixtures.downloadTicket("job-1")
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))
        await downloads.install("flux-dev:q4", on: plato)
        await settle { fake.callCount("downloadEvents") == 1 }

        fake.downloadStream?.yield(FakeFixtures.downloadEvent(type: "job_done", id: "job-1", model: "flux-dev:q4"))
        await settle { downloads.active[plato.id] == nil }

        #expect(downloads.finished[plato.id]?.map(\.id) == ["job-1"])
        #expect(downloads.streams[plato.id] == nil)

        downloads.clearFinished(on: plato.id)
        #expect(downloads.finished[plato.id] == nil)
    }
}
