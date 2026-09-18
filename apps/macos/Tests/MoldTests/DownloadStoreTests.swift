import Foundation
import MoldClient
import Testing

@testable import Mold

/// `DownloadStore`'s install routing, licence-refusal recovery, and adopting
/// a machine's whole download queue rather than only this app's own clicks.
@MainActor
struct DownloadStoreTests {
    private func machine(_ name: String = "workstation") -> MoldHost {
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
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.downloadTicket = FakeFixtures.downloadTicket("job-manifest")
        fake.catalogInstallAnswer = FakeFixtures.catalogInstall(primary: "job-catalog")
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))

        await downloads.install("flux-dev:q4", on: workstation)
        await downloads.install("cv:252914", on: workstation)

        #expect(fake.startedDownloads == ["flux-dev:q4"])
        #expect(fake.catalogInstalls == ["cv:252914"])
    }

    @Test func aCompanionOnlyInstallIsStillWatched() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.catalogInstallAnswer = FakeFixtures.catalogInstall(
            primary: nil, companions: [("clip-l", "job-a"), ("t5", "job-b")])
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))

        await downloads.install("cv:252914", on: workstation)

        #expect(Set((downloads.active[workstation.id] ?? [:]).keys) == ["job-a", "job-b"])
    }

    /// **Fails today**: the 403 decodes to `.licenseRequired`, but `install`
    /// hands every error to `hosts.report`, so it lands in the banner as
    /// prose instead of on `pendingLicense`.
    @Test func aLicenceRefusalIsHeldForASheetAndNeverReachesTheBanner() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.plantedErrors["startDownload"] = MoldClientError.licenseRequired(refusal(), mismatch: false)
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))

        await downloads.install("hunyuan3d-2.1:fp16", on: workstation)

        #expect(hosts.failures.isEmpty)
        #expect(downloads.pendingLicense?.refusal.id == "tencent-hunyuan3d-2.1")
    }

    @Test func acceptingAndRetryingStartsTheSameDownload() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.plantedErrors["startDownload"] = MoldClientError.licenseRequired(refusal(), mismatch: false)
        fake.downloadTicket = FakeFixtures.downloadTicket("job-1")
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let licenses = LicenseStore(hosts: hosts)
        let downloads = DownloadStore(hosts: hosts, licenses: licenses)
        await downloads.install("hunyuan3d-2.1:fp16", on: workstation)
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
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.downloadsListing = DownloadsListing(
            activeJobs: [DownloadJob(id: "job-mine", model: "flux-dev:q4", status: .active)])
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))

        await downloads.refresh(on: workstation.id)
        await settle { fake.callCount("downloadEvents") == 1 }

        fake.downloadStream?.yield(FakeFixtures.downloadEvent(
            type: "snapshot",
            listing: DownloadsListing(activeJobs: [
                DownloadJob(id: "job-mine", model: "flux-dev:q4", status: .active),
                DownloadJob(id: "job-cli", model: "sd15:fp16", status: .active),
            ])))
        await settle { downloads.active[workstation.id]?.count == 2 }

        #expect(Set((downloads.active[workstation.id] ?? [:]).keys) == ["job-mine", "job-cli"])
    }

    @Test func aFinishedJobStaysInThePopoversListAndDoesNotHoldAStream() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.downloadTicket = FakeFixtures.downloadTicket("job-1")
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))
        await downloads.install("flux-dev:q4", on: workstation)
        await settle { fake.callCount("downloadEvents") == 1 }

        fake.downloadStream?.yield(FakeFixtures.downloadEvent(type: "job_done", id: "job-1", model: "flux-dev:q4"))
        await settle { downloads.active[workstation.id] == nil }

        #expect(downloads.finished[workstation.id]?.map(\.id) == ["job-1"])
        #expect(downloads.streams[workstation.id] == nil)

        downloads.clearFinished(on: workstation.id)
        #expect(downloads.finished[workstation.id] == nil)
    }

    /// **Fails today**: `catalog_ready` has a non-nil `id` and is not
    /// terminal, so it falls into the progress branch and inserts a row keyed
    /// by the CATALOG id -- nameless, stuck on "Starting…", with a Cancel
    /// that would `DELETE /api/downloads/hf%3Aowner%2Frepo` and 404. Nothing
    /// ever removes it, so the toolbar reads "downloading" and the SSE
    /// connection is held open for the rest of the launch.
    @Test func aCatalogReadyFrameLeavesNoRowBehind() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.catalogInstallAnswer = FakeFixtures.catalogInstall(primary: "job-1")
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))
        await downloads.install("hf:black-forest-labs/FLUX.1-dev", on: workstation)
        await settle { fake.callCount("downloadEvents") == 1 }

        // What the server sends once every job in the group has settled
        // (`downloads.rs:521-523`); its `id` is the CATALOG entry's, and
        // there is no job by that name. The `progress` behind it is a
        // barrier: one stream, delivered in order, so seeing its effect
        // means the frame before it is fully applied.
        fake.downloadStream?.yield(FakeFixtures.downloadEvent(
            type: "catalog_ready", id: "hf:black-forest-labs/FLUX.1-dev"))
        fake.downloadStream?.yield(FakeFixtures.downloadEvent(
            type: "progress", id: "job-1", bytesDone: 7, bytesTotal: 10))
        await settle { downloads.active[workstation.id]?["job-1"]?.bytesDone == 7 }

        #expect(Set((downloads.active[workstation.id] ?? [:]).keys) == ["job-1"])

        // And with the real job settled there is nothing left to stream --
        // the phantom row is what used to hold this connection open for the
        // rest of the launch.
        fake.downloadStream?.yield(FakeFixtures.downloadEvent(
            type: "job_done", id: "job-1", model: "hf:black-forest-labs/FLUX.1-dev"))
        await settle { downloads.active[workstation.id] == nil }
        #expect(downloads.streams[workstation.id] == nil)
    }

    /// A delta about a job this client has never seen is not a new job:
    /// `downloads.ts:96-97` returns the state untouched rather than
    /// synthesising a row out of a partial frame.
    @Test func aProgressFrameForAnUnknownJobCreatesNothing() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.downloadTicket = FakeFixtures.downloadTicket("job-1")
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))
        await downloads.install("flux-dev:q4", on: workstation)
        await settle { fake.callCount("downloadEvents") == 1 }

        fake.downloadStream?.yield(FakeFixtures.downloadEvent(
            type: "progress", id: "job-somebody-elses", bytesDone: 5, bytesTotal: 10))
        fake.downloadStream?.yield(FakeFixtures.downloadEvent(
            type: "progress", id: "job-1", bytesDone: 7, bytesTotal: 10))
        await settle { downloads.active[workstation.id]?["job-1"]?.bytesDone == 7 }

        #expect(Set((downloads.active[workstation.id] ?? [:]).keys) == ["job-1"])
    }

    /// `enqueued` still MAY create one -- that is how a `mold pull` at a
    /// terminal appears here between two snapshots.
    @Test func anEnqueuedFrameStillIntroducesAJobThisAppNeverStarted() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.downloadTicket = FakeFixtures.downloadTicket("job-1")
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))
        await downloads.install("flux-dev:q4", on: workstation)
        await settle { fake.callCount("downloadEvents") == 1 }

        fake.downloadStream?.yield(FakeFixtures.downloadEvent(
            type: "enqueued", id: "job-cli", model: "sd15:fp16"))
        await settle { downloads.active[workstation.id]?.count == 2 }

        #expect(downloads.active[workstation.id]?["job-cli"]?.model == "sd15:fp16")
    }
}
