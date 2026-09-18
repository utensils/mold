import Foundation
import MoldClient
import Testing

@testable import Mold

/// What the sheet says about a held licence, and what "Accept" and "Cancel"
/// actually do -- the same `DownloadStore.accepted(_:)` call the button
/// makes, and the sheet's own pure text for the caption and the mismatch
/// line. `LicenseSheet` renders from `LicenseRefusal` alone -- the 403
/// payload -- which carries no `requiredBy`/`requiredByStyles` (those ride
/// only on `ThirdPartyLicense`, `GET /api/licenses`), so unlike the design
/// draft this sheet never lists gated model names; that list belongs to a
/// later "Show Licence…" surface reading the full listing, out of scope here.
@MainActor
struct LicenseSheetTests {
    private func machine(_ name: String = "workstation") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func refusal() -> LicenseRefusal {
        LicenseRefusal(
            id: "tencent-hunyuan3d-2.1", name: "Tencent Hunyuan3D 2.1", url: "https://example/LICENSE",
            canonical: "https://example/page", sha256: "abc123def456789", summary: "Non-commercial research only.")
    }

    // MARK: - Pure text

    /// **Fails today**: `LicenseSheet` does not exist yet.
    @Test func theCaptionNamesThePinnedURLAndAShaPrefix() {
        let text = LicenseSheet.caption(for: refusal())
        #expect(text.contains("https://example/LICENSE"))
        #expect(text.contains("abc123def456"))
    }

    @Test func noMismatchSentenceOnAnOrdinaryRefusal() {
        let pending = DownloadStore.PendingLicense(refusal: refusal(), mismatch: false, host: UUID()) {}
        #expect(LicenseSheet.mismatchSentence(for: pending) == nil)
    }

    @Test func aTermsMismatchSaysSoAndAcceptsTheSameWay() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.plantedErrors["startDownload"] = MoldClientError.licenseRequired(refusal(), mismatch: true)
        fake.downloadTicket = FakeFixtures.downloadTicket("job-1")
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))
        await downloads.install("hunyuan3d-2.1:fp16", on: workstation)
        guard let pending = downloads.pendingLicense else {
            Issue.record("expected a pending licence")
            return
        }

        // The sheet says so up front...
        #expect(LicenseSheet.mismatchSentence(for: pending)?.isEmpty == false)

        // ...and Accept still sends exactly what was shown, same as an
        // ordinary refusal -- a mismatch changes the WARNING, never the
        // acceptance itself.
        fake.plantedErrors["startDownload"] = nil
        await downloads.accepted(pending)

        #expect(downloads.pendingLicense == nil)
        #expect(fake.acceptedLicenses.last == [refusal().acceptance])
        #expect(fake.callCount("startDownload") == 2)
    }

    // MARK: - Accept sends the terms shown

    /// **Fails today** alongside the rest of this file, and duplicates
    /// `DownloadStoreTests.acceptingAndRetryingStartsTheSameDownload` on
    /// purpose: that test proves the STORE's contract; this one proves the
    /// sheet's Accept button has nothing else to do but call it -- the exact
    /// `(id, url, sha256)` the sheet displayed is what `acceptance` sends,
    /// with no separate copy for the button to drift from.
    @Test func acceptingSendsTheTermsThatWereShown() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.plantedErrors["startDownload"] = MoldClientError.licenseRequired(refusal(), mismatch: false)
        fake.downloadTicket = FakeFixtures.downloadTicket("job-1")
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))
        await downloads.install("hunyuan3d-2.1:fp16", on: workstation)
        guard let pending = downloads.pendingLicense else {
            Issue.record("expected a pending licence")
            return
        }
        let shownCaption = LicenseSheet.caption(for: pending.refusal)

        fake.plantedErrors["startDownload"] = nil
        await downloads.accepted(pending)

        let sent = fake.acceptedLicenses.last?.first
        #expect(sent?.id == refusal().id)
        #expect(sent?.url == refusal().url)
        #expect(sent?.sha256 == refusal().sha256)
        // The url this actually sent is the same one the caption named.
        if let sent {
            #expect(shownCaption.contains(sent.url))
        } else {
            Issue.record("expected an acceptance to have been sent")
        }
    }

    /// Cancel is not this file's to prove beyond the one fact that matters:
    /// `pendingLicense` is `internal(set)`, so the sheet's Cancel button can
    /// clear it directly -- no method needed on the store for this alone.
    @Test func cancelClearsThePendingLicenceWithoutCallingTheStore() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.plantedErrors["startDownload"] = MoldClientError.licenseRequired(refusal(), mismatch: false)
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))
        await downloads.install("hunyuan3d-2.1:fp16", on: workstation)
        #expect(downloads.pendingLicense != nil)

        downloads.pendingLicense = nil

        #expect(downloads.pendingLicense == nil)
        #expect(fake.callCount("acceptLicenses") == 0)
    }
}
