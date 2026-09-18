import Foundation
import MoldClient
import Testing

@testable import Mold

/// Per-machine consent for the built-in gated licences. `refresh(on:)` is
/// guarded on `capabilities.hasLicenses`, and acceptance never fans out
/// across the fleet.
@MainActor
struct LicenseStoreTests {
    private func machine(_ name: String = "workstation") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func licence(accepted: Bool = false) -> ThirdPartyLicense {
        ThirdPartyLicense(
            id: "tencent-hunyuan3d-2.1", name: "Tencent Hunyuan3D 2.1", url: "https://example/LICENSE",
            canonical: "https://example/page", sha256: "abc123", summary: "Non-commercial research only.",
            accepted: accepted, requiredBy: ["hunyuan3d-2.1:fp16"])
    }

    /// **Fails today**: `LicenseStore` does not exist.
    @Test func aMachineThatGatesNothingHoldsNoLicences() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.licenseRows = [licence()]
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        // Never told `hosts.capabilities[workstation.id]` anything -- the same
        // "never said" state as an older host with no `licenses` key.
        let licenses = LicenseStore(hosts: hosts)

        await licenses.refresh(on: workstation.id)

        #expect(licenses.byHost[workstation.id] == nil)
        #expect(fake.calls.isEmpty)
    }

    @Test func aMachineThatGatesReadsAndHoldsItsRows() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.licenseRows = [licence()]
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        hosts.capabilities[workstation.id] = FakeFixtures.capabilities(licenses: true)
        let licenses = LicenseStore(hosts: hosts)

        await licenses.refresh(on: workstation.id)

        #expect(licenses.licence(gating: "hunyuan3d-2.1:fp16", on: workstation.id)?.id == "tencent-hunyuan3d-2.1")
        #expect(licenses.isAccepted("tencent-hunyuan3d-2.1", on: workstation.id) == false)
    }

    @Test func acceptingRecordsConsentAndStoresTheRefreshedListing() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.licenseRows = [licence()]
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        hosts.capabilities[workstation.id] = FakeFixtures.capabilities(licenses: true)
        let licenses = LicenseStore(hosts: hosts)
        await licenses.refresh(on: workstation.id)

        let ok = await licenses.accept(licence(), on: workstation.id)

        #expect(ok)
        #expect(licenses.isAccepted("tencent-hunyuan3d-2.1", on: workstation.id))
        #expect(fake.acceptedLicenses.last?.map(\.id) == ["tencent-hunyuan3d-2.1"])
    }

    @Test func acceptingByRefusalRecordsTheSameConsent() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.licenseRows = [licence()]
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        hosts.capabilities[workstation.id] = FakeFixtures.capabilities(licenses: true)
        let licenses = LicenseStore(hosts: hosts)
        let refusal = LicenseRefusal(
            id: "tencent-hunyuan3d-2.1", name: "Tencent Hunyuan3D 2.1", url: "https://example/LICENSE",
            canonical: "https://example/page", sha256: "abc123", summary: "Non-commercial research only.")

        let ok = await licenses.accept(refusal, on: workstation.id)

        #expect(ok)
        #expect(fake.acceptedLicenses.last?.map(\.id) == ["tencent-hunyuan3d-2.1"])
    }
}
