import MoldClient
import Testing
@testable import Mold

@MainActor
struct NativeDiscoveryDetailTests {
    @Test func featuredIsTheUnfilteredStartingPoint() {
        #expect(DiscoverLanding.showsFeatured(text: "  ", family: nil, browsingCatalog: false))
        #expect(!DiscoverLanding.showsFeatured(text: "flux", family: nil, browsingCatalog: false))
        #expect(!DiscoverLanding.showsFeatured(text: "", family: "flux", browsingCatalog: false))
        #expect(!DiscoverLanding.showsFeatured(text: "", family: nil, browsingCatalog: true))
    }

    @Test func featuredUsesManifestGeneratorsIncludingInstalledModels() {
        let models = [FakeFixtures.model("flux", downloaded: true),
                      FakeFixtures.model("z-image"),
                      FakeFixtures.model("hf:external"),
                      FakeFixtures.model("vae", family: "controlnet")]
        #expect(DiscoverLanding.featured(models).map(\.name) == ["flux", "z-image"])
    }

    @Test func detailsFollowOnlyTheOwningMachinesLiveRow() {
        let original = FakeFixtures.queueEntry("job", state: "queued")
        let running = FakeFixtures.queueEntry("job", state: "running")
        #expect(QueueDetailSheet.current(original, in: [running])?.state == .running)
        #expect(QueueDetailSheet.current(original, in: []) == nil)
        #expect(QueueDetailSheet.current(original, in: [FakeFixtures.queueEntry("other")]) == nil)
    }
}
