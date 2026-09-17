import Foundation
import MoldClient
import Testing

@testable import Mold

/// Browsing a machine's catalog: which scope a host offers, `CatalogStore`'s
/// debounced search and pagination, and the pure rules `DiscoverTable` and
/// `CatalogDetailSheet` render from (design M5 S6).
@MainActor
struct DiscoverTests {
    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    // MARK: - Scope

    /// **Fails today**: `ModelScope` does not exist.
    @Test func aMachineThatCannotBrowseHasNoDiscoverScope() {
        #expect(ModelScope.available(capabilities: nil) == [.installed])
        #expect(ModelScope.available(capabilities: FakeFixtures.capabilities(catalog: false)) == [.installed])
    }

    @Test func aMachineThatCanBrowseOffersBothScopes() {
        #expect(ModelScope.available(capabilities: FakeFixtures.capabilities(catalog: true)) == [.installed, .discover])
    }

    @Test func aStoredDiscoverScopeFallsBackToInstalledOnAMachineThatCannotBrowse() {
        #expect(ModelScope.resolved(stored: .discover, available: [.installed]) == .installed)
        #expect(ModelScope.resolved(stored: .discover, available: [.installed, .discover]) == .discover)
    }

    // MARK: - Search

    /// **Fails today**: `CatalogStore` does not exist.
    @Test func aSearchAsksTheFakeWithTheExactQueryString() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        var query = CatalogQuery(includeNSFW: false)
        query.text = "dreamshaper"
        fake.catalogPages[query.queryString] = FakeFixtures.catalogListing([FakeFixtures.catalogEntry(id: "cv:1")])
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let catalog = CatalogStore(hosts: hosts)

        catalog.setText("dreamshaper", on: plato.id)
        try? await Task.sleep(for: .milliseconds(400))
        await settle { catalog.entries(on: plato.id).count == 1 }

        #expect(catalog.entries(on: plato.id).map(\.id) == ["cv:1"])
        #expect(fake.calls.filter { $0 == "searchCatalog" }.count == 1)
    }

    @Test func aSecondPageAppends() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        let page1 = CatalogQuery(includeNSFW: false)
        fake.catalogPages[page1.queryString] = FakeFixtures.catalogListing(
            [FakeFixtures.catalogEntry(id: "cv:1")], page: 1, total: 2)
        var page2 = page1
        page2.page = 2
        fake.catalogPages[page2.queryString] = FakeFixtures.catalogListing(
            [FakeFixtures.catalogEntry(id: "cv:2")], page: 2, total: 2)
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let catalog = CatalogStore(hosts: hosts)

        catalog.search(on: plato.id)
        try? await Task.sleep(for: .milliseconds(400))
        await settle { catalog.entries(on: plato.id).count == 1 }
        await catalog.more(on: plato.id)

        #expect(catalog.entries(on: plato.id).map(\.id) == ["cv:1", "cv:2"])
        #expect(catalog.hasMore(on: plato.id) == false)
    }

    /// One provider being down beside rows the other returned is a PARTIAL
    /// SUCCESS -- a note the pane shows, never a `HostFailure`.
    @Test func oneProviderFailingIsANoteAboveTheRowsItDidGet() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        let query = CatalogQuery(includeNSFW: false)
        fake.catalogPages[query.queryString] = FakeFixtures.catalogListing(
            [FakeFixtures.catalogEntry(id: "cv:1")],
            providerErrors: [(source: "civitai", message: "timed out")])
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let catalog = CatalogStore(hosts: hosts)

        catalog.search(on: plato.id)
        try? await Task.sleep(for: .milliseconds(400))
        await settle { !catalog.entries(on: plato.id).isEmpty }

        #expect(catalog.providerErrors(on: plato.id).map(\.source) == ["civitai"])
        #expect(hosts.failures.isEmpty)
        #expect(DiscoverTable.providerNote(catalog.providerErrors(on: plato.id)) == "Civitai didn't answer.")
    }

    @Test func noProviderErrorsMeansNoNote() {
        #expect(DiscoverTable.providerNote([]) == nil)
    }

    // MARK: - Row resolution

    @Test func anUnsupportedRowOffersNoInstall() {
        let entry = FakeFixtures.catalogEntry(id: "hf:x", supported: false, pageUrl: "https://example.com/x")
        #expect(DiscoverRow.resolve(entry) == .unsupported(pageURL: URL(string: "https://example.com/x")))
    }

    /// The catalog row's OWN flag is the answer for a catalog id -- there is
    /// no `Model` listing to cross-reference against here.
    @Test func anInstalledRowResolvesToInstalledWithoutAskingTheModelsListing() {
        let entry = FakeFixtures.catalogEntry(id: "cv:1", supported: false, installed: true)
        #expect(DiscoverRow.resolve(entry) == .installed)
    }

    @Test func aSupportedUninstalledRowOffersInstall() {
        let entry = FakeFixtures.catalogEntry(id: "cv:1", supported: true, installed: false)
        #expect(DiscoverRow.resolve(entry) == .install)
    }

    // MARK: - Licence metadata

    /// All-null is the ORDINARY case (measured on plato) and means no
    /// information -- never a fabricated "no" (decision 18, M5).
    @Test func anAllNullLicenceBlockRendersNothing() {
        let entry = FakeFixtures.catalogEntry(id: "cv:1")
        #expect(CatalogDetailSheet.showsLicence(entry) == false)
        #expect(CatalogDetailSheet.licenceRows(entry.licenseFlags).isEmpty)
    }

    @Test func aPopulatedLicenceBlockRendersTriStateRows() {
        let entry = FakeFixtures.catalogEntry(id: "cv:1", commercial: true, derivatives: false)
        #expect(CatalogDetailSheet.showsLicence(entry))
        let rows = CatalogDetailSheet.licenceRows(entry.licenseFlags)
        #expect(rows.map(\.value) == ["Yes", "No", "Unknown"])
    }

    @Test func aBareLicenceStringWithNoFlagsStillShowsTheSection() {
        let entry = FakeFixtures.catalogEntry(id: "cv:1", license: "CreativeML OpenRAIL-M")
        #expect(CatalogDetailSheet.showsLicence(entry))
        #expect(CatalogDetailSheet.licenceRows(entry.licenseFlags).isEmpty)
    }

    // MARK: - Family and sort menus

    /// The machine's own lists, never a client guess -- plato's measured
    /// fourteen families and three sorts (design fact, M5).
    @Test func theFamilyAndSortMenusComeFromTheMachine() {
        let families = [
            "flux", "flux2", "sd15", "sdxl", "sd3", "z-image", "ltx-video", "ltx2",
            "wan", "minimax-h3", "qwen-image", "qwen-image-edit", "wuerstchen", "hunyuan3d",
        ]
        let caps = FakeFixtures.capabilities(catalog: true, families: families, sort: ["downloads", "recent", "rating"])
        #expect(caps.catalogFamilies.count == 14)
        #expect(caps.catalogSortOptions == ["downloads", "recent", "rating"])
    }

    // MARK: - NSFW

    @Test func anUnknownNsfwIsNotDrawnAsSafe() {
        #expect(DiscoverTable.nsfwBadge(FakeFixtures.catalogEntry(id: "cv:1", nsfw: false)) == nil)
        #expect(DiscoverTable.nsfwBadge(FakeFixtures.catalogEntry(id: "cv:1", nsfw: true)) == "NSFW")
    }
}
