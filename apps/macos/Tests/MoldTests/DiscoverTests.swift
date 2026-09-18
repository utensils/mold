import Foundation
import MoldClient
import Testing

@testable import Mold

/// Browsing a machine's catalog: which scope a host offers, `CatalogStore`'s
/// debounced search and pagination, and the pure rules `DiscoverTable` and
/// `CatalogDetailSheet` render from (design M5 S6).
@MainActor
struct DiscoverTests {
    private func machine(_ name: String = "workstation") -> MoldHost {
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
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        var query = CatalogQuery(includeNSFW: false)
        query.text = "dreamshaper"
        fake.catalogPages[query.queryString] = FakeFixtures.catalogListing([FakeFixtures.catalogEntry(id: "cv:1")])
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let catalog = CatalogStore(hosts: hosts)

        catalog.setText("dreamshaper", on: workstation.id)
        try? await Task.sleep(for: .milliseconds(400))
        await settle { catalog.entries(on: workstation.id).count == 1 }

        #expect(catalog.entries(on: workstation.id).map(\.id) == ["cv:1"])
        #expect(fake.calls.filter { $0 == "searchCatalog" }.count == 1)
    }

    @Test func aSecondPageAppends() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        let page1 = CatalogQuery(includeNSFW: false)
        fake.catalogPages[page1.queryString] = FakeFixtures.catalogListing(
            [FakeFixtures.catalogEntry(id: "cv:1")], page: 1, total: 2)
        var page2 = page1
        page2.page = 2
        fake.catalogPages[page2.queryString] = FakeFixtures.catalogListing(
            [FakeFixtures.catalogEntry(id: "cv:2")], page: 2, total: 2)
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let catalog = CatalogStore(hosts: hosts)

        catalog.search(on: workstation.id)
        try? await Task.sleep(for: .milliseconds(400))
        await settle { catalog.entries(on: workstation.id).count == 1 }
        await catalog.more(on: workstation.id)

        #expect(catalog.entries(on: workstation.id).map(\.id) == ["cv:1", "cv:2"])
        #expect(catalog.hasMore(on: workstation.id) == false)
    }

    /// One provider being down beside rows the other returned is a PARTIAL
    /// SUCCESS -- a note the pane shows, never a `HostFailure`.
    @Test func oneProviderFailingIsANoteAboveTheRowsItDidGet() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        let query = CatalogQuery(includeNSFW: false)
        fake.catalogPages[query.queryString] = FakeFixtures.catalogListing(
            [FakeFixtures.catalogEntry(id: "cv:1")],
            providerErrors: [(source: "civitai", message: "timed out")])
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let catalog = CatalogStore(hosts: hosts)

        catalog.search(on: workstation.id)
        try? await Task.sleep(for: .milliseconds(400))
        await settle { !catalog.entries(on: workstation.id).isEmpty }

        #expect(catalog.providerErrors(on: workstation.id).map(\.source) == ["civitai"])
        #expect(hosts.failures.isEmpty)
        #expect(DiscoverTable.providerNote(catalog.providerErrors(on: workstation.id)) == "Civitai didn't answer.")
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

    // MARK: - The contextual menu

    /// **Fails today**: a Discover row has no contextual menu at all, and its
    /// details open on a double-click and nothing else -- unreachable from a
    /// right click or the keyboard. The menu carries exactly what the row's
    /// own cells offer, in the row's own reading order.
    @Test func aDiscoverRowsMenuCarriesTheSameActionsItsCellsDo() throws {
        let installable = FakeFixtures.catalogEntry(id: "cv:1", supported: true, installed: false)
        #expect(DiscoverRow.menuItems(for: installable).map(\.title) == ["Details…", "Install"])

        let unsupported = FakeFixtures.catalogEntry(
            id: "hf:x", supported: false, pageUrl: "https://example.com/x")
        #expect(DiscoverRow.menuItems(for: unsupported).map(\.title) == ["Details…", "Open Page"])
        let url = try #require(URL(string: "https://example.com/x"))
        #expect(DiscoverRow.menuItems(for: unsupported).last?.kind == .openPage(url))

        // No page to open is no item, not an item that goes nowhere.
        let nowhere = FakeFixtures.catalogEntry(id: "hf:y", supported: false)
        #expect(DiscoverRow.menuItems(for: nowhere).map(\.title) == ["Details…"])

        // An installed row is managed from the Installed table, which has the
        // whole install/load/delete menu -- the State column offers nothing
        // here either.
        let installed = FakeFixtures.catalogEntry(id: "cv:2", installed: true)
        #expect(DiscoverRow.menuItems(for: installed).map(\.title) == ["Details…"])
    }

    // MARK: - Licence metadata

    /// All-null is the ORDINARY case (measured on workstation) and means no
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

    /// The machine's own lists, never a client guess -- workstation's measured
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

    // MARK: - S6b: the subtitle/footer said "0 installed" while Discover's
    // search field held text that matched none of the INSTALLED rows.

    @Test func theInstalledSubtitleUsesTheMachinesUnfilteredTotal() {
        #expect(ModelsPane.subtitle(scope: .installed, hostName: "workstation", installedCount: 82, discoverTotal: nil)
            == "82 installed on workstation")
    }

    @Test func theDiscoverSubtitleSaysNothingBeforeASearchAnswers() {
        #expect(ModelsPane.subtitle(scope: .discover, hostName: "workstation", installedCount: 82, discoverTotal: nil) == "")
    }

    @Test func theDiscoverSubtitleReportsTheSearchsOwnTotalNeverTheInstalledCount() {
        #expect(ModelsPane.subtitle(scope: .discover, hostName: "workstation", installedCount: 82, discoverTotal: 32)
            == "32 results on workstation")
        #expect(ModelsPane.subtitle(scope: .discover, hostName: "workstation", installedCount: 82, discoverTotal: 1)
            == "1 result on workstation")
    }

    @Test func noMachineIsSaidRegardlessOfScope() {
        #expect(ModelsPane.subtitle(scope: .installed, hostName: nil, installedCount: 0, discoverTotal: nil) == "No machine")
        #expect(ModelsPane.subtitle(scope: .discover, hostName: nil, installedCount: 0, discoverTotal: 5) == "No machine")
    }

    @Test func aHostHasNotAnsweredUntilItsFirstSearchLands() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.catalogPages[CatalogQuery(includeNSFW: false).queryString] = FakeFixtures.catalogListing([])
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let catalog = CatalogStore(hosts: hosts)

        #expect(catalog.hasAnswered(on: workstation.id) == false)

        catalog.search(on: workstation.id)
        try? await Task.sleep(for: .milliseconds(400))
        await settle { catalog.hasAnswered(on: workstation.id) }

        #expect(catalog.hasAnswered(on: workstation.id))
    }

    // MARK: - S6b: the Sort picker drew with nothing selected.

    @Test func adoptingAHostSeedsSortFromItsFirstAdvertisedOption() {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let catalog = CatalogStore(hosts: hosts)

        catalog.adopt(workstation.id, sortOptions: ["downloads", "recent", "rating"])

        #expect(catalog.query(on: workstation.id).sort == "downloads")
    }

    @Test func adoptingAHostWithNoAdvertisedSortsLeavesTheQueryUnsorted() {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let catalog = CatalogStore(hosts: hosts)

        catalog.adopt(workstation.id, sortOptions: [])

        #expect(catalog.query(on: workstation.id).sort == nil)
    }

    @Test func adoptingAHostNeverOverwritesASortAlreadyChosen() {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let catalog = CatalogStore(hosts: hosts)
        catalog.setSort("rating", on: workstation.id)

        catalog.adopt(workstation.id, sortOptions: ["downloads", "recent", "rating"])

        #expect(catalog.query(on: workstation.id).sort == "rating")
    }

    // MARK: - S6b: two blank rows appeared above Load more on a live capture.

    @Test func withNothingMoreToLoadTheRowsFunctionReturnsExactlyTheEntries() {
        let entries = [FakeFixtures.catalogEntry(id: "cv:1"), FakeFixtures.catalogEntry(id: "cv:2")]
        #expect(DiscoverTable.rows(for: entries).count == entries.count)
        #expect(DiscoverTable.rows(for: entries).map(\.id) == ["cv:1", "cv:2"])
    }

    @Test func duplicateEntriesNeverProduceMoreThanOneRowEach() {
        let entries = [FakeFixtures.catalogEntry(id: "cv:1"), FakeFixtures.catalogEntry(id: "cv:1")]
        #expect(DiscoverTable.rows(for: entries).count == 1)
    }

    @Test func noEntriesIsNoRows() {
        #expect(DiscoverTable.rows(for: []).isEmpty)
    }
}

/// **Fails today**: a catalog row's page link is whatever string the MACHINE
/// sent, and it went straight to `NSWorkspace.open`. A hostile or compromised
/// machine could hand this Mac `file:///…`, `ssh://…` or another app's
/// deep-link scheme and have it opened on a click labelled "Open Page". A page
/// is a web page: anything else is no page, so there is no item and no cell.
@MainActor
struct DiscoverPageLinkTests {
    @Test func onlyAWebPageIsAPage() {
        for hostile in ["file:///etc/passwd", "ssh://evil.example", "x-apple.systempreferences:",
                        "javascript:alert(1)", "https://", "not a url"] {
            let entry = FakeFixtures.catalogEntry(id: "hf:x", supported: false, pageUrl: hostile)
            #expect(DiscoverRow.menuItems(for: entry).map(\.title) == ["Details…"], "\(hostile)")
        }
        for page in ["https://huggingface.co/x/y", "http://example.com/page"] {
            let entry = FakeFixtures.catalogEntry(id: "hf:x", supported: false, pageUrl: page)
            #expect(DiscoverRow.menuItems(for: entry).map(\.title) == ["Details…", "Open Page"], "\(page)")
        }
    }
}
