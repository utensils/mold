import Foundation
import Testing

@testable import MoldClient

// The Library's Filter menu: one Photos-style pick (Favourites, Not in a
// Collection, Portrait, ...) that combines with the Model Family and Machine
// submenus, all answered on the client like every other Library token.

private let workstation = UUID()
private let hal = UUID()

private func entry(_ name: String, host: UUID = workstation, format: String = "png",
                   tags: [String]? = nil, favorite: Bool? = nil, collections: [String]? = nil,
                   metadata: [String: Any] = [:]) -> LibraryEntry {
    let data = try! JSONSerialization.data(withJSONObject: metadata)
    let print = GalleryPrint(
        filename: name, metadata: try! MoldJSON.decoder.decode(OutputMetadata.self, from: data),
        timestamp: 1_000, format: format, sizeBytes: 100, mediaVersion: "v1", title: nil, tags: tags,
        favorite: favorite, collections: collections, trashedAt: nil, purgeAt: nil)
    return LibraryEntry(host: MoldHost(id: host, name: "m", baseURL: URL(string: "http://h")!), print: print)
}

private let library = [
    entry("filed.png", collections: ["c1"], metadata: ["family": "flux", "width": 768, "height": 1344]),
    entry("loose.png", tags: ["owls"], metadata: ["family": "sdxl", "width": 1344, "height": 768]),
    entry("square.png", favorite: true, metadata: ["family": "flux", "width": 1024, "height": 1024,
                                                    "source_image_sha256": "abc"]),
    entry("old.mp4", format: "mp4", metadata: ["family": "ltx_video", "width": 768, "height": 512,
                                               "keyframes": [["frame": 0, "sha256": "k"]]]),
    entry("chair.glb", format: "glb", metadata: ["family": "hunyuan3d"]),
]

private func shown(_ tokens: [LibraryToken]) -> [String] {
    var query = LibraryQuery()
    query.tokens = tokens
    return query.apply(to: library).map(\.print.filename).sorted()
}

struct LibraryFilterMenuTests {
    @Test func collectionMembershipAnswersInAndNotInACollection() {
        #expect(shown([.filed(true)]) == ["filed.png"])
        #expect(shown([.filed(false)]) == ["chair.glb", "loose.png", "old.mp4", "square.png"])
    }

    @Test func aMergedPrintIsFiledWhenAnyCopyIs() {
        let local = entry("same.png", metadata: ["seed": 1])
        let remote = entry("same.png", host: hal, collections: ["c9"], metadata: ["seed": 1])
        var merged = local
        merged.copies = [remote]
        var query = LibraryQuery()
        query.tokens = [.filed(false)]
        #expect(query.apply(to: [merged]).isEmpty)
    }

    @Test func madeFromASourceReadsEveryKindOfInput() {
        #expect(shown([.fromSource]) == ["old.mp4", "square.png"])
    }

    @Test func tagsAnswerTaggedAndUntagged() {
        #expect(shown([.tagged(true)]) == ["loose.png"])
        #expect(shown([.tagged(false)]).contains("filed.png"))
        #expect(!shown([.tagged(false)]).contains("loose.png"))
    }

    @Test func orientationComesFromTheDimensionsAndNeverGuessesForAMesh() {
        #expect(shown([.orientation(.portrait)]) == ["filed.png"])
        #expect(shown([.orientation(.landscape)]) == ["loose.png", "old.mp4"])
        #expect(shown([.orientation(.square)]) == ["square.png"])
    }

    @Test func familiesCanonicalizeOldSpellingsAndWidenEachOther() {
        #expect(shown([.family(slug: "ltx-video", name: "LTX Video")]) == ["old.mp4"])
        #expect(shown([.family(slug: "flux", name: "FLUX.1"), .family(slug: "sdxl", name: "SDXL")])
            == ["filed.png", "loose.png", "square.png"])
        // A family narrows the menu's pick, as every different-kind token does.
        #expect(shown([.family(slug: "flux", name: "FLUX.1"), .filed(false)]) == ["square.png"])
    }

    @Test func theMenuIsOnePickThatKeepsEverythingElse() {
        var query = LibraryQuery()
        query.text = "robot"
        query.tokens = [.machine(id: hal, name: "hal"), .family(slug: "flux", name: "FLUX.1"),
                        .favorite, .kind(.clip)]
        #expect(LibraryFilter.selected(in: query) == nil, "two picks are not one menu row")
        let next = LibraryFilter.notInCollection.applying(to: query)
        #expect(next.tokens == [.machine(id: hal, name: "hal"), .family(slug: "flux", name: "FLUX.1"), .filed(false)])
        #expect(next.text == "robot")
        #expect(LibraryFilter.selected(in: next) == .notInCollection)
        #expect(LibraryFilter.all.applying(to: next).tokens == [.machine(id: hal, name: "hal"),
                                                                .family(slug: "flux", name: "FLUX.1")])
        #expect(LibraryFilter.selected(in: LibraryQuery()) == .all)
        for filter in LibraryFilter.allCases {
            #expect(LibraryFilter.selected(in: filter.applying(to: LibraryQuery())) == filter)
        }
    }

    @Test func theFamilySubmenuListsOnlyWhatTheLibraryHolds() {
        let facets = LibraryFamilies.facets(in: library)
        #expect(facets.map(\.slug) == ["flux", "hunyuan3d", "ltx-video", "sdxl"])
        #expect(facets.map(\.label) == ["FLUX.1", "Hunyuan3D", "LTX Video", "SDXL"])
        #expect(facets.first?.count == 2)
        #expect(facets.allSatisfy { $0.color != nil })

        var query = LibraryFilter.favorites.applying(to: LibraryQuery())
        query = LibraryFamilies.toggling("sdxl", label: "SDXL", in: query)
        #expect(LibraryFamilies.selected(in: query) == ["sdxl"])
        query = LibraryFamilies.toggling("sdxl", label: "SDXL", in: query)
        #expect(LibraryFamilies.selected(in: query).isEmpty)
        #expect(query.tokens == [.favorite], "the family submenu never touches the pick")
    }

    @Test func familyColoursComeFromTheGeneratedAuthority() throws {
        let flux = try #require(Model.familyColor(forFamily: "flux"))
        #expect(flux.rgb == 0xC24CF6)
        #expect(abs(flux.red - 0xC2 / 255.0) < 0.0001)
        #expect(Model.familyColor(forFamily: "minimax_h3") == Model.familyColor(forFamily: "minimax-h3"))
        #expect(Model.familyLabel(forFamily: "ltx_video") == "LTX Video")
        #expect(Model.familyColor(forFamily: "companion") == nil)
    }
}

struct LibraryFilterSummaryTests {
    @Test func theFilterButtonReadsWhatIsApplied() {
        var query = LibraryQuery()
        #expect(LibraryFilter.summary(for: query) == "All Items")
        query = LibraryFilter.notInCollection.applying(to: query)
        #expect(LibraryFilter.summary(for: query) == "Not in a Collection")
        query = LibraryFamilies.toggling("flux", label: "FLUX.1", in: query)
        #expect(LibraryFilter.summary(for: query) == "Not in a Collection · FLUX.1")
        query = LibraryFamilies.toggling("sdxl", label: "SDXL", in: query)
        #expect(LibraryFilter.summary(for: query) == "Not in a Collection · 2 Families")
        query = LibraryFilter.all.applying(to: query)
        #expect(LibraryFilter.summary(for: query) == "2 Families")
        query.tokens += [.favorite, .kind(.clip)]
        #expect(LibraryFilter.summary(for: query) == "Multiple Filters · 2 Families")
        #expect(!LibraryFilter.isActive(LibraryQuery()))
        #expect(LibraryFilter.isActive(query))
        #expect(LibraryFilter.clearing(query).tokens.isEmpty)
    }

    @Test @MainActor func theFamilyListIsDerivedOncePerLibraryRevision() {
        let cache = LibraryFamilyFacetCache()
        let pool = [entry("a.png", metadata: ["family": "flux"])]
        #expect(cache.facets(in: pool, revision: 1).map(\.slug) == ["flux"])
        #expect(cache.facets(in: [], revision: 1).map(\.slug) == ["flux"], "same revision reuses the answer")
        #expect(cache.facets(in: [], revision: 1, partition: .trash).isEmpty, "Trash is its own pool")
        #expect(cache.facets(in: [], revision: 2).isEmpty)
        #expect(cache.derivations == 3)
    }

    @Test func infoNamesTheFamilyAndCarriesItsColour() throws {
        let print = entry("old.mp4", format: "mp4", metadata: ["family": "ltx_video", "model": "ltx-video"])
        let model = try #require(PrintDetails.groups(for: print).first { $0.title == "Model" })
        let family = try #require(model.rows.first { $0.label == "Family" })
        #expect(family.value == "LTX Video")
        #expect(family.swatch == 0xEE5A5A)
        #expect(model.rows.first { $0.label == "Model" }?.swatch == nil)
    }
}
