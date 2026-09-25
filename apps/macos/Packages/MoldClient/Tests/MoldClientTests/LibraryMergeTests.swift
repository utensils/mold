import Foundation
import Testing

@testable import MoldClient

/// One tile per print, however many machines hold it -- the desktop app's
/// `mergeBuckets` rule (`desktop/src/stores/gallery.test.ts`, "merged grid"
/// and "identity dedupe"), ported case by case.
struct LibraryMergeTests {
    let local = UUID()
    let workstation = UUID()
    let hal = UUID()

    private func entry(_ filename: String, on host: UUID, name: String, timestamp: UInt64 = 1_000,
                       bytes: Int? = 100, seed: UInt64? = nil, model: String? = nil,
                       synthetic: Bool = false) -> LibraryEntry {
        var print = GalleryPrint(
            filename: filename, metadata: PrintFixtures.metadata(model: model, seed: seed),
            timestamp: timestamp, format: "png", sizeBytes: bytes, mediaVersion: "v1",
            title: nil, tags: nil, favorite: nil, collections: nil, trashedAt: nil, purgeAt: nil)
        if synthetic { print.metadataSynthetic = true }
        return LibraryEntry(host: MoldHost(id: host, name: name, baseURL: URL(string: "http://h")!),
                            print: print)
    }

    @Test func aSavedCopyIsOneTileLedByThisMac() {
        let merged = LibraryMerge.merge([
            entry("cat.png", on: workstation, name: "workstation"),
            entry("cat.png", on: local, name: "This Mac"),
            entry("cat.png", on: hal, name: "hal9000"),
        ], localHost: local)

        #expect(merged.count == 1)
        #expect(merged[0].hostID == local, "the local copy leads whatever order it arrived in")
        #expect(merged[0].hostNames == ["This Mac", "workstation", "hal9000"])
        #expect(merged[0].hostBadge(compact: false) == "This Mac · workstation · hal9000")
        #expect(merged[0].hostBadge(compact: true) == "This Mac +2")
    }

    @Test func copiesAcrossRemoteMachinesMergeWithoutALocalCopy() {
        let merged = LibraryMerge.merge([
            entry("cat.png", on: workstation, name: "workstation"),
            entry("cat.png", on: hal, name: "hal9000"),
        ], localHost: local)
        #expect(merged.count == 1)
        #expect(merged[0].hostID == workstation, "first in machine order leads")
    }

    @Test func differentPrintsStayApart() {
        let merged = LibraryMerge.merge([
            entry("cat.png", on: workstation, name: "workstation"),
            entry("dog.png", on: local, name: "This Mac"),
        ], localHost: local)
        #expect(merged.count == 2)
        let noneMerged = merged.allSatisfy { $0.copies.isEmpty }
        #expect(noneMerged)
    }

    /// A copy the app made under a collision-renamed filename joins its
    /// source through the sync record, exactly.
    @Test func aSyncLinkJoinsARenamedCopy() {
        let source = entry("cat.png", on: workstation, name: "workstation", seed: 1)
        let copy = entry("cat~ws-ab12.png", on: local, name: "This Mac", timestamp: 999_999)
        let links = [copy.id: source.id, source.id: copy.id]

        let merged = LibraryMerge.merge([source, copy], localHost: local, links: links)

        #expect(merged.count == 1)
        #expect(merged[0].id == copy.id)
    }

    @Test func seedSizeAndModelJoinCopiesWhoseNamesDiverged() {
        let merged = LibraryMerge.merge([
            entry("a.png", on: workstation, name: "workstation", timestamp: 1_000, bytes: 4_096,
                  seed: 42, model: "flux-dev:q8"),
            entry("b.png", on: local, name: "This Mac", timestamp: 1_030, bytes: 4_096,
                  seed: 42, model: "Flux Dev Q8"),
        ], localHost: local)
        #expect(merged.count == 1)
    }

    /// Two files on ONE machine are two prints, however alike: the machine
    /// itself lists them separately, and a merge would hide one behind the
    /// other (and trash both with one press).
    @Test func twoFilesOnOneMachineNeverMerge() {
        let merged = LibraryMerge.merge([
            entry("a.png", on: hal, name: "hal9000", timestamp: 1_000, bytes: 4_096,
                  seed: 42, model: "flux-dev:q8"),
            entry("b.png", on: hal, name: "hal9000", timestamp: 1_010, bytes: 4_096,
                  seed: 42, model: "flux-dev:q8"),
            entry("c.png", on: local, name: "This Mac", timestamp: 1_020, bytes: 4_096,
                  seed: 42, model: "flux-dev:q8"),
        ], localHost: local)
        #expect(merged.count == 2)
        #expect(merged.allSatisfy { Set($0.everyCopy.map(\.hostID)).count == $0.everyCopy.count })
    }

    /// A genuine re-render reusing a seed, much later, is a different print.
    @Test func identityMatchesOnlyCountWithinTheWindow() {
        let merged = LibraryMerge.merge([
            entry("a.png", on: workstation, name: "workstation", timestamp: 1_000, bytes: 4_096,
                  seed: 42, model: "flux-dev:q8"),
            entry("b.png", on: local, name: "This Mac", timestamp: 1_000 + 7_200, bytes: 4_096,
                  seed: 42, model: "flux-dev:q8"),
        ], localHost: local)
        #expect(merged.count == 2)
    }

    @Test func rowsWithoutSeedOrSizeNeverMatchByIdentity() {
        #expect(LibraryMerge.identity(of: entry("a.png", on: local, name: "x", seed: nil).print) == nil)
        #expect(LibraryMerge.identity(of: entry("a.png", on: local, name: "x", bytes: nil,
                                                seed: 1).print) == nil)
    }

    /// A synthesized video row recorded seed 0 as "unknown"; the auto-save
    /// filename carries the real one.
    @Test func aSyntheticRowTakesItsSeedFromTheFilename() {
        let row = entry("mold-ltx-2-123456-1700000000.mp4", on: local, name: "x", seed: 0,
                        synthetic: true)
        #expect(LibraryMerge.seed(of: row.print) == 123_456)
        let unparsable = entry("clip.mp4", on: local, name: "x", seed: 0, synthetic: true)
        #expect(LibraryMerge.identity(of: unparsable.print) == nil)
    }

    /// Filtering to a machine is asking what is on it: the merged tile is
    /// presented as THAT machine's copy.
    @Test func aMachineFilterShowsThatMachinesCopy() {
        let merged = LibraryMerge.merge([
            entry("cat.png", on: workstation, name: "workstation"),
            entry("cat.png", on: local, name: "This Mac"),
        ], localHost: local)
        var query = LibraryQuery()
        query.tokens = [.machine(id: workstation, name: "workstation")]

        let shown = query.apply(to: merged)

        #expect(shown.map(\.hostID) == [workstation])
        #expect(shown[0].copies.map(\.hostID) == [local])
    }

    @Test func aTagOnAnyCopyFindsThePrint() {
        var tagged = entry("cat.png", on: workstation, name: "workstation")
        let print = GalleryPrint(
            filename: "cat.png", metadata: PrintFixtures.metadata(), timestamp: 1_000,
            format: "png", sizeBytes: 100, mediaVersion: "v1", title: nil, tags: ["pets"],
            favorite: nil, collections: nil, trashedAt: nil, purgeAt: nil)
        tagged = tagged.replacingPrint(print)
        let merged = LibraryMerge.merge([tagged, entry("cat.png", on: local, name: "This Mac")],
                                        localHost: local)
        var query = LibraryQuery()
        query.tokens = [.tag("pets")]
        #expect(query.apply(to: merged).count == 1)
    }
}
