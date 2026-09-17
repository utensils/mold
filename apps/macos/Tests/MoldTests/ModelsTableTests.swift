import Foundation
import MoldClient
import Testing

@testable import Mold

/// The Installed table's ordering, grouping and footer sentence -- the pure
/// gates `ModelSort` and `ModelsFooter` answer before anything is drawn,
/// tested without a view (design S3, M5).
@MainActor
struct ModelsTableTests {
    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    // MARK: - The Installed scope

    @Test func theInstalledScopeListsEveryFamilyTheMachineHolds() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.modelRows = [
            FakeFixtures.model("flux-dev:q4", family: "flux", downloaded: true),
            FakeFixtures.model("real-esrgan-x4plus:fp16", family: "upscaler", downloaded: true),
            FakeFixtures.model("qwen3-4b:q4", family: "qwen3-expand", downloaded: true),
        ]
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let models = ModelStore(hosts: hosts)
        await models.refresh(on: plato.id)

        let sections = ModelSort.grouped(models.installed(on: plato.id), by: ModelSort())

        #expect(Set(sections.map(\.family)) == ["flux", "upscaler", "qwen3-expand"])
    }

    /// A half-installed row is a MANAGEMENT row (design fact 5, M5): it must
    /// reach the table through the same door as any other installed model,
    /// grouped and sorted like everything else.
    @Test func aHalfInstalledModelIsInTheInstalledScope() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.modelRows = [
            FakeFixtures.model("flux-dev:bf16", family: "flux", downloaded: true, remainingDownloadBytes: 500),
        ]
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let models = ModelStore(hosts: hosts)
        await models.refresh(on: plato.id)

        let sections = ModelSort.grouped(models.installed(on: plato.id), by: ModelSort())

        #expect(sections.flatMap(\.rows).map(\.name) == ["flux-dev:bf16"])
    }

    // MARK: - Sorting

    @Test func sortingBySizeKeepsARowWithNothingOnDiskHonest() {
        let installed = FakeFixtures.model("flux-dev:q4", downloaded: true, diskUsageBytes: 1_000_000)
        let available = FakeFixtures.model("flux-dev:bf16", downloaded: false, remainingDownloadBytes: 500_000)

        let ascending = ModelSort.sorted([installed, available], by: ModelSort(column: .size, ascending: true))
        #expect(ascending.map(\.name) == ["flux-dev:bf16", "flux-dev:q4"])

        let descending = ModelSort.sorted([installed, available], by: ModelSort(column: .size, ascending: false))
        #expect(descending.map(\.name) == ["flux-dev:q4", "flux-dev:bf16"])
    }

    @Test func sortingByStatePutsWhatIsBrokenFirst() {
        let loaded = FakeFixtures.model("a", downloaded: true, isLoaded: true)
        let installed = FakeFixtures.model("b", downloaded: true)
        let broken = FakeFixtures.model("c", downloaded: true, remainingDownloadBytes: 10)
        let available = FakeFixtures.model("d", downloaded: false, remainingDownloadBytes: 999)

        let ordered = ModelSort.sorted([loaded, installed, broken, available],
                                        by: ModelSort(column: .state, ascending: true))

        #expect(ordered.map(\.name) == ["c", "d", "b", "a"])
    }

    /// Plain alphabetical order on the full headline would put "FLUX.1 Dev L"
    /// -- an unrelated model whose name happens to fall between "BF16" and
    /// "Q4" -- between the two FLUX.1 Dev variants. `sortHeadline`'s
    /// baseTitle-first key keeps them together instead (design S3, M5).
    @Test func variantsOfOneModelStayTogetherWhenSortedByName() {
        let bf16 = FakeFixtures.model("flux-dev:bf16", downloaded: true, description: "FLUX.1 Dev BF16 — best quality")
        let q4 = FakeFixtures.model("flux-dev:q4", downloaded: true, description: "FLUX.1 Dev Q4 — fast")
        let other = FakeFixtures.model("flux-devl:fp16", downloaded: true, description: "FLUX.1 Dev L — an unrelated model")

        let ordered = ModelSort.sorted([q4, other, bf16], by: ModelSort(column: .model, ascending: true))

        #expect(ordered.map(\.name) == ["flux-dev:bf16", "flux-dev:q4", "flux-devl:fp16"])
    }

    @Test func sectionsCarryTheServersOwnFamilyName() {
        let rows = [FakeFixtures.model("ltx2-13b:bf16", family: "ltx2", downloaded: true)]

        let sections = ModelSort.grouped(rows, by: ModelSort())

        // Not "LTX-2": rewriting the server's own family string is a client
        // lexicon for a server id, which M4 already refused for recipe
        // labels (decision 9, M5).
        #expect(sections.map(\.family) == ["ltx2"])
    }

    // MARK: - The footer

    /// The column sum here (23 GB + 18 GB = 41 GB) deliberately exceeds the
    /// disk's own total (30 GB) -- a shared VAE or encoder counted once per
    /// model that references it (design fact 3, M5, pinned).
    @Test func theFooterNeverAddsTheSizeColumnUp() {
        let status = FakeFixtures.serverStatus(modelsDiskTotal: 30_000_000_000, modelsDiskFree: 10_000_000_000)

        let sentence = ModelsFooter.sentence(count: 2, hostName: "plato", disk: status.modelsDisk)

        let expectedUsed = Int64(20_000_000_000).formatted(.byteCount(style: .file))
        let expectedTotal = Int64(30_000_000_000).formatted(.byteCount(style: .file))
        #expect(sentence == "2 installed on plato · \(expectedUsed) of \(expectedTotal) used")
    }

    /// An older host that predates `models_disk` gets the count alone -- never
    /// a fabricated zero.
    @Test func aMachineWithNoDiskFigureHasNoSecondClause() {
        let sentence = ModelsFooter.sentence(count: 5, hostName: "hal9000", disk: nil)
        #expect(sentence == "5 installed on hal9000")
    }
}
