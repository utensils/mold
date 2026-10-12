import Foundation
import Testing
@testable import MoldClient

struct FuzzyModelSearchTests {
    let fields = ["FLUX.1 Dev Q4", "flux-dev:q4", "black-forest-labs/FLUX.1-dev"]

    @Test(arguments: ["FLUX", "dev flux", "flux_dev:q4", "fluxdevq4", "flx dev", "flxu dev", "fluz dev", "fluxx dev", "flx q4", "forest q4"])
    func matches(_ query: String) {
        #expect(FuzzyModelSearch.score(query, fields: fields) > 0)
    }

    @Test(arguments: ["q8", "banana", "flux banana", "dx", "4q"])
    func refuses(_ query: String) {
        #expect(FuzzyModelSearch.score(query, fields: fields) == 0)
    }

    @Test func accentsAndFields() {
        #expect(FuzzyModelSearch.score("cafe video", fields: ["Café", "Video"]) > 0)
        #expect(FuzzyModelSearch.score("猫", fields: ["猫のモデル"]) > 0)
        #expect(FuzzyModelSearch.score(String(repeating: "x", count: 10000), fields: fields) == 0)
    }

    @Test func ranking() {
        let rows = ["fluxx", "flux", "flux"]
        #expect(FuzzyModelSearch.ranked(rows, query: "flux", fields: { [$0] }) == ["flux", "flux", "fluxx"])
        #expect(FuzzyModelSearch.ranked(rows, query: "  ", fields: { [$0] }) == rows)
    }
    @Test func exactHitInLaterFamilyComesFirst() throws {
        let rows = try MoldJSON.decoder.decode([Model].self, from: Data(#"""
        [{"name":"fluxx", "family":"first", "description":"", "size_gb":1, "is_loaded":false, "downloaded":true, "hf_repo":"test/fluxx"},
         {"name":"flux", "family":"later", "description":"", "size_gb":1, "is_loaded":false, "downloaded":true, "hf_repo":"test/flux"}]
        """#.utf8))
        let families = [(family: "first", models: [rows[0]]), (family: "later", models: [rows[1]])]
        #expect(FuzzyModelSearch.groups(families, query: "flux").map(\.family) == ["later", "first"])
        #expect(FuzzyModelSearch.groups(families, query: "").map(\.family) == ["first", "later"])
    }

}
