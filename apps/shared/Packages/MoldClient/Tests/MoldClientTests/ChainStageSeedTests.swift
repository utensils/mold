import Foundation
import Testing
@testable import MoldClient

struct ChainStageSeedTests {
    @Test func stageSeedAcceptsDeployedStringsAndNumericFullWidth() throws {
        for wire in ["\"18446744073709551615\"", "18446744073709551615", "\"5961655008711492417\"", "null"] {
            let data = Data("{\"idx\":0,\"state\":\"pending\",\"seed\":\(wire)}".utf8)
            let stage = try MoldJSON.decoder.decode(ChainJobStageDetail.self, from: data)
            let expected: UInt64? = wire == "null" ? nil : wire.contains("5961655") ? 5961655008711492417 : UInt64.max
            #expect(stage.seed == expected)
        }
    }
    @Test func stageSeedRejectsMalformedOrOverflowingStrings() {
        for seed in ["invalid", "-1", "18446744073709551616"] {
            #expect(throws: DecodingError.self) {
                try MoldJSON.decoder.decode(ChainJobStageDetail.self, from: Data("{\"idx\":0,\"state\":\"pending\",\"seed\":\"\(seed)\"}".utf8))
            }
        }
    }
}
